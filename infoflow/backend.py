"""
Optional PyTorch kernels for batched conditional mutual information.

Parent selection evaluates the same conditional mutual information for hundreds of
permutations of one candidate. These kernels evaluate a whole batch of permuted
candidates at once, on a GPU (CUDA or Apple MPS) or vectorized on the CPU:

* :func:`plugin_cmi_batch`: plug-in estimates from symbol counts (``scatter_add``);
* :func:`ksg_cmi_batch`: Frenzel--Pompe/KSG estimates :cite:`Kraskov2004,Frenzel2007`
  by brute-force max-norm distances, with neighbours counted strictly inside each
  point's radius (as in :mod:`dit.inference`).

Install with ``pip install infoflow[gpu]``. MPS has no float64, so on Apple GPUs the
kernels compute in float32; elsewhere they use float64.
"""

import threading
from contextlib import nullcontext

import numpy as np

__all__ = ("ksg_cmi_batch", "plugin_cmi_batch", "resolve_device")


def _torch():
    try:
        import torch
    except ImportError as error:  # pragma: no cover - optional dependency
        raise ImportError("GPU kernels need the optional dependency torch (pip install infoflow[gpu])") from error
    return torch


def resolve_device(device):
    """
    A ``torch.device`` for `device`, or None for the default NumPy/SciPy path.

    ``'auto'`` prefers CUDA, then Apple MPS, then the CPU.
    """
    if device is None:
        return None
    torch = _torch()
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available")
    return device


_MPS_LOCK = threading.Lock()


def _guard(device):
    """
    MPS is not safe to drive from several threads at once; serialize its kernels.
    """
    return _MPS_LOCK if device.type == "mps" else nullcontext()


def _float(device):
    torch = _torch()
    return torch.float32 if device.type == "mps" else torch.float64


def _dense(codes):
    _, inverse = np.unique(codes, return_inverse=True)
    inverse = inverse.reshape(np.shape(codes)).astype(np.int64)
    return inverse, int(inverse.max()) + 1 if inverse.size else 1


def _plugin_cmi_batch(y, Ky, x_batch, Kx, z, Kz, device, max_cells=50_000_000):
    """
    Plug-in :math:`I[Y : X_b \\mid Z]` in bits for every row ``b`` of `x_batch`.

    Parameters
    ----------
    y, z : np.ndarray
        Integer codes, shape (n,), with alphabets `Ky` and `Kz`.
    x_batch : np.ndarray
        Integer codes, shape (B, n), alphabet `Kx`.
    device : torch.device
    max_cells : int
        Largest number of count cells held at once; batches are split to fit.

    Returns
    -------
    np.ndarray
        Shape (B,), clipped at zero.
    """
    torch = _torch()
    x_batch = np.asarray(x_batch)
    B, n = x_batch.shape
    if Kz * Kx * Ky > 2**40:
        z, Kz = _dense(z)
    if Kz * Kx * Ky > 2**40:
        raise ValueError("joint alphabet too large for count tables")
    dtype = _float(device)
    yt = torch.as_tensor(np.asarray(y, dtype=np.int64), device=device)
    zt = torch.as_tensor(np.asarray(z, dtype=np.int64), device=device)
    log2 = float(np.log(2))

    def entropy(code, K):
        # code: (b, n) int64 -> plug-in entropy (nats) per row
        rows = code.shape[0]
        if rows * K <= max_cells:
            counts = torch.zeros(rows, K, dtype=dtype, device=device)
            counts.scatter_add_(1, code, torch.ones(code.shape, dtype=dtype, device=device))
        else:
            flat = code + K * torch.arange(rows, device=device).unsqueeze(1)
            cells, counts = torch.unique(flat, return_counts=True)
            c = counts.to(dtype)
            sums = torch.zeros(rows, dtype=dtype, device=device).index_add_(0, cells // K, c * torch.log(c))
            return float(np.log(n)) - sums / n
        terms = torch.where(counts > 0, counts * torch.log(torch.clamp(counts, min=1)), 0)
        return float(np.log(n)) - terms.sum(1) / n

    h_yz = entropy((yt * Kz + zt).unsqueeze(0), Ky * Kz)[0]
    h_z = entropy(zt.unsqueeze(0), Kz)[0]
    step = max(1, min(B, max_cells // max(Kx * Kz * Ky, 1)))
    out = []
    for lo in range(0, B, step):
        xb = torch.as_tensor(x_batch[lo : lo + step].astype(np.int64), device=device)
        xz = xb * Kz + zt
        yxz = yt * (Kx * Kz) + xz
        value = entropy(xz, Kx * Kz) + h_yz - entropy(yxz, Ky * Kx * Kz) - h_z
        out.append(value.cpu().double().numpy())
    return np.maximum(np.concatenate(out) / log2, 0.0)


def _ksg_cmi_batch(x_batch, y, z, k, device, chunk=None, max_elements=200_000_000):
    """
    KSG :math:`I[Y : X_b \\mid Z]` in bits for every row ``b`` of `x_batch`.

    Parameters
    ----------
    x_batch : np.ndarray
        Shape (B, n, dx).
    y : np.ndarray
        Shape (n,).
    z : np.ndarray
        Shape (n, dz); ``dz = 0`` gives the mutual information.
    k : int
    device : torch.device
    chunk : int, None
        Rows of the distance matrix per step (default: as many as fit `max_elements`).

    Returns
    -------
    np.ndarray
        Shape (B,), not clipped at zero (only the ordering matters in a permutation test).
    """
    torch = _torch()
    from scipy.special import digamma

    x_batch = np.asarray(x_batch, dtype=float)
    if x_batch.ndim == 2:
        x_batch = x_batch[:, :, None]
    B, n, _ = x_batch.shape
    z = np.asarray(z, dtype=float).reshape(n, -1)
    dtype = _float(device)
    inf = float("inf")
    xb = torch.as_tensor(x_batch, dtype=dtype, device=device)
    yz = torch.as_tensor(np.column_stack([y, z]), dtype=dtype, device=device)
    zt = torch.as_tensor(z, dtype=dtype, device=device) if z.shape[1] else None
    if chunk is None:
        chunk = max(1, min(n, max_elements // max(B * n, 1)))
    n_xz = torch.zeros(B, n, dtype=dtype, device=device)
    n_yz = torch.zeros(B, n, dtype=dtype, device=device)
    n_z = torch.zeros(B, n, dtype=dtype, device=device)
    for lo in range(0, n, chunk):
        hi = min(lo + chunk, n)
        d_yz = torch.cdist(yz[lo:hi], yz, p=inf)
        d_x = torch.cdist(xb[:, lo:hi], xb, p=inf)
        eps = torch.topk(torch.maximum(d_x, d_yz), k + 1, dim=-1, largest=False).values[..., -1:]
        n_yz[:, lo:hi] = (d_yz.unsqueeze(0) < eps).sum(-1).to(dtype)
        if zt is not None:
            d_z = torch.cdist(zt[lo:hi], zt, p=inf)
            n_xz[:, lo:hi] = (torch.maximum(d_x, d_z) < eps).sum(-1).to(dtype)
            n_z[:, lo:hi] = (d_z.unsqueeze(0) < eps).sum(-1).to(dtype)
        else:
            n_xz[:, lo:hi] = (d_x < eps).sum(-1).to(dtype)
    n_xz, n_yz, n_z = (t.cpu().double().numpy() for t in (n_xz, n_yz, n_z))
    if zt is not None:
        value = digamma(k) - np.mean(digamma(n_xz) + digamma(n_yz) - digamma(n_z), axis=1)
    else:
        value = digamma(k) + digamma(n) - np.mean(digamma(n_xz) + digamma(n_yz), axis=1)
    return value / np.log(2)


def plugin_cmi_batch(y, Ky, x_batch, Kx, z, Kz, device, max_cells=50_000_000):
    with _guard(device):
        return _plugin_cmi_batch(y, Ky, x_batch, Kx, z, Kz, device, max_cells)


plugin_cmi_batch.__doc__ = _plugin_cmi_batch.__doc__


def ksg_cmi_batch(x_batch, y, z, k, device, chunk=None, max_elements=200_000_000):
    with _guard(device):
        return _ksg_cmi_batch(x_batch, y, z, k, device, chunk, max_elements)


ksg_cmi_batch.__doc__ = _ksg_cmi_batch.__doc__

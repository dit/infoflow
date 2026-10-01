"""
Tests for the batched PyTorch kernels (CPU always; MPS/CUDA when present).
"""

import dataclasses

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from infoflow import datasets  # noqa: E402
from infoflow.backend import ksg_cmi_batch, plugin_cmi_batch, resolve_device  # noqa: E402
from infoflow.data import DiscreteData  # noqa: E402
from infoflow.selection import SkeletonSettings, _cmi, infer_skeleton  # noqa: E402

GPUS = [d for d, ok in (("mps", torch.backends.mps.is_available()), ("cuda", torch.cuda.is_available())) if ok]


def _codes(seed=0, n=3000):
    rng = np.random.default_rng(seed)
    z = rng.integers(0, 40, n)
    x = (z % 5 + rng.integers(0, 2, n)) % 5
    y = (x + z + rng.integers(0, 3, n)) % 6
    perms = np.stack([rng.permutation(n) for _ in range(30)])
    return y, x, z, perms


def test_resolve_device():
    assert resolve_device(None) is None
    assert resolve_device("cpu").type == "cpu"
    assert resolve_device("auto").type in ("cpu", "mps", "cuda")
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match="CUDA"):
            resolve_device("cuda")


@pytest.mark.parametrize("device", ["cpu", *GPUS])
def test_plugin_batch_matches_loop(device):
    y, x, z, perms = _codes()
    batch = np.vstack([x, x[perms]])
    expected = np.array([_cmi(y, 6, row, 5, z, 40) for row in batch])
    got = plugin_cmi_batch(y, 6, batch, 5, z, 40, resolve_device(device))
    assert np.allclose(got, expected, atol=1e-9 if device != "mps" else 1e-5)


def test_plugin_batch_large_alphabet_path():
    y, x, z, perms = _codes()
    batch = x[perms[:4]]
    expected = np.array([_cmi(y, 6, row, 5, z, 40) for row in batch])
    got = plugin_cmi_batch(y, 6, batch, 5, z, 40, resolve_device("cpu"), max_cells=10)
    assert np.allclose(got, expected, atol=1e-9)


@pytest.mark.parametrize("device", ["cpu", *GPUS])
def test_ksg_batch_matches_trees(device):
    from scipy.spatial import KDTree
    from scipy.special import digamma

    rng = np.random.default_rng(1)
    n = 1500
    z = rng.normal(size=(n, 2))
    x = z.sum(1) + rng.normal(size=n)
    y = x + z[:, 0] + rng.normal(size=n)
    batch = np.stack([x, x[rng.permutation(n)]])[:, :, None]
    got = ksg_cmi_batch(batch, y, z, 4, resolve_device(device))

    def reference(xv):
        xyz = np.column_stack([xv, y, z])
        r = np.nextafter(KDTree(xyz).query(xyz, 5, p=np.inf)[0][:, -1], 0)

        def count(a):
            return KDTree(a).query_ball_point(a, r, p=np.inf, return_length=True)

        v = digamma(4) - np.mean(
            digamma(count(np.column_stack([xv, z]))) + digamma(count(np.column_stack([y, z]))) - digamma(count(z))
        )
        return max(v / np.log(2), 0.0)

    expected = np.array([reference(b[:, 0]) for b in batch])
    assert np.allclose(got, expected, atol=1e-9 if device != "mps" else 2e-3)
    assert got[0] > 0.3 and got[1] < 0.05


@pytest.mark.parametrize("device", ["cpu", *GPUS])
def test_selection_on_device_matches_default(device):
    data = DiscreteData.from_discrete([datasets.chain(2000, seed=0)])
    fast = SkeletonSettings(
        n_perm_max_stat=50, n_perm_min_stat=50, n_perm_omnibus=50, n_perm_max_seq=50, n_perm_pairs=50
    )
    default = infer_skeleton(data, settings=fast, prng=0)
    batched = infer_skeleton(data, settings=dataclasses.replace(fast, device=device), prng=0)
    for t in default:
        assert default[t].sources == batched[t].sources
        assert default[t].target_past == batched[t].target_past

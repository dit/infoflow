GPU and batched kernels
=======================

Parent selection evaluates each candidate's conditional mutual information under
hundreds of permutations. With ``device=`` (``infer_multiplex(device=...)`` or
``SkeletonSettings(device=...)``), these permutation nulls are evaluated in batches by
the PyTorch kernels of :mod:`infoflow.backend`, on an NVIDIA GPU (``"cuda"``), an Apple
GPU (``"mps"``), or vectorized on the CPU (``"cpu"``); ``"auto"`` picks the first
available in that order. Install with ``pip install infoflow[gpu]``. The default,
``device=None``, keeps the NumPy/SciPy path and its results unchanged to the bit.

* **Plug-in estimates** are counted with ``scatter_add`` over a whole batch of
  permuted candidates.
* **KSG estimates** :cite:`Kraskov2004,Frenzel2007` use brute-force max-norm
  distances with neighbours counted strictly inside each radius, matching the
  tree-based CPU estimate (exactly in float64; within float32 rounding on MPS,
  which has no float64).

Apple GPUs are not safe to drive from several threads at once, so MPS kernels are
serialized; the CPU-side work of other targets proceeds meanwhile.

Measured speed (Apple M-series, 16 cores; MuTE network, 10 000 samples, plug-in
selection, identical parent sets on every path):

==================  =========
Path                Selection
==================  =========
NumPy (default)     22.1 s
batched, CPU        9.6 s
batched, MPS        4.6 s
==================  =========

Where the GPU does *not* help, measured on the same machine:

* **KSG on Apple GPUs.** Brute force costs :math:`O(n^2)` per estimate and is memory
  bound; at :math:`n = 10\,000` an MPS batch takes about 85 ms per estimate, against
  about 20 ms for the threaded tree search on the CPU (the default KSG path). The
  brute-force kernel is meant for CUDA GPUs and large batches.
* **Layer estimation.** Each bootstrap and null resample of an edge needs its own
  intrinsic-flow optimization (a few thousand small problems per edge, about 2.6 ms
  each with SciPy). Solving them together by L-BFGS on their summed objective was six
  times *slower*, because the joint problem converges only when the slowest one does;
  stopping after a fixed number of steps would leave the null values under-optimized
  relative to the observed ones and bias the synergy and shared tests. Layer
  estimation therefore always runs on the CPU, and dominates plug-in runtimes on small
  networks (90% of a five-node run); selection dominates as networks grow, since its
  candidates grow with the number of processes.

"""
Checkpointing for long analyses: results are written as they complete and reused on resume.
"""

import pickle
from hashlib import sha256
from pathlib import Path

__all__ = ("Checkpoint",)


class Checkpoint:
    """
    A directory of pickled partial results keyed by name.

    Pass the same directory (and the same ``prng`` seed) to
    :func:`~infoflow.infer_multiplex` to resume an interrupted run: finished targets
    and edges are loaded instead of recomputed, and the remaining work uses the same
    random seeds it would have used.
    """

    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def fingerprint(value):
        """Identify a trusted, pickleable analysis configuration and its data."""
        return sha256(pickle.dumps(value, protocol=5)).hexdigest()[:24]

    def scoped(self, configuration):
        """
        A separate cache namespace for a candidate-guided analysis.

        Include actual input data, split/candidate definitions, settings and RNG
        state. Legacy unrestricted keys are intentionally not read in this scope.
        """
        return Checkpoint(self.directory / f"guided-{self.fingerprint(configuration)}")

    def _path(self, key):
        return self.directory / f"{key}.pkl"

    def __contains__(self, key):
        return self._path(key).exists()

    def get(self, key):
        path = self._path(key)
        if not path.exists():
            return None
        with open(path, "rb") as fh:
            return pickle.load(fh)

    def put(self, key, value):
        tmp = self._path(key).with_suffix(".tmp")
        with open(tmp, "wb") as fh:
            pickle.dump(value, fh)
        tmp.replace(self._path(key))
        return value

    def cached(self, key, compute):
        """
        Load `key` if present; otherwise compute, store, and return it.
        """
        value = self.get(key)
        return value if value is not None else self.put(key, compute())

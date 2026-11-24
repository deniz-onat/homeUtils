"""Load BrainChip's preprocessed KWS pickle without executing arbitrary code.

pickletools shows it references only numpy's array reconstructors; any other global is refused.
"""

import pickle

ALLOWED = {
    (core, name) for core in ("numpy.core.multiarray", "numpy._core.multiarray") for name in ("_reconstruct", "scalar")
} | {("numpy", "ndarray"), ("numpy", "dtype")}  # numpy 2 writes numpy._core


class _NumpyOnly(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) not in ALLOWED:
            raise pickle.UnpicklingError(f"refused global {module}.{name}")
        return super().find_class(module, name)


def load(path):
    with open(path, "rb") as f:
        return _NumpyOnly(f).load()


if __name__ == "__main__":
    import io

    import numpy as np

    buf = io.BytesIO(pickle.dumps([np.arange(3, dtype=np.uint8)]))
    assert (_NumpyOnly(buf).load()[0] == np.arange(3)).all()
    try:
        _NumpyOnly(io.BytesIO(pickle.dumps(print))).load()
        raise AssertionError("print was allowed")
    except pickle.UnpicklingError:
        pass
    print("ok")

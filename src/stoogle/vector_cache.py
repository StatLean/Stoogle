"""Content-addressed vectors: reuse only identical model revision and text."""
import hashlib
import os
import tempfile
from pathlib import Path

import numpy as np


def encode_cached(embedder, texts, model, revision, directory: Path | None):
    if directory is None:
        return embedder.encode(texts), {"reused": 0, "encoded": len(texts)}
    directory.mkdir(parents=True, exist_ok=True)
    vectors, missing, keys = {}, [], []
    for i, text in enumerate(texts):
        key = hashlib.sha256((model + "\0" + revision + "\0" + text).encode()).hexdigest()
        keys.append(key)
        path = directory / (key + ".npy")
        if path.exists():
            vector = np.load(path, allow_pickle=False)
            if vector.ndim != 1 or not np.isfinite(vector).all() or not np.isclose(np.linalg.norm(vector), 1, atol=1e-4):
                raise ValueError(f"Invalid cached vector: {path}")
            vectors[i] = vector
        else:
            missing.append(i)
    if missing:
        encoded = embedder.encode([texts[i] for i in missing])
        for i, vector in zip(missing, encoded):
            vectors[i] = vector
            fd, temporary = tempfile.mkstemp(prefix=".vector-", dir=directory)
            try:
                with os.fdopen(fd, "wb") as stream:
                    np.save(stream, vector, allow_pickle=False)
                os.replace(temporary, directory / (keys[i] + ".npy"))
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
    return np.asarray([vectors[i] for i in range(len(texts))], dtype=np.float32), {"reused": len(texts)-len(missing), "encoded": len(missing)}

# config and helpers for manually caching long computations to make the rendering of .qmds/.ipynbs faster
import hashlib
import inspect
import os
import shutil
import warnings
from datetime import datetime

import dill as pickle

cache_root = os.path.join(os.path.abspath(""), "cache")
cache_path = None
caching = True

# Marker for the stamped cache wrapper. Pickles without it are treated as legacy
# (unstamped) payloads and loaded as-is for backward compatibility.
_CACHE_FORMAT = 1


def _cache_meta(func):
    """Environment/source stamp used to detect stale caches across library upgrades."""
    meta = {"format": _CACHE_FORMAT}
    try:
        import jax

        meta["jax"] = jax.__version__
    except Exception:
        meta["jax"] = None
    try:
        from tvboptim import __version__ as tvbo_version

        meta["tvboptim"] = tvbo_version
    except Exception:
        meta["tvboptim"] = None
    try:
        meta["src_hash"] = hashlib.sha256(inspect.getsource(func).encode()).hexdigest()
    except (OSError, TypeError):
        meta["src_hash"] = None
    return meta


def _stale_reason(stored, current):
    """Return a human-readable reason if the stored stamp is incompatible, else None.

    Only compares keys where the stored value is known (not None), so caches written
    by an older version of this util (or in an environment where a version could not
    be determined) are not needlessly invalidated.
    """
    for key in ("jax", "tvboptim", "src_hash"):
        old = stored.get(key)
        new = current.get(key)
        if old is not None and new is not None and old != new:
            return f"{key} changed ({old} -> {new})"
    return None


def set_cache_path(experiment=""):
    """
    Set the path where the cache is stored, relative to the cache root which lies next to this file in /cache.

    * `experiment`: name of the experiment / notebook -> becomes a subdirectory in the cache root
    """
    global cache_path
    cache_path = os.path.join(cache_root, experiment)
    print(f"Cache stored here: {cache_path}")
    return cache_path


def clear_all_cache():
    """
    Clear all caches in the current cache root.
    """
    if os.path.exists(cache_root):
        shutil.rmtree(cache_root)


def clear_experiment_cache():
    """
    Clear the cache for the current experiment / notebook.
    """
    if os.path.exists(cache_path):
        shutil.rmtree(cache_path)


def cache(fname="", redo=False):
    """
    Caching decorator - a local authoring convenience, use with care!

    On the first call the decorated function runs and its return value is pickled to
    `{fname}.pkl` under `cache_path`; later calls load the pickle instead of recomputing.

    The cache is meant to speed up local iteration, not to be a durable artifact: the
    pickles store live JAX/equinox objects and can go stale across library upgrades. To
    stay robust the loader recomputes (instead of crashing) whenever the pickle fails to
    load or its environment stamp (jax/tvboptim version, function source) no longer
    matches. For committed docs, prefer Quarto's freeze over relying on these pickles.

    * `fname`: name of the file where the computation is stored under `cache_path`
    * `redo`: if True, the computation is run even if the file already exists
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            global cache_path
            if cache_path is None:
                raise ValueError(
                    "No cache path set. Use `set_cache_path` to set the path to the cache."
                )
            cache_file = os.path.join(cache_path, f"{fname}.pkl")

            def compute_and_store():
                print(f"Running computations for {fname}")
                result = func(*args, **kwargs)
                if caching:
                    os.makedirs(cache_path, exist_ok=True)
                    with open(cache_file, "wb") as f:
                        # Use protocol 4 for Python 3.4+ compatibility
                        # Avoid HIGHEST_PROTOCOL which changes between versions
                        pickle.dump(
                            {"meta": _cache_meta(func), "payload": result},
                            f,
                            protocol=4,
                        )
                return result

            if redo or not os.path.exists(cache_file):
                return compute_and_store()

            print(
                f"Loading {fname} from cache, last modified {datetime.fromtimestamp(os.path.getmtime(cache_file))}"
            )
            try:
                with open(cache_file, "rb") as f:
                    stored = pickle.load(f)
            except Exception as exc:
                # e.g. ModuleNotFoundError when a pickled library type moved between
                # versions. Recompute instead of failing the whole render.
                warnings.warn(
                    f"Cache '{fname}' could not be loaded ({exc!r}); recomputing.",
                    stacklevel=2,
                )
                return compute_and_store()

            if isinstance(stored, dict) and "meta" in stored and "payload" in stored:
                reason = _stale_reason(stored["meta"], _cache_meta(func))
                if reason is not None:
                    warnings.warn(
                        f"Cache '{fname}' is stale ({reason}); recomputing.",
                        stacklevel=2,
                    )
                    return compute_and_store()
                return stored["payload"]

            # Legacy unstamped pickle that loaded successfully: use as-is.
            return stored

        return wrapper

    return decorator

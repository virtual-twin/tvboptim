# config and helpers for manually caching long computations to make the rendering of .qmds/.ipynbs faster
import dill as pickle
import os
import shutil
from datetime import datetime

cache_root = os.path.join(os.path.abspath(''), "cache")
cache_path = None
caching = True

def set_cache_path(experiment = ""):
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


def cache(fname = "", redo = False):
    """
    Caching decorator - use with care!
    
    Brings variable in global scope and assigns it either a cached version or the result of decorated function.

    * `fname`: name of the file where the computation is stored under `cache_path`
    * `redo`: if True, the computation is run even if the file already exists
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            global cache_path
            if cache_path is None:
                raise ValueError("No cache path set. Use `set_cache_path` to set the path to the cache.")
            cache_file = os.path.join(cache_path, f"{fname}.pkl")
            
            if not redo and os.path.exists(cache_file):
                print(f"Loading {fname} from cache, last modified {datetime.fromtimestamp(os.path.getmtime(cache_file))}")
                with open(cache_file, 'rb') as f:
                    # exec(f"global {variable}\n{variable} = pickle.load(f)")
                    comp = pickle.load(f)
            else:
                print(f"Running computations for {fname}")
                comp = func(*args, **kwargs)
                if caching:
                    if not os.path.exists(cache_path):
                        os.makedirs(cache_path)
                    with open(cache_file, 'wb') as f:
                        # Use protocol 4 for Python 3.4+ compatibility
                        # Avoid HIGHEST_PROTOCOL which changes between versions
                        pickle.dump(comp, f, protocol=4)
                # exec(f"global {fname}\n{fname} = comp")
            return comp
        return wrapper
    return decorator

 
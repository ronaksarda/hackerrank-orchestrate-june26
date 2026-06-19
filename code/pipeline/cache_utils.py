import json
import os
import threading

_CACHE_LOCK = threading.Lock()

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", ".cache")
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

TEXT_CACHE_PATH = os.path.join(CACHE_DIR, "text_cache.json")
VISION_CACHE_PATH = os.path.join(CACHE_DIR, "vision_cache.json")

def load_cache(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}

def save_cache(path, cache_dict):
    with _CACHE_LOCK:
        with open(path, "w") as f:
            json.dump(cache_dict, f, indent=2)

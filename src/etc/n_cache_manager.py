import numpy as np
import os
import json


class cache_manager:
    def __init__(self):
        if not os.path.exists("cache"):
            os.mkdir("cache")

        self.cache = {}

    def set_cache(self, key, value):
        if type(value) == np.ndarray:
            value = value.tolist()
        self.cache[key] = value

    def get_cache(self, key):
        try:
            res = np.array(self.cache[key])
            return res
        except:
            return self.cache[key]

    def save_cache(self):
        with open("cache/cache.json", "w") as json_file:
            json.dump(self.cache, json_file, sort_keys=True, indent=4)

    def load_cache(self):
        if os.path.exists("cache/cache.json"):
            with open("cache/cache.json", "r") as json_file:
                res = json.load(json_file)
            self.cache = res
        else:
            print( "Json Cache File has been lost.")


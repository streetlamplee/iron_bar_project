"""
계산 비용이 큰 결과(주로 카메라 캘리브레이션 값)를 JSON 파일에 저장/재사용하는 도구.

저장 위치는 실행 위치 기준 cache/cache.json 이므로,
항상 프로젝트 루트에서 실행해야 같은 캐시를 재사용할 수 있다.
"""

import numpy as np
import os
import json


class cache_manager:
    def __init__(self):
        # 캐시 폴더가 없으면 만들어 둔다 (저장 시점에 실패하지 않도록).
        if not os.path.exists("cache"):
            os.mkdir("cache")

        self.cache = {}

    def set_cache(self, key, value):
        """값을 메모리 캐시에 넣는다. 실제 파일 저장은 save_cache()에서 이뤄진다."""
        # numpy 배열은 JSON으로 직렬화할 수 없으므로 리스트로 바꿔 저장한다.
        if type(value) == np.ndarray:
            value = value.tolist()
        self.cache[key] = value

    def get_cache(self, key):
        """저장된 값을 읽는다. 배열로 변환 가능하면 numpy 배열로 돌려준다."""
        try:
            res = np.array(self.cache[key])
            return res
        except:
            # 숫자/문자열처럼 배열로 만들 수 없는 값은 원래 형태 그대로 반환
            return self.cache[key]

    def save_cache(self):
        """메모리 캐시를 cache/cache.json 으로 덮어쓴다."""
        with open("cache/cache.json", "w") as json_file:
            json.dump(self.cache, json_file, sort_keys=True, indent=4)

    def load_cache(self):
        """cache/cache.json 을 읽어온다. 파일이 없으면 빈 캐시 상태를 유지한다."""
        if os.path.exists("cache/cache.json"):
            with open("cache/cache.json", "r") as json_file:
                res = json.load(json_file)
            self.cache = res
        else:
            print( "Json Cache File has been lost.")

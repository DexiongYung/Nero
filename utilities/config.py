import json
from collections import OrderedDict


def load_json(jsonpath: str) -> dict:
    with open(jsonpath) as jsonfile:
        return json.load(jsonfile, object_pairs_hook=OrderedDict)


def save_json(jsonpath: str, content):
    with open(jsonpath, 'w') as jsonfile:
        json.dump(content, jsonfile)

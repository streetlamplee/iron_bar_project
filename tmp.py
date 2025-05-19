import json

with open('find-cross-point-model/valid/data.json', 'r') as f:
    j = json.load(f)

l_ = 0

for item in j.values():
    l = len(item['mask'])
    l_ += l

j['len'] = l_

with open('find-cross-point-model/valid/data.json', 'w') as f:
    json.dump(j, f, indent=1)
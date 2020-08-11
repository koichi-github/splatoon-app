import json
import glob
import os
import csv
import random

root_dir = "sample/special/"
json_file_paths = glob.glob(f'{root_dir}json/*.json')
random.shuffle(json_file_paths)
# json_file_paths = sorted(json_file_paths)

train_file = f'{root_dir}train.txt'
test_file = f'{root_dir}test.txt'

train_rate = 0.9

N = len(json_file_paths)
for i, path in enumerate(json_file_paths):
    with open(path) as f:
        json_data = json.load(f)

    img_file = json_data["asset"]["name"]
    label = json_data["regions"][0]["tags"][0]

    if i / N < train_rate:
        with open(train_file, "a") as f:
            writer = csv.writer(f)
            writer.writerow([f"data/{root_dir}images/{img_file}", label])
    else:
        with open(test_file, "a") as f:
            writer = csv.writer(f)
            writer.writerow([f"data/{root_dir}images/{img_file}", label])
        



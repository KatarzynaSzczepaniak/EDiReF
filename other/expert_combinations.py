from collections import defaultdict
import json

my_dict = defaultdict()

with open('expert_combinations.txt', 'r') as f:
    file = f.readlines()

    for idx, combination in enumerate(file):
        my_dict[idx + 1] = combination

with open('expert_combinations.txt', 'w') as fw:
    for key, value in my_dict.items():
        print(key, value)
        fw.write(f'{key}:  {value}')
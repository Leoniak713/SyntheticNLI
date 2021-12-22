import json

from generation import PropertyBasedGenerator, convert_to_json

world_generator = PropertyBasedGenerator()
worlds = [world_generator.generate_world(3) for _ in range(3)]
data_json = convert_to_json(worlds)
print(data_json)
with open('generation_test.json', 'wt') as json_file:
    json.dump(data_json, json_file)

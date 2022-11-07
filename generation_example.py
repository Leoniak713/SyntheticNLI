from SyntheticNLI.generation import IncrementalGenerator, convert_to_json
from SyntheticNLI.dropbox_utils import DropboxClient

world_generator = IncrementalGenerator(
    "http://kb.openrobots.org/", 
    num_triplets = 5, 
    new_graph_prob = 0.1, 
    extend_graph_prob = 0.6, 
    add_edge_prob = 0.3
)
worlds = [world_generator.generate_world() for _ in range(2)]
data_json = convert_to_json(worlds)
print(data_json)
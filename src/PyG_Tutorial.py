import numpy as np
import torch
from torch_geometric.data import Data

print('hi')
edge_index = torch.tensor([[0, 1, 1, 2],
                           [1, 0, 2, 1]], dtype=torch.long)
x = torch.tensor([[-1], [0], [1]], dtype=torch.float)
data = Data(x=x, edge_index=edge_index)
print(data)
print(data.x)
print(data.edge_index)
print(data.num_nodes)
print(data.num_edges)
print(data.num_node_features)
print(data.num_edge_features)


playlist_features = []
for i in range(5):
    track_features = np.array([i, .1, .6, .0, 5, .1, -5.0, 1, .05, 120.0, .9])  # dummy feature data
    playlist_features.append(track_features)

x = np.mean(playlist_features, axis=0)
print(x)

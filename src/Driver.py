import os
from pathlib import Path as Data_Path

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data

from JSONFile import JSONFile

PROJ_DIR="/Users/robel/Documents/GMU/CS782/Project/"
os.chdir(PROJ_DIR)
DATA_DIR=Data_Path('dataset')

N_FILES_TO_USE = 1

file_names = sorted(os.listdir(DATA_DIR))
file_names_to_use = file_names[:N_FILES_TO_USE]
n_playlists = 0

JSONs = []

for file_name in file_names_to_use:
  json_file =  JSONFile(DATA_DIR, file_name, n_playlists)
  json_file.process_file()
  n_playlists += len(json_file.playlists)
  JSONs.append(json_file)
print(JSONFile)

playlist_data = {}
playlists = {}
tracks = {}

for json_file in JSONs:
  # playlists += [{p.name: p.x} for p in json_file.playlists.values()]
  playlists = playlists | {p.name: p.x for p in json_file.playlists.values()}
  # tracks += [{track.uri, track.x} for playlist in json_file.playlists.values() for track in list (playlist.tracks.values())]
  tracks = tracks | {track.uri: track.x for playlist in json_file.playlists.values() for track in list (playlist.tracks.values())} 
  playlist_data = playlist_data | json_file.playlists

G = nx.Graph()

# Simply build a list of all playlist nodes
playlistNodes = []
for p in playlists.keys():
  playlistNodes.append( (p, {"node_type" : "playlist"}) )
G.add_nodes_from(playlistNodes)


# Simply build a list of all track nodes
trackNodes = []
for t in tracks.keys():
  trackNodes.append( (t, {"node_type" : "track"}) )
G.add_nodes_from(trackNodes)

# adding edges
edge_list = []
for p_name, playlist in playlist_data.items():
  # edge_list += [(p_name, t) for t in playlist.tracks]
  for t in playlist.tracks:
    edge_list.append((p_name, t)) 

G.add_edges_from(edge_list)

n_nodes, n_edges = G.number_of_nodes(), G.number_of_edges()

# by sorting them we get an ordering playlist1, ..., playlistN, track1, ..., trackN
sorted_nodes = sorted(list(G.nodes()))

# create dictionaries to index to 0 to n_nodes, will be necessary for when we are using tensors
node2id = dict(zip(sorted_nodes, np.arange(n_nodes)))

# Build x feature matrix
feature_matrix = []
for n in sorted_nodes:
  if 'playlist' in n:
    feature_matrix.append(playlists[n]) 
  else:
    feature_matrix.append(tracks[n]) 

print(G.nodes())
print(G.edges())

G = nx.relabel_nodes(G, node2id)

print(G.nodes())
print(G.edges())

edges = np.array(list(G.edges()), dtype=np.int64)
edge_index = torch.from_numpy(edges).t().contiguous()
edge_index = edge_index.to(torch.long)
num_nodes = int(G.number_of_nodes())
graph_data = Data(x=feature_matrix, edge_index=edge_index, num_nodes=num_nodes)

print(graph_data)
print(graph_data.x)
print(graph_data.edge_index)
print(graph_data.num_nodes)
print(graph_data.num_edges)
print(graph_data.num_node_features)
print(graph_data.num_edge_features)

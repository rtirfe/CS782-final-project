import os
from pathlib import Path as Data_Path

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit

from JSONFile import JSONFile
from MyGNN import MyGNN

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

n_tracks = len(tracks)
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
feature_matrix = torch.tensor(feature_matrix, dtype=torch.float32)
graph_data = Data(x=feature_matrix, edge_index=edge_index, num_nodes=num_nodes)

print(graph_data)
print(graph_data.x)
print(graph_data.edge_index)
print(graph_data.num_nodes)
print(graph_data.num_edges)
print(graph_data.num_node_features)
print(graph_data.num_edge_features)

# convert to train/val/test splits
transform = RandomLinkSplit(
    is_undirected=True,
    add_negative_train_samples=False,
    neg_sampling_ratio=0,
    num_val=0.15, num_test=0.15
)
train_split, val_split, test_split = transform(graph_data)

# note these are stored as float32, we need them to be int64 for future training

# Edge index: message passing edges
train_split.edge_index = train_split.edge_index.type(torch.int64)

val_split.edge_index = val_split.edge_index.type(torch.int64)
test_split.edge_index = test_split.edge_index.type(torch.int64)
# Edge label index: supervision edges
train_split.edge_label_index = train_split.edge_label_index.type(torch.int64)
val_split.edge_label_index = val_split.edge_label_index.type(torch.int64)
test_split.edge_label_index = test_split.edge_label_index.type(torch.int64)

print(f"Validation set has {val_split.edge_label_index.shape[1]} positive supervision edges")
print(f"Train set has {train_split.edge_label_index.shape[1]} positives upervision edges")
print(f"Test set has {test_split.edge_label_index.shape[1]} positive supervision edges")

print(f"Train set has {train_split.edge_index.shape[1]} message passing edges")
print(f"Validation set has {val_split.edge_index.shape[1]} message passing edges")
print(f"Test set has {test_split.edge_index.shape[1]} message passing edges")

# Train
def train(datasets, model, optimizer, args):
  print(f"Beginning training ")

  train_data = datasets["train"]
  val_data = datasets["val"]

  stats = {
      'train': {
        'loss': [],
        'roc' : []
      },
      'val': {
        'loss': [],
        'recall': [],
        'roc' : []
      }

  }
  val_neg_edge = None
  for epoch in range(args["epochs"]): # loop over each epoch
    model.train()
    optimizer.zero_grad()

    neg_edge_index = sample_negative_edges(train_data, n_playlists, n_tracks, args["device"])

    # calculate embedding
    embed = model.get_embedding(train_data.x, train_data.edge_index)
    # calculate pos, negative scores using embedding
    pos_scores = model.predict_link_embedding(embed, train_data.edge_label_index)
    neg_scores = model.predict_link_embedding(embed, neg_edge_index)

    # calculate loss function
    loss = model.recommendation_loss(pos_scores, neg_scores, lambda_reg = 0)

    loss.backward()
    optimizer.step()

    val_loss, val_neg_edge = test(
        model=model, data=val_data, neg_edge_index=val_neg_edge
    )

    print(f"Epoch {epoch}; Train loss {loss}; Val loss {val_loss}")

    if epoch % 10 == 0:
      # calculate recall @ K
      val_recall = recall_at_k(val_data, model, k = 300, device = args["device"])
      print(f"Val recall {val_recall}")
      stats['val']['recall'].append(val_recall)

  return stats

def test(model, data, neg_edge_index = None):

  model.eval()
  with torch.no_grad(): # want to save RAM

    # conduct negative sampling
    neg_edge_index = sample_negative_edges(data, n_playlists, n_tracks, device=None)

    # obtain model embedding
    embed = model.get_embedding(data.x, data.edge_index)

    # calculate pos, neg scores using embedding
    pos_scores = model.predict_link_embedding(embed, data.edge_label_index)
    neg_scores = model.predict_link_embedding(embed, neg_edge_index)

    # calculate loss
    loss = model.recommendation_loss(pos_scores, neg_scores, lambda_reg = 0)

  return loss, neg_edge_index

def recall_at_k(data, model, k = 300, batch_size = 64, device = None):
    with torch.no_grad():
        embeddings = model.get_embedding(data.edge_index)
        playlists_embeddings = embeddings[:n_playlists]
        tracks_embeddings = embeddings[n_playlists:]

    hits_list = []
    relevant_counts_list = []

    for batch_start in range(0, n_playlists, batch_size):
        batch_end = min(batch_start + batch_size, n_playlists)
        batch_playlists_embeddings = playlists_embeddings[batch_start:batch_end]

        # Calculate scores for all possible item pairs
        scores = torch.matmul(batch_playlists_embeddings, tracks_embeddings.t())

        # Set the scores of message passing edges to negative infinity
        mp_indices = ((data.edge_index[0] >= batch_start) & (data.edge_index[0] < batch_end)).nonzero(as_tuple=True)[0]
        scores[data.edge_index[0, mp_indices] - batch_start, data.edge_index[1, mp_indices] - n_playlists] = -float("inf")

        # Find the top k highest scoring items for each playlist in the batch
        _, top_k_indices = torch.topk(scores, k, dim=1)

        # Ground truth supervision edges
        ground_truth_edges = data.edge_label_index

        # Create a mask to indicate if the top k items are in the ground truth supervision edges
        mask = torch.zeros(scores.shape, device=device, dtype=torch.bool)
        gt_indices = ((ground_truth_edges[0] >= batch_start) & (ground_truth_edges[0] < batch_end)).nonzero(as_tuple=True)[0]
        mask[ground_truth_edges[0, gt_indices] - batch_start, ground_truth_edges[1, gt_indices] - n_playlists] = True

        # Check how many of the top k items are in the ground truth supervision edges
        hits = mask.gather(1, top_k_indices).sum(dim=1)
        hits_list.append(hits)

        # Calculate the total number of relevant items for each playlist in the batch
        relevant_counts = torch.bincount(ground_truth_edges[0, gt_indices] - batch_start, minlength=batch_end - batch_start)
        relevant_counts_list.append(relevant_counts)

    # Compute recall@k
    hits_tensor = torch.cat(hits_list, dim=0)
    relevant_counts_tensor = torch.cat(relevant_counts_list, dim=0)
    # Handle division by zero case
    recall_at_k = torch.where(
        relevant_counts_tensor != 0,
        hits_tensor.true_divide(relevant_counts_tensor),
        torch.ones_like(hits_tensor)
    )
    # take average
    recall_at_k = torch.mean(recall_at_k)

    if recall_at_k.numel() == 1:
        return recall_at_k.item()
    else:
        raise ValueError("recall_at_k contains more than one item.")

def sample_negative_edges(data, num_playlists, num_tracks, device=None):
    positive_playlists, positive_tracks = data.edge_label_index

    # Create a mask tensor with the shape (num_playlists, num_tracks)
    mask = torch.zeros(num_playlists, num_tracks, device=device, dtype=torch.bool)
    mask[positive_playlists, positive_tracks - num_playlists] = True

    # Flatten the mask tensor and get the indices of the negative edges
    flat_mask = mask.flatten()
    negative_indices = torch.where(~flat_mask)[0]

    # Sample negative edges from the negative_indices tensor
    sampled_negative_indices = negative_indices[
        torch.randint(0, negative_indices.size(0), size=(positive_playlists.size(0),), device=device)
    ]

    # Convert the indices back to playlists and tracks tensors
    playlists = torch.floor_divide(sampled_negative_indices, num_tracks)
    tracks = torch.remainder(sampled_negative_indices, num_tracks)
    tracks = tracks + num_playlists

    neg_edge_index = torch.stack((playlists, tracks), dim=0)
    # neg_edge_label = torch.zeros(neg_edge_index.shape[1], device=device)

    # return neg_edge_index, neg_edge_label
    return neg_edge_index

# create a dictionary of the dataset splits
datasets = {
    'train':train_split,
    'val':val_split,
    'test': test_split
  }

# initialize our arguments
args = {
    'device' : 'cuda' if torch.cuda.is_available() else 'cpu',
    'num_layers' :  3,
    'emb_size' : 64,
    'weight_decay': 1e-5,
    'lr': 0.01,
    'epochs': 301
  }

# initialize model and and optimizer
num_nodes = n_playlists + n_tracks
model = MyGNN(
    num_nodes = num_nodes, 
    num_layers = args['num_layers'],
    embedding_dim = args['emb_size'], 
)

# model.initialize_embeddings(graph_data)
optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])

sage_stats = train(datasets, model, optimizer, args)
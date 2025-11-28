import numpy as np

from Track import Track 
from feature_store import get_track_features

class Playlist:
  """
  Simple class for a playlist, containing its attributes:
    1. Name (playlist and its associated index)
    2. Title (playlist title in the Spotify dataset)
    3. Loaded dictionary from the raw json for the playlist
    4. Dictionary of tracks (track_uri : Track), populated by .load_tracks()
    5. List of artists uris
  """

  def __init__(self, json_data, index):

    self.name = f"playlist_{index}"
    self.title = json_data["name"]
    self.data = json_data['tracks']

    self.tracks = {}
    self.artists = []
    self.x = []

  def load_tracks(self):
    """ Call this function to load all of the tracks in the json data for the playlist."""
    tracks_list = self.data
    playlist_features = []
    # self.tracks = {x["track_uri"] : Track(x, self.name) for x in tracks_list}
    for t in tracks_list:
      track_features = get_track_features(t['track_uri'])
      self.tracks[t["track_uri"]] = Track(t, self.name, track_features)
      playlist_features.append(track_features)

    self.artists = [x["artist_uri"] for x in tracks_list]
    self.x = np.mean(playlist_features, axis=0)

  def __str__(self):
    return f"Playlist {self.name} with {len(self.tracks)} tracks loaded."

  def __repr__(self):
    return f"Playlist {self.name}"

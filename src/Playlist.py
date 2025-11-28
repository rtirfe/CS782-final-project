import numpy as np
from Track import Track 

X_DB = {
  "spotify:track:0WqIKmW4BTrj3eJFmnCKMv" : np.array([.1, .1, .6, .0, 5, .1, -5.0, 1, .05, 120.0, .9]),
  "spotify:track:6I9VzXrHxO9rA9A5euc8Ak" : np.array([.2, .1, .6, .0, 5, .1, -5.0, 1, .05, 120.0, .9]),
  "spotify:track:0UaMYEvWZi0ZqiDOoHU3YI" : np.array([.3, .1, .6, .0, 5, .1, -5.0, 1, .05, 120.0, .9]),

  "spotify:track:2HHtWyy5CgaQbC7XSoOb0e" : np.array([.11, .1, .6, .0, 5, .1, -5.0, 1, .05, 120.0, .9]),
  "spotify:track:1MYYt7h6amcrauCOoso3Gx" : np.array([.29, .1, .6, .0, 5, .1, -5.0, 1, .05, 120.0, .9]),
  "spotify:track:3x2mJ2bjCIU70NrH49CtYR" : np.array([.3, .1, .6, .0, 5, .1, -5.0, 1, .05, 120.0, .9]),
  "spotify:track:1Pm3fq1SC6lUlNVBGZi3Em" : np.array([.3, .1, .6, .0, 5, .1, -5.0, 1, .05, 120.0, .9]),
} 

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
      #TODO: UPDATE TO READ Track.feature DATA
      """ 
    • Danceability – [float] describes how suitable a track is for dancing
    • Acousticness – [float] whether the track is acoustic (0 to 1)
    • Energy – [float] a perceptual measure of intensity and activity
    • Instrumentalness – [float] Predicts whether a track contains no vocals
    • Key – [integer] The key the track is in
    • Liveness – [float] Detects the presence of an audience in the recording
    • Loudness – [float] The overall loudness of a track in decibels (dB).
    • Mode – [integer] indicates the modality (major/1 or minor/0) of a track
    • Speechiness – [float] detects the presence of spoken words in a track
    • Tempo – [float] the overall estimated tempo of a track in beats/minute
    • Valence – [float] the musical positiveness conveyed by a track 
      """
      track_features = X_DB[t["track_uri"]]
      self.tracks[t["track_uri"]] = Track(t, self.name, track_features)
      playlist_features.append(track_features)

    self.artists = [x["artist_uri"] for x in tracks_list]
    self.x = np.mean(playlist_features, axis=0)

  def __str__(self):
    return f"Playlist {self.name} with {len(self.tracks)} tracks loaded."

  def __repr__(self):
    return f"Playlist {self.name}"

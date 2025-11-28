import json
import os
from os.path import join
from pathlib import Path as Data_Path

import numpy as np

""" 
    • Acousticness - [float] whether the track is acoustic (0 to 1)
    • Danceability - [float] describes how suitable a track is for dancing (0 to 1)
    • Energy - [float] a perceptual measure of intensity and activity (0 to 1)
    • Instrumentalness - [float] Predicts whether a track contains no vocals (0 to 1)
    • Key - [integer] The key the track is in (-1 to 11)
    • Liveness - [float] Detects the presence of an audience in the recording (0 to 1)
    • Loudness - [float] The overall loudness of a track in decibels (dB). (-60 to 0 dB)
    • Mode - [integer] indicates the modality (major/1 or minor/0) of a track (1 or 0)
    • Speechiness - [float] detects the presence of spoken words in a track (0 to 1)
    • Tempo - [float] the overall estimated tempo of a track in beats/minute (~50 to ~200 BPM)
    • Valence - [float] the musical positiveness conveyed by a track (0 to 1)
"""

# Shared feature store for all modules.
X_DB = {}
X_PATH = Data_Path("resources/features")


def load_X_DB():
    if X_DB:
        return X_DB

    for file_name in os.listdir(X_PATH):
        with open(join(X_PATH, file_name)) as json_file:
            json_data = json.load(json_file)
            for track_feature in json_data['content']:
                id = "spotify:track:" + track_feature['href'].split('/')[-1]
                X_DB[id] = np.array( [
                    track_feature["acousticness"],
                    track_feature["danceability"],
                    track_feature["energy"],
                    track_feature["instrumentalness"],
                    track_feature["key"],
                    track_feature["liveness"],
                    track_feature["loudness"],
                    track_feature["mode"],
                    track_feature["speechiness"],
                    track_feature["tempo"],
                    track_feature["valence"],
                ],
                dtype=float,
            )

    return X_DB

def get_track_features(track_uri):
    """ Retrieve the feature vector for a given track URI. """
    if not X_DB:
        load_X_DB()

    track_features = X_DB.get(track_uri)
    if track_features is None:
        rng = np.random.default_rng()
        track_features = np.array(
            [
                rng.uniform(0.0, 1.0),  # acousticness
                rng.uniform(0.0, 1.0),  # danceability
                rng.uniform(0.0, 1.0),  # energy
                rng.uniform(0.0, 1.0),  # instrumentalness
                float(rng.choice([-1] + list(range(12)))),  # key
                rng.uniform(0.0, 1.0),  # liveness
                rng.uniform(-60.0, 0.0),  # loudness (dB)
                float(rng.integers(0, 2)),  # mode (0/1)
                rng.uniform(0.0, 1.0),  # speechiness
                rng.uniform(50.0, 200.0),  # tempo (BPM)
                rng.uniform(0.0, 1.0),  # valence
            ],
            dtype=float,
        )
        print(f"Missing {track_uri}: Random features: {track_features}") 
    return track_features

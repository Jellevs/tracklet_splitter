import numpy as np
from collections import Counter
from utils.config import SplitterConfig


class JerseySplitter():
    def __init__(self, config=None):
        """ Initialize JerseySplitter with configuration """
        self.config = config if config else SplitterConfig()


    def split_tracklet(self, tracklet, next_available_id):
        """ Split tracklet at persistent jersey switches """

        jerseys = tracklet.pred_attributes.get('jerseys', [])

        if not jerseys or len(jerseys) == 0:
            return None

        # Detect switch points
        switch_indices = self.detect_id_switch(jerseys)

        if not switch_indices:
            return None

        # Create fragment boundaries
        boundaries = [0] + switch_indices + [len(jerseys)]

        fragments = []
        current_id = next_available_id

        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1] - 1

            # Check minimum fragment size
            fragment_length = end - start + 1
            if fragment_length < self.config.jersey_min_fragment:
                # Skip fragments that are too short
                continue

            # Extract sub-tracklet
            fragment = tracklet.extract(start, end)
            fragment.track_id = current_id
            fragment.parent_id = tracklet.parent_id

            fragments.append(fragment)
            current_id += 1

        # If filtering removed all fragments, return None
        if len(fragments) == 0:
            return None

        # If only one fragment remains, no split occurred
        if len(fragments) == 1:
            return None

        return fragments


    
    def detect_id_switch(self, jerseys):
        """ Detect switch points where jersey identity changes persistently """
        switch_points = []
        n = len(jerseys)

        current_jersey = None
        i = 0
        while i < n:
            if self.is_valid_number(jerseys[i]):
                candidate = jerseys[i]

                # Check if first value is actually persistent and not noise
                if self.is_persistent_switch(jerseys, i, candidate):
                    current_jersey = candidate
                    break

            i += 1

        if current_jersey is None:
            return []

        # Scan through predictions
        while i < n:
            if not self.is_valid_number(jerseys[i]):
                i += 1
                continue

            # Check if this is a different number
            if jerseys[i] != current_jersey:
                candidate = jerseys[i]

                # Verify persistence using a window
                if self.is_persistent_switch(jerseys, i, candidate):
                    switch_points.append(i)
                    current_jersey = candidate

            i += 1

        return switch_points



    def is_valid_number(self, value):
        """ Check if jersey value is valid """
        if value is None:
            return False
        if isinstance(value, float) and np.isnan(value):
            return False
        return True


    def is_persistent_switch(self, jerseys, start_idx, new_jersey):
        """ Check if new jersey persists in the lookahead window """
        lookahead = self.config.jersey_lookahead
        min_persistence = self.config.jersey_min_persistence

        # Extract window
        window_end = min(start_idx + lookahead, len(jerseys))
        window = [j for j in jerseys[start_idx:window_end] if self.is_valid_number(j)]

        # Count occurrences of new jersey
        new_count = sum(1 for j in window if j == new_jersey)

        # Must appear at least min_persistence times
        if new_count < min_persistence:
            return False

        # Should be the dominant value in the window (filters out noise)
        if len(window) > 0:
            counts = Counter(window)
            most_common_jersey = counts.most_common(1)[0][0]
            return most_common_jersey == new_jersey

        return new_count >= min_persistence

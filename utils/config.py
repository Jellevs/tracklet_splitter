from dataclasses import dataclass
from pathlib import Path


@dataclass
class Paths:
    img_path: Path
    gt_detections_path: Path
    output_path: Path
    cache_path: Path
    legibility_model_path: Path
    reid_model_path: Path
    parseq_model_path: Path
    centroid_reid_path: Path
    siglip_model_path: Path
    vitpose_model_path: Path
    sequence: str

    def set_cache_path(self, name, sequence):
        return self.cache_path / f"cache_{name}_{sequence}.pkl"
    

@dataclass
class TrackerConfig:
    track_thresh: float = 0.6
    track_low_thresh: float = 0.3
    new_track_thresh: float = 0.4
    track_buffer: int = 60
    match_thresh: float = 0.8
    proximity_thresh: float = 0.5
    appearance_thresh: float = 0.2
    with_reid: bool = True
    reid_model_name: str = "osnet_x0_25"
    frame_rate: int = 25


@dataclass
class JerseyPredictorConfig:
    use_legibility: bool = True
    legibility_arch: str = "resnet34"
    legibility_threshold: float = 0.3
    use_pose_cropper: bool = True
    use_reid_filter: bool = True
    reid_threshold_std: float = 2.0
    debug_tracklet_id: int = None
    debug_dir: Path = None


@dataclass
class SplitterConfig:
    """Configuration for tracklet splitting."""

    # Jersey splitter (frame-by-frame with persistence)
    jersey_min_fragment: int = 20   # Minimum frames for a valid fragment
    jersey_min_persistence: int = 5  # New number must appear 3+ times to confirm switch
    jersey_lookahead: int = 20       # Look ahead 20 frames to check persistence

    # Trajectory refinement
    use_trajectory_refinement: bool = True
    jersey_backtrack_window: int = 50    # How far back to look for trajectory signals
    velocity_jump_threshold: float = 2.0  # Multiplier for velocity change detection
    direction_change_threshold: float = 90  # Degrees for direction change

    # Team splitter (for later)
    team_window_size: int = 15
    team_min_fragment: int = 20
    team_switch_threshold: float = 0.6
    team_switch_distance: int = 10
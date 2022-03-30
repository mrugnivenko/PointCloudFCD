"""This module contains DTO for project configuration."""
import typing as tp

import yaml
from pydantic import BaseModel


class LocalAggregationConfig(BaseModel):
    """Holds settings of local aggregation operator."""

    position_embedding: str
    reduction: str
    output_conv: bool


class Config(BaseModel):

    """Holds setting for data."""

    path_to_data: str
    path_to_allowed_subjects: str
    subjects: list
    path_to_folds: str
    features: list
    MEANS: dict
    STDS: dict
    brains_modality: str
    size: tuple
    crop_size: int
    batch_size: int
    num_points: int
    x_angle_range: float
    y_angle_range: float
    z_angle_range: float
    scale_low: float
    scale_high: float
    noise_std: float
    noise_clip: float
    translate_range: float
    color_drop: float
    augment_symmetries: list
    in_radius: float
    num_steps: int
    datasets: str
    data_root: str
    num_parts: list
    input_features_dim: int
    num_classes: int

    """Holds settings for DL model."""
    backbone: str
    head: str
    radius: float
    sampleDl: float
    density_parameter: float
    nsamples: list
    npoints: list
    width: int
    depth: int
    bottleneck_ratio: int
    bn_momentum: float

    """Holds settings of training procedure."""
    is_experiment: bool
    name_of_experiment: str
    epochs: int
    start_epoch: int
    device: int
    num_workers: int
    is_return_absolute_coordinates: bool
    get_rid_of_air_points: bool
    coin_flip_threshold: float
    loss: str
    weighted_loss: bool
    base_learning_rate: float
    lr_scheduler: str
    optimizer: str
    warmup_epoch: int
    warmup_multiplier: int
    lr_decay_steps: int
    lr_decay_rate: float
    weight_decay: int
    momentum: float
    grid_clip_norm: int

    """Holds settings of training procedure."""
    load_path: str
    print_freq: int
    save_freq: int
    val_freq: int
    log_dir: str
    local_rank: int
    amp_opt_level: str
    rng_seed: int

    """Holds settings of local aggregation operator."""
    local_aggregation_type: str
    pospool: LocalAggregationConfig


def read_config(path_to_cfg: str) -> Config:
    """
    Parse .YAML file with project options and build options object.

    Parameters:
        path_to_cfg: Path to configuration .YAML file.

    Returns:
        Options serialized in object.
    """
    with open(path_to_cfg, "r") as yf:
        yml_file = yaml.safe_load(yf)
    return Config.parse_obj(yml_file)

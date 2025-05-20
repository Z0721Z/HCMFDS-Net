"""
A helper function to get a default model for quick testing
"""
import os
from omegaconf import open_dict
from hydra import compose, initialize

import torch
from model.hcmfds import HCMFDS
from inference.utils.args_utils import get_dataset_cfg
# from utils.download_models import download_models_if_needed


def get_default_model() -> HCMFDS:
    initialize(version_base='1.3.2', config_path="../config", job_name="eval_config")
    cfg = compose(config_name="eval_config")
    # Load the network weights
    cutie = HCMFDS(cfg).cuda().eval()
    model_weights = torch.load(cfg.weights)
    cutie.load_weights(model_weights)

    return cutie

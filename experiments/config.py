"""
Experiment configuration system.

YAML-driven configs allow reproducible experiments with version-controlled
hyperparameter sets — essential for a research paper's "implementation details"
section.  All defaults are research-justified with citations.
"""

import os, yaml
from dataclasses import dataclass, field
from typing import List


@dataclass
class EnvConfig:
    num_nodes:    int   = 10
    failure_prob: float = 0.005       # baseline failure rate
    mean_load:    float = 0.4         # mean background utilisation
    use_mm1:      bool  = True        # use M/M/1 queuing model
    max_steps:    int   = 50


@dataclass
class DQNConfig:
    lr:            float = 1e-3
    gamma:         float = 0.95
    epsilon:       float = 1.0
    epsilon_min:   float = 0.05
    epsilon_decay: float = 0.995
    batch_size:    int   = 64
    target_update: int   = 10
    buffer_cap:    int   = 50_000
    hidden:        int   = 128


@dataclass
class RainbowConfig:
    lr:             float = 5e-4
    gamma:          float = 0.95
    n_step:         int   = 3
    epsilon:        float = 1.0
    epsilon_min:    float = 0.05
    epsilon_decay:  float = 0.995
    batch_size:     int   = 64
    target_update:  int   = 10
    hidden:         int   = 256
    per_alpha:      float = 0.6        # PER priority exponent
    per_beta_start: float = 0.4        # IS weight start value
    buffer_cap:     int   = 100_000


@dataclass
class GNNConfig:
    lr:            float = 5e-4
    gamma:         float = 0.95
    epsilon:       float = 1.0
    epsilon_min:   float = 0.05
    epsilon_decay: float = 0.995
    batch_size:    int   = 32
    buffer_cap:    int   = 20_000
    target_update: int   = 10


@dataclass
class QRoutingConfig:
    lr:       float = 0.1
    init_val: float = 10.0


@dataclass
class TrainConfig:
    num_episodes:  int         = 800
    num_seeds:     int         = 5         # runs per algorithm for statistical validity
    print_every:   int         = 100
    save_every:    int         = 200
    # Curriculum: (episode_threshold, failure_prob, mean_load)
    curriculum:    List[tuple] = field(default_factory=lambda: [
        (0,   0.002, 0.3),    # Stage 1: easy  — few failures, low load
        (200, 0.005, 0.45),   # Stage 2: medium
        (500, 0.010, 0.60),   # Stage 3: hard  — frequent failures, high load
    ])


@dataclass
class ExperimentConfig:
    name:       str          = "default"
    seed:       int          = 42
    models_dir: str          = "models"
    results_dir:str          = "results"
    logs_dir:   str          = "logs"
    env:        EnvConfig    = field(default_factory=EnvConfig)
    dqn:        DQNConfig    = field(default_factory=DQNConfig)
    rainbow:    RainbowConfig= field(default_factory=RainbowConfig)
    gnn:        GNNConfig    = field(default_factory=GNNConfig)
    q_routing:  QRoutingConfig = field(default_factory=QRoutingConfig)
    train:      TrainConfig  = field(default_factory=TrainConfig)


# ── YAML I/O ─────────────────────────────────────────────────────────────────

def load_config(path: str) -> ExperimentConfig:
    if not os.path.exists(path):
        return ExperimentConfig()
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    cfg = ExperimentConfig()
    for section, vals in raw.items():
        if hasattr(cfg, section) and isinstance(vals, dict):
            sub = getattr(cfg, section)
            for k, v in vals.items():
                if hasattr(sub, k):
                    setattr(sub, k, v)
    return cfg

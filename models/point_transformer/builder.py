"""
Model Builder

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

from .utils import Registry

MODELS = Registry('models')
MODULES = Registry('modules')


def build_model(cfg):
    """Build models."""
    return MODELS.build(cfg)
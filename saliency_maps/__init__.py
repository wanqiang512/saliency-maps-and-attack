"""
@author: Britney(wanqiang512)
@software: PyCharm
@file: __init__.py
@time: 2023/10/15 23:22
"""
from saliency_maps.core import CAM, GradCAM, GradCAMpp, SmoothGradCAMpp, ScoreCAM, LayerCAM, RISE, EigenCAM

try:
    from .version import __version__
except ImportError:
    pass

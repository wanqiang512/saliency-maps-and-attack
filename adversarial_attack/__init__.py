"""
@author: Britney(wanqiang512)
@software: PyCharm
@file: __init__.py
@time: 2023/10/15 23:15
"""
from adversarial_attack import AAAM, FIA, FGSM, IFGSM, PGD, PIM, RPA, NAA, SSA, TAIG, TIM, test, SFVA, DIM, SVD, SIM, \
    MIFGSM

try:
    from .version import __version__
except ImportError:
    pass

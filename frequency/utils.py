"""
@author: Britney(wanqiang512)
@software: PyCharm
@file: utils.py
@time: 2023/9/9 22:45
"""
import numpy as np


def fft(img):
    return np.fft.fft2(img)


def fftshift(img):
    return np.fft.fftshift(fft(img))


def ifft(img):
    return np.fft.ifft2(img)


def ifftshift(img):
    return ifft(np.fft.ifftshift(img))


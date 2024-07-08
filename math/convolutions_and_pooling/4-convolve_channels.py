#!/usr/bin/env python3
"""
Convolution with channels
"""

import numpy as np

def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """
    Function that performs a convolution on images with channels:
    """
    m, height, width, c = images.shape
    kh, kw, kc = kernel.shape
    sh, sw = stride
    if padding == 'same':
        ph = ((((height - 1) * sh) + kh - height) // 2) + 1
        pw = ((((width - 1) * sw) + kw - width) // 2) + 1
    elif padding == 'valid':
        ph = 0
        pw = 0
    else:
        ph, pw = padding
    images = np.pad(images, ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                    'constant', constant_values=0)
    ch = ((height + (2 * ph) - kh) // sh) + 1
    cw = ((width + (2 * pw) - kw) // sw) + 1
    convoluted = np.zeros((m, ch, cw))

    for i, h in enumerate(range(0, (height + (2 * ph) - kh + 1), sh)):
        for j, w in enumerate(range(0, (width + (2 * pw) - kw + 1), sw)):
            output = np.sum(images[:, h: h + kh, w: w + kw, :] * kernel,
                            axis=(1, 2, 3))
            convoluted[:, i, j] = output

    return convoluted
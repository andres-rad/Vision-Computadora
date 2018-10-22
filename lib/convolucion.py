import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
from scipy import misc

def convolucion(img, mask):
    k_x = len(mask)
    k_y = len(mask[0])
    off_x = range(-(k_x//2),k_x//2+(k_x % 2))
    off_y = range(-(k_y//2),k_y//2+(k_y % 2))

    res = np.zeros(img.shape)
    for x in range(len(img)):
        for y in range(len(img[0])):
            val = 0
            for o_x in off_x:
                for o_y in off_y:
                    i = (x+o_x) % len(img)
                    j = (y+o_y )% len(img[0])
                    m_i = o_x+k_x//2
                    m_j = o_y+k_y//2
                    val += img[i][j]*mask[m_i][m_j]
            res[x][y] = min(255,val)
    return res

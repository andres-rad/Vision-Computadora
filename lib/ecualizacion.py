from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt
import histosRGB

def ecualizacion (img):
    njs = np.zeros(256) # contendra la dist acumulada de la imagen
    # Cuento la frecuencia absoluta de grises
    for l in img:
        for p in l:
            njs[p]+=1

    njs/=np.size(img)
    for i in range(1,256):
        # voy calculando la acumulada
        njs[i]+=njs[i-1]

    smin = min(njs)
    for i in range(len(img)):
        for j in range(len(img[i])):
            # sabemos que el pixel s de salida es la acumulada en r = img[i]
            s = njs[img[i][j]]
            # transformamos al espacio de grises 0...255
            img[i][j] = int((s-smin)/(1-smin)*(256-1)+0.5)
    return img


def ecualizacionRGB(img):
    njs_r = np.zeros(256) # contendra la dist acumulada de la imagen
    njs_g = np.zeros(256) # contendra la dist acumulada de la imagen
    njs_b = np.zeros(256) # contendra la dist acumulada de la imagen
    # Cuento la frecuencia absoluta de grises
    for l in img:
        for p in l:
            njs_r[p[0]]+=1
            njs_g[p[1]]+=1
            njs_b[p[2]]+=1

    njs_r/=np.size(img)/3
    njs_g/=np.size(img)/3
    njs_b/=np.size(img)/3
    for i in range(1,256):
        # voy calculando la acumulada
        njs_r[i]+=njs_r[i-1]
        njs_g[i]+=njs_g[i-1]
        njs_b[i]+=njs_b[i-1]

    smin_r = min(njs_r)
    smin_g = min(njs_g)
    smin_b = min(njs_b)
    for i in range(len(img)):
        for j in range(len(img[i])):
            # sabemos que el pixel s de salida es la acumulada en r = img[i]
            # sabemos que el pixel s de salida es la acumulada en r = img[i]
            s_r = njs_r[img[i][j][0]]
            s_g = njs_g[img[i][j][1]]
            s_b = njs_b[img[i][j][2]]
            # transformamos al espacio de grises 0...255
            img[i][j][0] = int((s_r-smin_r)/(1-smin_r)*(256-1)+0.5)
            img[i][j][1] = int((s_g-smin_g)/(1-smin_g)*(256-1)+0.5)
            img[i][j][2] = int((s_b-smin_b)/(1-smin_b)*(256-1)+0.5)
    return img

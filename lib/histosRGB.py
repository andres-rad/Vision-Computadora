from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt

def histos_RGB(img,fig):
    RR = np.zeros(256)
    GG = np.zeros(256)
    BB = np.zeros(256)
    for l in img:
        for p in l:
            RR[p[0]]+=1
            GG[p[1]]+=1
            BB[p[2]]+=1
    rgbs = plt.figure(fig)
    plt.plot(RR,'r',label='R')
    plt.plot(GG,'g',label='G')
    plt.plot(BB,'b',label='B')
    plt.legend()
    rgbs.show()

def histos_HSI(img,fig):
    HH = np.zeros(256)
    SS = np.zeros(256)
    II = np.zeros(256)
    for l in img:
        for p in l:
            HH[p[0]]+=1
            SS[p[1]]+=1
            II[p[2]]+=1
    rgbs = plt.figure(fig)
    plt.plot(HH,label='H')
    plt.plot(SS,label='S')
    plt.plot(II,label='I')
    plt.legend()
    rgbs.show()

def histo_gris(img,fig):
    pp = np.zeros(256)
    for l in img:
        for p in l:
            pp[p]+=1
    grises = plt.figure(fig)
    plt.plot(pp)
    grises.show()

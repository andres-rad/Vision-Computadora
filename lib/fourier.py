import numpy as np
from scipy import misc
from scipy.fftpack import ifftn, fftn
from matplotlib import pyplot as plt
import sys
import math

# convolucion
def convolve(I, m):
# def convolve(I, m, z_padding = True):
    # I es la imagen
    image_width = I.shape[1]
    image_heigth = I.shape[0]
    # m es la mascara, debe ser de m x n
    # size de la mascara
    kern_m = m.shape[0]
    kern_n = m.shape[1]
    # kernel center
    kern_m_center = int((kern_m-1)/2)
    kern_n_center = int((kern_n-1)/2)
    # generate arrays to index
    kern_m_indexes = [a-kern_m_center for a in range(0, kern_m)]
    # kern_m_indexes = range(0, kern_m)
    # kern_m_indexes = map(lambda x: x - kern_m_center,kern_m_indexes)


    kern_n_indexes = [a-kern_n_center for a in range(0, kern_n)]
    # kern_n_indexes = range(0, kern_n)
    # kern_n_indexes = map(lambda x: x - kern_n_center,kern_n_indexes)


    res_image = np.zeros(I.shape)

    for i in range(0, I.shape[0]):
        for j in range(0, I.shape[1]):
            # pixel I(i,j)
            local_sum = 0
            for row_index in kern_m_indexes:
                # if i+row_index in range(kern_m_indexes[0],kern_m_center):
                    # #TODO: doy con borde sup
                    # IM = I[i+row_index,range((j+kern_n_indexes[0]),(j+kern_n_indexes[-1])+1)]
                    # continue
                # elif i+row_index in \
                        # range(image_heigth-kern_m_center,image_heigth+kern_m_indexes[-1]):
                    # #TODO: doy con borde inf
                    # continue
                if i+row_index < 0 or i+row_index > image_heigth-1:
                    continue
                else:
                    # sin borde arriba o abajo, en cuyo caso se anula, ya que
                    # lo estoy tratando por filas
                    # me armo un vector por finla que se multiplica, y uso prod interno
                    KERN = m[kern_m_center+row_index, :]
                    if j in range(0,kern_n_center):
                        #TODO: doy con borde izq
                        # IM = [0]*cantidad que se va de rango + pedazo de la imagen que no se va
                        IM = np.append([0]*(kern_n_center-j), I[i+row_index,0:(kern_n-(kern_n_center-j))])
                    elif j in range(image_width-1-kern_n_center+1, image_width):
                        #[WIDTH - center , WIDTH -1]
                        #TODO: doy con borde der
                        # IM = pedazo de la imagen que no se va + [0]*cantidad que se va de rango
                        # image_width-1-j == 0 si j == image_width-1
                        # image_width-1-j == pixels que faltan hasta ultima posicion
                        IM = np.append(I[i+row_index,range(j+kern_n_indexes[0], image_width)], [0]*(kern_n_center-(image_width-1-j)))
                    else:
                        # no doy con borde a los costados
                        IM = I[i+row_index,range((j+kern_n_indexes[0]),(j+kern_n_indexes[-1])+1)]
                        # print("Accediendo a %d, [%d, %d]" % (i+row_index,\
                                # (j+kern_n_indexes[0]),(j+kern_n_indexes[-1])))
                        # print(IM)
                    try:
                        local_sum+= np.inner(IM, KERN)
                    except:
                        print("kern_n_center: %d" % (kern_n_center))
                        print("i=%d | j=%d | row_index= %d" % (i,j,row_index))
                        print("IM = ", IM)
                        print("KERN = ", KERN)
                        sys.exit(1)
            res_image[i,j] = local_sum
    return res_image

def DFT(f):
    N = len(f)
    F = [None]*N

    for k in range(N):
        suma = 0
        for n in range(N):
            suma += f[n]*np.exp(-2j*math.pi*n*k/N)
        F[k] = suma/np.sqrt(N)
    return F

def IDFT(F):
    N = len(F)
    f = [None]*N
    for k in range(N):
        suma = 0
        for n in range(N):
            suma += F[n]*np.exp(2j*math.pi*n*k/N)
        f[k] = suma/np.sqrt(N)
    return f

def DFT_2D(f):
    M = len(f)
    N = len(f[0])
    F = np.zeros((M,N))

    for u in range(M):
        for v in range(N):
            suma = 0
            for x in range(M):
                for y in range(N):
                    suma += f[x][y]*np.exp(-2j*math.pi*((u*x/M) + (v*y/N)))
            F[u][v] = suma/(N*M)
    return F

def IDFT_2D(F):
    M = len(F)
    N = len(F[0])
    f = np.zeros((M,N))

    for x in range(M):
        for y in range(N):
            suma = 0
            for u in range(M):
                for v in range(N):
                    suma += F[u][v]*np.exp(-2j*math.pi*((u*x/M) + (v*y/N)))
            f[x][y] = suma
    return f

def uncompress_cmpx(num):
    return complex_norm(num), np.angle(num)

# get complex numbers norm
def complex_norm(X):
    return np.sqrt(np.add( np.square(np.real(X)), np.square(np.imag(X))))

def assemble_complex(norm, angle):
    return np.multiply(norm ,np.add(\
            np.multiply(1j, np.sin(angle)),\
            np.cos(angle)\
            ))

def IFFT_TO_UINT8(ifft_img):
    return np.uint8(np.real(ifft_img))

def FFT_NORM_EQU(img):
    return equ.ecualizacion(np.uint8(np.divide(img, np.amax(img))))

def equ_feo_fft(img):
    img_size = len(img)*len(img[0])
    dtype = [('value', float),('i', int),('j', int)]
    nums = np.empty(img_size, dtype=dtype)
    for i in range(len(img)):
        for j in range(len(img[0])):
            # nums[i*len(img[0])+j] = [img[i][j],(i,j)]
            nums[i*len(img[0])+j]['value'] = img[i][j]
            nums[i*len(img[0])+j]['i'] = i
            nums[i*len(img[0])+j]['j'] = j
    nums = np.sort(nums, order='value')
    for i in range(img_size):
        img[nums[i][1]][nums[i][2]] = i/img_size
    return img

def log_transform(img, gamma=2):
    return np.power(np.log(img + 1), gamma)

def fix_norm_plot_regions(img):
    N = len(img[0])
    N_half = int(N/2)
    M = len(img)
    M_half = int(M/2)
    out = np.zeros((M,N), dtype=np.complex)
    for i in range(M):
        for j in range(N):
            # si esta en A
            if i < (M_half-1) and j < (N_half-1):
                # lo debo mandar a D
                out[M_half+i][N_half+j] = img[i][j]
            # si esta en B
            elif i < (M_half-1) and j >= (N_half-1):
                # lo debo mandar a C
                out[M_half+i][j-N_half] = img[i][j]
            # si esta en C
            elif i >= (M_half-1) and j < (N_half-1):
                # lo debo mandar a B
                out[i-M_half][j+N_half] = img[i][j]
            # si esta en D
            else:
                # lo debo mandar a A
                out[i-M_half][j-N_half] = img[i][j]
    return out
def plot_fourier_abs(img, gamma = 2):
    plt.imshow(IFFT_TO_UINT8(fix_norm_plot_regions(log_transform(img, gamma))),cmap='gray')


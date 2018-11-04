import numpy as np

def drawCorner(image, aCorner, radius = 1, value = 255):
    """Draw a circle of a certain radius around a corner."""
    for deltaX in np.arange(-radius, radius+1):
        for deltaY in np.arange(-radius, radius+1):
            # Check a circle is being drawn
            if (deltaX^2 + deltaY^2) <= radius^2:
                image[aCorner[0] + deltaX, aCorner[1] + deltaY] = value

def moravecCornerDetection(image, threshold = 100, overdrawCorners = False):
    """Detect image corners using Moravec method."""
    # Assuming BW 1 byte image
    cornerGrade = np.zeros(image.shape, dtype=np.uint8)
    # Moravec window displacements
    windowDisplacements = [(1,0), (0,1), (-1,1), (1,1)]

    for i in np.arange(1, image.shape[0]-1):
        for j in np.arange(1, image.shape[1]-1):
            minimumCornerness = np.inf
            for displacement in windowDisplacements:
                # Calculate current cornerness
                u,v = displacement
                aCornerness = (image[i+u, j+v] - image[i,j])
                aCornerness *= aCornerness

                # Keep minimum cornerness found so far
                if aCornerness < minimumCornerness:
                    minimumCornerness = aCornerness
                
            if minimumCornerness > threshold:
                cornerGrade[i,j] = minimumCornerness
                if overdrawCorners:
                    drawCorner(cornerGrade, (i,j), value=minimumCornerness)

    return cornerGrade

from scipy.signal import convolve2d

def convolvedMoravecCornerDetection(image, threshold = 100, overdrawCorners = False):
    """Convolved version of the Moravec corner operator."""
    # Window kernel for each displacement direction
    windowsKernels = [
        np.array([[-1,1]]),
        np.array([[1],[-1]]),
        np.array([[1,0],[0,-1]]),
        np.array([[-1,0],[0,1]])
    ]

    # Calculate convolution for each displacement, and obtain square difference
    displacedImages = [convolve2d(image, aKernel, mode='same') for aKernel in windowsKernels]
    displacedImages = [np.multiply(displacedImage, displacedImage) for displacedImage in displacedImages]

    # Minimum across all displacements
    minimizedCornerness = displacedImages[0]
    for i in range(1, len(displacedImages)):
        minimizedCornerness = np.minimum(minimizedCornerness, displacedImages[i])

    # Apply theshold
    belowCornernessThreshold = minimizedCornerness < threshold
    minimizedCornerness[belowCornernessThreshold] = 0

    # Overdraw if necessary
    if overdrawCorners:
        aboveCornernessThreshold = ~belowCornernessThreshold
        for i in np.arange(1, image.shape[0]-1):
            for j in np.arange(1, image.shape[1]-1):
                if aboveCornernessThreshold[i,j]:
                    drawCorner(minimizedCornerness, (i,j), value=minimizedCornerness[i,j])

    return minimizedCornerness

from .edge_detectors import _gaussian_kern as getGaussianKernel

import math

def harrisCornerDetector(image, Rthreshold = 100, windowSize = 3):
    cornerness = np.zeros(image.shape)

    # Use numpy default gradient operator, change later to a gaussian gradient?
    dx, dy = np.gradient(image)
    Ixx = dx ** 2
    Iyy = dy ** 2
    Ixy = dx * dy

    # Take by default a 5x5 gaussian kernel with sigma = 1
    gaussianKernel = getGaussianKernel(5, 1)
    Ixx =  convolve2d(Ixx, gaussianKernel, mode="same")
    Iyy =  convolve2d(Iyy, gaussianKernel, mode="same")
    Ixy =  convolve2d(Ixy, gaussianKernel, mode="same")

    offset = math.floor(windowSize/2)

    # Taking by default a 3x3 window, change later
    for i in np.arange(offset, image.shape[0]-offset):
        for j in np.arange(offset, image.shape[1]-offset):
            # W refers to the current window 
            Wxx = Ixx[i-offset:i+1+offset, j-offset:j+offset+1]
            Wyy = Iyy[i-offset:i+1+offset, j-offset:j+offset+1]
            Wxy = Ixy[i-offset:i+1+offset, j-offset:j+offset+1]

            # Compute cornerness measure
            Sxx = np.sum(Wxx)
            Syy = np.sum(Wyy)
            Sxy = np.sum(Wxy)

            # Empiric constant
            alpha = .05
            determinant = Sxx*Syy - Sxy ** 2
            trace = Sxx + Syy
            # Harris measure
            R = determinant - alpha * (trace**2)

            if R > Rthreshold:
                cornerness[i,j] = R

    # TODO:Step missing: Non-Maximum supression

    return cornerness



                





            


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

                





            


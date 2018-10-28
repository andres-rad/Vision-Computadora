from .edge_detectors import _gaussian_kern, sobel_gradient
from scipy.signal import convolve2d
import numpy as np

class AngleMatcher():
    angleDelta = np.pi/8

    def __init__(self, angle, leftNeighbour, rightNeighbour):
        self.matchingAngle = angle
        self.leftNeighbour = leftNeighbour
        self.rightNeighbour = rightNeighbour
    
    def matches(self, anAngle):
        return  self.matchingAngle - self.__class__.angleDelta <= anAngle and\
                anAngle < self.matchingAngle + self.__class__.angleDelta
    
    def isEdge(self, image, i, j):
        return image[self.getLeftNeighbour(i,j)] < image[(i,j)] and\
            image[self.getRightNeighbour(i,j)] < image[(i,j)]

    def getLeftNeighbour(self, i, j):
        return (i + self.leftNeighbour[0], j + self.leftNeighbour[1])

    def getRightNeighbour(self, i, j):
        return (i + self.rightNeighbour[0], j + self.rightNeighbour[1])

class CannyEdgeDetector():

    def __init__(self, anImage, **kwargs):
        self.image = anImage
        self.imageWidth = self.image.shape[0]
        self.imageHeight = self.image.shape[1]

        # Maybe add KWArguments for special parameters configuraion
        self.gaussianBlurKernelDimension = 5
        self.gaussianBlurKernelDeviation = 1

        if kwargs is not None:
            for arg in kwargs:
                if arg == 'gaussianBlurKernelDimension':
                    self.gaussianBlurKernelDimension = kwargs[arg]
                elif arg == 'gaussianBlurKernelDeviation':
                    self.gaussianBlurKernelDeviation = kwargs[arg]

    def apply(self):
        """Apply canny edge detector to image."""
        # Convert image to grayscale if necessary

        # Apply gaussain blur
        gaussianBlurKernel = _gaussian_kern(self.gaussianBlurKernelDimension, self.gaussianBlurKernelDeviation)
        print("Gaussian kernel being used: \n", gaussianBlurKernel)
        self.image = convolve2d(self.image, gaussianBlurKernel, mode='same')

        # Determine intensity and angle of gradient for blurred image
        # Add a KWArgument to select method of calculating the gradients, by default its sobel
        self.gradientNorm, self.gradientAngle = sobel_gradient(self.image)

        # Apply non maximum supression
        self.nonMaximumSupression()

        # "Umbralización por histerésis"

        # Close contours
        return self.edges

    def nonMaximumSupression(self): 
        # Initilize edge detected image
        self.edges = np.zeros(self.image.shape, dtype=np.uint8)

        # For each pixel, do non maximum supression
        # Leave a 1 pixel border around the image to avoid border cases
        # Apply thread pool here?
        for i in np.arange(1, self.imageWidth-1):
            for j in np.arange(1, self.imageHeight-1):
                self.doNonMaximumSupression(i,j)

    def doNonMaximumSupression(self, i, j):
        # Find matching angle
        currentAngle = self.gradientAngle[i, j]
        matcher = None

        # Since np.arctan results in an angle between pi/2 and -pi/2
        angleMatchers = [
            AngleMatcher(np.pi/2, (0,1), (0,-1)),
            AngleMatcher(np.pi/4, (1,1), (-1,-1)),
            AngleMatcher(0, (-1,0), (1,0)),
            AngleMatcher(-np.pi/4, (-1,1), (1, -1))
        ]

        for index, angleMatcher in enumerate(angleMatchers):
            if angleMatcher.matches(currentAngle):
                matcher = angleMatcher
                break

        # Angle should be -pi/2 if not set already
        if matcher is None:
            matcher = angleMatchers[0]
        
        # Check if its an edge
        if matcher.isEdge(self.gradientNorm, i, j):
            self.edges[i,j] = 255
    
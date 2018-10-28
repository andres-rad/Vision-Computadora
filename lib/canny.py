from .edge_detectors import _gaussian_kern, SobelGradient, RobertsGradient, PrewittGradient
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

class MatcherFinder():
    def __init__(self, matchersList):
        self.matchers = matchersList
    
    def find(self, anAngle):
        for matcher in self.matchers:
            if matcher.matches(anAngle):
                return matcher

        # By default, return first matcher in list
        return self.matchers[0]

class CannyEdgeDetector():

    def __init__(self, anImage, **kwargs):
        self.image = anImage
        self.imageWidth = self.image.shape[0]
        self.imageHeight = self.image.shape[1]

        # Maybe add KWArguments for special parameters configuraion
        self.gaussianBlurKernelDimension = 5
        self.gaussianBlurKernelDeviation = 1
        self.shouldApplyThresholding = True
        self.lowerThreshold = 100
        self.upperThreshold = 150

        self.angleMatcherFinder = MatcherFinder([
            AngleMatcher(np.pi/2, (-1,0), (1,0)),
            AngleMatcher(np.pi/4, (-1,1), (1,-1)),
            AngleMatcher(0, (0,1), (-1,0)),
            AngleMatcher(-np.pi/4, (1,1), (-1,-1))
        ])

        self.gradientCalculator = SobelGradient

        if kwargs is not None:
            for arg in kwargs:
                if arg == 'gaussianBlurKernelDimension':
                    self.gaussianBlurKernelDimension = kwargs[arg]
                elif arg == 'gaussianBlurKernelDeviation':
                    self.gaussianBlurKernelDeviation = kwargs[arg]
                elif arg == 'shouldApplyThresholding':
                    self.shouldApplyThresholding = False
                elif arg == 'lowerThreshold':
                    self.lowerThreshold = kwargs[arg]
                elif arg == 'upperThreshold':
                    self.upperThreshold = kwargs[arg]
                elif arg == 'gradient':
                    self.configureGradientCalculator(kwargs[arg])
    
    def configureGradientCalculator(self, aGradientName):
        if aGradientName == 'Prewitt':
            self.gradientAngle = PrewittGradient
        elif aGradientName == 'Roberts':
            self.gradientCalculator = RobertsGradient

    def apply(self):
        """Apply canny edge detector to image."""
        # Convert image to grayscale if necessary

        # Apply gaussain blur
        self.applyGaussianKernel()

        # Determine intensity and angle of gradient for blurred image
        # Add a KWArgument to select method of calculating the gradients, by default its sobel
        self.gradientNorm, self.gradientAngle = self.gradientCalculator(self.image)

        # Apply non maximum supression
        self.nonMaximumSupression()

        if self.shouldApplyThresholding:
            # Hystheresis thresholding
            self.hystheresisThresholding()

        # Close contours
        return self.edges

    def applyGaussianKernel(self):
        gaussianBlurKernel = _gaussian_kern(self.gaussianBlurKernelDimension, self.gaussianBlurKernelDeviation)
        self.image = convolve2d(self.image, gaussianBlurKernel, mode='same')
    
    def nonMaximumSupression(self): 
        # Initilize edge detected image
        self.edges = np.zeros(self.image.shape, dtype=np.float)
        self.debugEdges = np.zeros(self.image.shape, dtype=np.uint8)

        # For each pixel, do non maximum supression
        # Leave a 1 pixel border around the image to avoid border cases
        # Apply thread pool here?
        for i in np.arange(1, self.imageWidth-1):
            for j in np.arange(1, self.imageHeight-1):
                self.doNonMaximumSupression(i,j)

    def doNonMaximumSupression(self, i, j):
        # Find matching angle
        currentAngle = self.gradientAngle[i, j]

        # Since np.arctan results in an angle between pi/2 and -pi/2
        # For each angle matcher, since the gradient direction is perpendicular to the edge direction,
        # the selected neighbours should consider this.
        # See: http://www.sci.utah.edu/~gerig/CS6640-F2012/Materials/Canny-Gerig-Slides-updated.pdf slide 18

        matcher = self.angleMatcherFinder.find(currentAngle)

        # Check if its an edge
        if matcher.isEdge(self.gradientNorm, i, j):
            self.edges[i,j] = self.gradientNorm[i,j]

    def hystheresisThresholding(self):
        self.afterSupressionEdges = np.copy(self.edges)
        self.edges = np.zeros(self.edges.shape)

        for i in np.arange(1, self.imageWidth-1):
            for j in np.arange(1, self.imageHeight-1):
                if self.afterSupressionEdges[i,j] > self.upperThreshold:
                    self.doHystheresisThresholding(i,j)

    def doHystheresisThresholding(self, i, j):
        # Set upper threshold pixel as an edge
        self.edges[i,j] = 255
        # Expand it starting
        matcher = self.angleMatcherFinder.find(self.gradientAngle[i,j])
        leftNeighbour = matcher.getLeftNeighbour(i,j)
        rightNeighbour = matcher.getRightNeighbour(i,j)
        if self.edges[leftNeighbour] != 255 and self.afterSupressionEdges[leftNeighbour] > self.lowerThreshold:
            self.edges[i,j] = 255
            self.doHystheresisThresholdingWithPosition(leftNeighbour)
        if self.edges[rightNeighbour] != 255 and self.afterSupressionEdges[rightNeighbour] > self.lowerThreshold:
            self.edges[i,j] = 255
            self.doHystheresisThresholdingWithPosition(rightNeighbour)
    
    def doHystheresisThresholdingWithPosition(self, aPosition):
        """Do apply hystheresis thresholding but with a tuple-like argument."""
        self.doHystheresisThresholding(aPosition[0], aPosition[1])
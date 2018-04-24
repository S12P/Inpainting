import sys
import cv2
import numpy as np

Lap = np.array([[ 1.,  1.,  1.],[ 1., -8.,  1.],[ 1.,  1.,  1.]])
kerx = np.array([[ 0.,  0.,  0.], [-1.,  0.,  1.], [ 0.,  0.,  0.]])
kery = np.array([[ 0., -1.,  0.], [ 0.,  0.,  0.], [ 0.,  1.,  0.]])

def IdentifyTheFillFront(masque, source):
    """ Identifie le front de remplissage """
    dOmega = []
    normale = []
    lap = cv2.filter2D(masque, cv2.CV_32F, Lap)
    GradientX = cv2.filter2D(source, cv2.CV_32F, kerx)
    GradientY = cv2.filter2D(source, cv2.CV_32F, kery)
    xsize, ysize = lap.shape
    for x in range(xsize):
        for y in range(ysize):
            if lap[x, y] > 0:
                dOmega+=[(y, x)]
                dx = GradientX[x, y]
                dy = GradientY[x, y]
                N = (dy**2 + dx**2)**0.5
                if N != 0:
                    normale+=[(dy/N, -dx/N)]
                else:
                    normale+=[(dy, -dx)]
    return(dOmega, normale)

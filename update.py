import sys
import cv2
import numpy as np


def update(im, gradientX, gradientY, confiance, source, masque, dOmega, point, list, index, taillecadre):
    p = dOmega[index]
    px, py = p
    patch = Patch(im, taillecadre, p)
    x1, y1 = patch[0]
    x2, y2 = patch[1]
    px, py = point
    for (i, j) in list:
        im[y1+i, x1+j] = im[py+i, px+j]
        confiance[y1+i, x1+j] = confiance[py, px]
        source[y1+i, x1+j] = 1
        masque[y1+i, x1+j] = 0
    return(im, gradientX, gradientY, confiance, source, masque)

def Patch(im, taillecadre, point):
    """
    Permet de calculer les deux points extreme du patch
    Voici le patch avec les 4 points
        1 _________ 2
          |        |
          |        |
         3|________|4
    """
    px, py = point
    xsize, ysize, c = im.shape
    x3 = max(px - taillecadre, 0)
    y3 = max(py - taillecadre, 0)
    x2 = min(px + taillecadre, ysize - 1)
    y2 = min(py + taillecadre, xsize - 1)
    return((x3, y3),(x2, y2))

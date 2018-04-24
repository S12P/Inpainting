import sys
import cv2
import numpy as np

def patch_complet(x, y, xsize, ysize, original):
    for i in range(xsize):
        for j in range(ysize):
            if original[x+i,y+j]==0:
                return(False)
    return(True)

def crible(xsize,ysize,x1,y1,masque):
    compteur=0
    cibles,ciblem=[],[]
    for i in range(xsize):
        for j in range(ysize):
            if masque[y1+i, x1+j] == 0:
                compteur += 1
                cibles+=[(i, j)]
            else:
                ciblem+=[(i, j)]
    return (compteur,cibles,ciblem,xsize,ysize)

def calculPatch(dOmega, cibleIndex, im, original, masque, taillecadre):
    mini = minvar = sys.maxsize
    sourcePatch,sourcePatche = [],[]
    p = dOmega[cibleIndex]
    patch = Patch(im, taillecadre, p)
    x1, y1 = patch[0]
    x2, y2 = patch[1]
    Xsize, Ysize, c = im.shape
    compteur,cibles,ciblem,xsize,ysize=crible(y2-y1+1,x2-x1+1,x1,y1,masque)
    for x in range(Xsize - xsize):
        for y in range(Ysize - ysize):
            if patch_complet(x, y, xsize, ysize, original):
                sourcePatch+=[(x, y)]
    for (y, x) in sourcePatch:
        R = V = B = ssd = 0
        for (i, j) in cibles:
            ima = im[y+i,x+j]
            omega = im[y1+i,x1+j]
            for k in range(3):
                difference = float(ima[k]) - float(omega[k])
                ssd += difference**2
            R += ima[0]
            V += ima[1]
            B += ima[2]
        ssd /= compteur
        if ssd < mini:
            variation = 0
            for (i, j) in ciblem:
                ima = im[y+i,x+j]
                differenceR = ima[0] - R/compteur
                differenceV = ima[1] - V/compteur
                differenceB = ima[2] - B/compteur
                variation += differenceR**2 + differenceV**2 + differenceB**2
            if ssd <  mini or variation < minvar:
                minvar = variation
                mini = ssd
                pointPatch = (x, y)
    return(ciblem, pointPatch)


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

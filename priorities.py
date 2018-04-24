import sys
import cv2
import numpy as np

Lap = np.array([[ 1.,  1.,  1.],[ 1., -8.,  1.],[ 1.,  1.,  1.]])
kerx = np.array([[ 0.,  0.,  0.], [-1.,  0.,  1.], [ 0.,  0.,  0.]])
kery = np.array([[ 0., -1.,  0.], [ 0.,  0.,  0.], [ 0.,  1.,  0.]])

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

def calculConfiance(confiance, im, taillecadre, masque, dOmega):
    """Permet de calculer la confiance définie dans l'article"""
    for k in range(len(dOmega)):
        px, py = dOmega[k]
        patch = Patch(im, taillecadre, dOmega[k])
        x3, y3 = patch[0]
        x2, y2 = patch[1]
        compteur = 0
        taille_psi_p = ((x2-x3+1) * (y2-y3+1))
        for x in range(x3, x2 + 1):
            for y in range(y3, y2 + 1):
                if masque[y, x] == 0: # intersection avec not Omega
                    compteur += confiance[y, x]
        confiance[py, px] = compteur / taille_psi_p
    return(confiance)

def calculData(dOmega, normale, data, gradientX, gradientY, confiance):
    """Permet de calculer data définie dans l'article"""
    for k in range(len(dOmega)):
        x, y = dOmega[k]
        NX, NY = normale[k]
        data[y, x] = (((gradientX[y, x] * NX)**2 + (gradientY[y, x] * NY)**2)**0.5) / 255.
    return(data)


def calculPriority(im, taillecadre, masque, dOmega, normale, data, gradientX, gradientY, confiance):
    """Permet de calculer la priorité du patch"""
    C = calculConfiance(confiance, im, taillecadre, masque, dOmega)
    D = calculData(dOmega, normale, data, gradientX, gradientY, confiance)
    index = 0
    maxi = 0
    for i in range(len(dOmega)):
        x, y = dOmega[i]
        P = C[y,x]*D[y,x]
        if P > maxi:
            maxi = P
            index = i
    return(C, D, index)

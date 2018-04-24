#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import cv2
import numpy as np
import fillfront
import priorities
import bestpatch
import update

try:
    arg=sys.argv
    cheminimage = str(arg[1])
    cheminmasque = str(arg[2])
    if len(arg) == 4:
        taillecadre = int(arg[3])
    else:
        taillecadre = 3
except:
    print("Vérifier la syntaxe")
    exit()

image = cv2.imread(cheminimage,1)
masque = cv2.imread(cheminmasque,0)
xsize, ysize, channels = image.shape # meme taille pour filtre et image

#on verifie les tailles

x, y = masque.shape

if x != xsize or y != ysize:
    print("La taille de l'image et du filtre doivent être les même")
    exit()

tau = 170 #valeur pour séparer les valeurs du masque
omega=[]
confiance = np.copy(masque)
masque = np.copy(masque)
for x in range(xsize):
    for y in range(ysize):
        v=masque[x,y]
        if v<tau:
            omega.append([x,y])
            image[x,y]=[255,255,255]
            masque[x,y]=1
            confiance[x,y]=0.
        else:
            masque[x,y]=0
            confiance[x,y]=1.

cv2.imwrite(cheminimage[:-4] + "_avec_masque.png",image)
source = np.copy(confiance)
original= np.copy(confiance)
dOmega = []
normale = []


im = np.copy(image)
result = np.ndarray(shape = image.shape)


data = np.ndarray(shape = image.shape[:2])
Lap = np.array([[ 1.,  1.,  1.],[ 1., -8.,  1.],[ 1.,  1.,  1.]])
kerx = np.array([[ 0.,  0.,  0.], [-1.,  0.,  1.], [ 0.,  0.,  0.]])
kery = np.array([[ 0., -1.,  0.], [ 0.,  0.,  0.], [ 0.,  1.,  0.]])


bool = True #pour le while
print("Algorithme en fonctionnement")
k=0

niveau_de_gris = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)

gradientX = np.float32(cv2.convertScaleAbs(cv2.Scharr(niveau_de_gris, cv2.CV_32F, 1, 0)))

gradientY = np.float32(cv2.convertScaleAbs(cv2.Scharr(niveau_de_gris, cv2.CV_32F, 0, 1)))
while bool:
    print(k)
    k+=1
    xsize, ysize = source.shape

    niveau_de_gris = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)

    gradientX = np.float32(cv2.convertScaleAbs(cv2.Scharr(niveau_de_gris, cv2.CV_32F, 1, 0)))

    gradientY = np.float32(cv2.convertScaleAbs(cv2.Scharr(niveau_de_gris, cv2.CV_32F, 0, 1)))

    for x in range(xsize):
        for y in range(ysize):
            if masque[x][y] == 1:
                gradientX[x][y] = 0
                gradientY[x][y] = 0
    gradienX, gradientY = gradientX/255, gradientY/255


    dOmega, normale = fillfront.IdentifyTheFillFront(masque, source)


    confiance, data, index = priorities.calculPriority(im, taillecadre, masque, dOmega, normale, data, gradientX, gradientY, confiance)


    list, pp = bestpatch.calculPatch(dOmega, index, im, original, masque, taillecadre)


    im, gradientX, gradientY, confiance, source, masque = update.update(im, gradientX, gradientY, confiance, source, masque, dOmega, pp, list, index, taillecadre)

        # on verifie si on a fini
    bool = False
    for x in range(xsize):
        for y in range(ysize):
            if source[x, y] == 0:
                bool = True

        # on enregistre a chaque fois pour voir l'avancée
    cv2.imwrite(cheminimage[:-4] + "_resultat.jpg", im)

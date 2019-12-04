import numpy as np
import cv2
import os 
import windowmatching as wm
import dynamicprogramming as dp
import graphcuts as gc

INPUT_DIR = "input"
OUTPUT_DIR = "output"

def subPixel(im):
    left = np.copy(im).astype(float)
    left[:,1:] += im[:,:-1]
    left[:,1:] /= 2

    right = np.copy(im).astype(float)
    right[:,:-1] += im[:,1:]
    right[:,:-1] /=2

    up = np.copy(im).astype(float)
    up[1:,:] += im[:-1,:]
    up[1:,:] /=2

    down = np.copy(im).astype(float)
    down[:-1,:] += im[1:,:]
    down[:-1,:] /=2
    
    minimum = np.array([im, left, right, up, down]).min(axis=0)
    maximum = np.array([im, left, right, up, down]).max(axis=0)
    return minimum, maximum

def isValidLoc(coordP, coordR):
    rowValid = 0 <= coordP[0] < coordR[0]
    colValid = 0 <= coordP[1] < coordR[1]
    return rowValid and colValid

def addPairwise(g, n1, n2, E00, E01, E10, E11):
	g.add_edge(n1, n2, 0, (E01+E10)-(E00+E11))
	g.add_tedge(n1, E11, E01)
	g.add_tedge(n2, 0, E00 - E01)

def forbid01(g, n1, n2, OCCLUDED):
    g.add_edge(n1, n2, OCCLUDED, 0)


def video(shape, method):

    pathL = os.path.join(INPUT_DIR, "cropped", "left")
    pathR = os.path.join(INPUT_DIR, "cropped", "right")
    output = os.path.join(OUTPUT_DIR, "drive_"+method+".avi")

    filesL = [f for f in os.listdir(pathL) if os.path.isfile(os.path.join(pathL, f))]
    filesR = [f for f in os.listdir(pathR) if os.path.isfile(os.path.join(pathR, f))]
    fps = 25
    
    filesL.sort(key = lambda x: int(x[5:-4]))
    filesR.sort(key = lambda x: int(x[5:-4]))

    filesL = filesL[:100]
    filesR = filesR[:100]

    if method == "wm":
        getFrame = wm.windowMatchingGray
    elif method == "dp":
        getFrame = dp.DP
    elif method == "gc":
        getFrame = gc.start

    writer = cv2.VideoWriter(output,cv2.VideoWriter_fourcc(*'DIVX'), fps, (shape[0], shape[1]))
    for i, (l, r) in enumerate(zip(filesL,filesR)):
        filenameL=os.path.join(pathL, l)
        filenameR=os.path.join(pathR, r)

        #reading each files
        imL= cv2.imread(filenameL)
        imR = cv2.imread(filenameR)

        disparity = getFrame(imL,imR)
        disparity = cv2.applyColorMap(disparity, cv2.COLORMAP_PARULA)
        #inserting the frames into an image array
        writer.write(disparity)

        if i %10 ==0:
            print(100*i/len(filesL))

    writer.release()
 
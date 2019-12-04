import cv2
import os
import windowmatching as wm
import dynamicprogramming as dp
import graphcuts as gc
import utils
from time import perf_counter
# I/O directories
INPUT_DIR = "input"
OUTPUT_DIR = "output"


def windowMatching(imL, imR, color=False, window=3, thresPercent=0.04):
    if color:
        disparity = wm.windowMatching(imL, imR, window=window, thresPercent=thresPercent)
        disparity = cv2.applyColorMap(disparity, cv2.COLORMAP_PARULA)
    else:
        imL = cv2.cvtColor(imL, cv2.COLOR_BGR2GRAY)
        imR = cv2.cvtColor(imR, cv2.COLOR_BGR2GRAY)
        
        disparity = wm.windowMatchingGray(imL, imR, window=window, thresPercent=thresPercent)
        disparity = cv2.applyColorMap(disparity, cv2.COLORMAP_PARULA)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "window_matching.png"), disparity)

def dynamicProgramming(imL, imR):
    imL = cv2.cvtColor(imL, cv2.COLOR_BGR2GRAY)
    imR = cv2.cvtColor(imR, cv2.COLOR_BGR2GRAY)

    # blurring will help alignment
    imL = cv2.bilateralFilter(imL, 15, 40, 20)
    imR = cv2.bilateralFilter(imR, 15, 40, 20)

    disparity = dp.DP(imL, imR)
    disparity = cv2.applyColorMap(disparity, cv2.COLORMAP_PARULA)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "dynamic_programming.png"), disparity)

def graphCut(imL, imR, dispSize=16):
    imL = cv2.cvtColor(imL, cv2.COLOR_BGR2GRAY)
    imR = cv2.cvtColor(imR, cv2.COLOR_BGR2GRAY)

    disparity = gc.start(imL, imR, dispSize=dispSize)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "graph_cuts.png"), disparity)
    

if __name__ == '__main__':
    imL = cv2.imread(os.path.join(INPUT_DIR, "imL.png"))
    imR = cv2.imread(os.path.join(INPUT_DIR, "imR.png"))
    # imL = cv2.imread(os.path.join(INPUT_DIR, "pipesL.png"))
    # imR = cv2.imread(os.path.join(INPUT_DIR, "pipesR.png"))
    # imL = cv2.resize(imL, (0,0), fx=.1, fy=.1)
    # imR = cv2.resize(imR, (0,0), fx=.1, fy=.1)
    assert imL.shape == imR.shape, "image dimensions don't match"

    # times based on 384x288 Tsukuba image
    # Intel i5 6600k

    # ~0.75 secs - greyscale
    # ~1.2 secs - color
    start = perf_counter()
    windowMatching(imL, imR, color=False,window=3, thresPercent=0.01)
    print("Window Matching took: ", perf_counter() - start, "seconds")

    # ~12 secs
    start = perf_counter()
    dynamicProgramming(imL, imR)
    print("Dynamic Programming took: ", perf_counter() - start, "seconds")

    # ~120 secs
    start = perf_counter()
    graphCut(imL, imR,  dispSize=16)
    print("Graph Cuts took: ", perf_counter() - start, "seconds")
    
    # utils.video(.75, "wm")
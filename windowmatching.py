import cv2
import numpy as np
from numpy.lib.stride_tricks import as_strided
import multiprocessing as mp

def windowMatchingHelper(leftStrided,rightStrided,maxDist):
    disparity = np.zeros((leftStrided.shape[0],leftStrided.shape[2] - leftStrided.shape[1] + 1))
    for row, (l, r) in enumerate(zip(leftStrided,rightStrided)):
        for col in range(l.shape[1] - l.shape[0] + 1):
            end = max(0,col - maxDist)
            match = cv2.matchTemplate(l[:,col:col+l.shape[0],:], r[:,end:col+l.shape[0],:], method=cv2.TM_SQDIFF_NORMED)
            disparity[row,col] = col - end - np.argmin(match)
    return disparity

def windowMatching(imL,imR, window=3, thresPercent=0.10):
    assert window % 2 == 1, "window not odd"
    assert imL.shape[1] >= window, "window too big"
    
    # precompute rolling windows with as_strided()
    # strided = as_strided(test, 
    #                     shape=((test.shape[0] - window + 1), window, test.shape[1], 3),
    #                     strides=(test.strides[0], test.strides[0], test.strides[1],test.strides[2]))
    leftStrided = as_strided(imL, 
                            shape=((imL.shape[0] - window + 1), window, imL.shape[1],3),
                            strides=(imL.strides[0], imL.strides[0], imL.strides[1], imR.strides[2]))

    rightStrided = as_strided(imR, 
                        shape=((imR.shape[0] - window + 1), window, imR.shape[1],3),
                        strides=(imR.strides[0], imR.strides[0], imR.strides[1], imR.strides[2]))

    # divide into sections for multiprocessing
    cpus = mp.cpu_count()
    leftSections = np.array_split(leftStrided, cpus)
    rightSections = np.array_split(rightStrided, cpus)
    maxDist = [int(thresPercent * imL.shape[1])] * cpus
    with mp.Pool(processes=cpus) as pool:
        disparity = pool.starmap(windowMatchingHelper, zip(leftSections,rightSections,maxDist))
    disparity = np.vstack(disparity)

    # normalize so values are 0 - 255
    disparity = cv2.normalize(disparity,0,255,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    return disparity

########################################################################################################
def disparityGrayHelper(leftStrided,rightStrided,maxDist):
    disparity = np.zeros((leftStrided.shape[0],leftStrided.shape[2] - leftStrided.shape[1] + 1))
    for row, (l, r) in enumerate(zip(leftStrided,rightStrided)):
        for col in range(l.shape[1] - l.shape[0] + 1):
            end = max(0,col - maxDist)
            match = cv2.matchTemplate(l[:,col:col+l.shape[0]], r[:,end:col+l.shape[0]], method=cv2.TM_SQDIFF)
            minVal, _, minLoc, _ = cv2.minMaxLoc(match)
            # disparity[row,col] = minVal
            argmin = minLoc[0]
            disparity[row,col] = col - end - argmin
    return disparity

def windowMatchingGray(imL,imR, window=3, thresPercent=0.10):
    assert window % 2 == 1, "window not odd"
    assert imL.shape[1] >= window, "window too big"
    
    # precompute rolling windows with as_strided()
    leftStrided = as_strided(imL, 
                            shape=((imL.shape[0] - window + 1), window, imL.shape[1]),
                            strides=(imL.strides[0], imL.strides[0], imL.strides[1]))

    rightStrided = as_strided(imR, 
                        shape=((imR.shape[0] - window + 1), window, imR.shape[1]),
                        strides=(imR.strides[0], imR.strides[0], imR.strides[1]))

    # divide into sections for multiprocessing
    cpus = max(1, mp.cpu_count()-1)
    leftSections = np.array_split(leftStrided, cpus)
    rightSections = np.array_split(rightStrided, cpus)
    maxDist = [int(thresPercent * imL.shape[1])] * cpus
    with mp.Pool(processes=cpus) as pool:
        disparity = pool.starmap(disparityGrayHelper, zip(leftSections,rightSections,maxDist))
    disparity = np.vstack(disparity)

    # normalize so values are 0 - 255
    disparity = cv2.normalize(disparity,0,255,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    return disparity

########################################################################################################
# def disparityHelper2(leftStrided,rightStrided):
#     disparity = np.zeros((rightStrided.shape[0],rightStrided.shape[2] - rightStrided.shape[1] + 1))
#     for row, z in enumerate(zip(leftStrided,rightStrided)):
#         lWindows, rRows = z
#         for col, lWindow in enumerate(lWindows):
#             match = cv2.matchTemplate(lWindow, rRows, method=cv2.TM_CCORR_NORMED)
#             _, _, _, max_loc = cv2.minMaxLoc(match)
#             dif = abs(max_loc[0] - col)
#             disparity[row,col] = dif if dif <= 14 else -1
#     return disparity
# # as_strided the horizontal sliding window
# def getDisparityMap2(imL,imR, window=3):
#     assert window % 2 == 1, "window not odd"
#     assert imL.shape[1] >= window, "window too big"
#     # leftStrided = as_strided(imL, 
#     #                         shape=((imL.shape[0] - window + 1), window, imL.shape[1]),
#     #                         strides=(imL.strides[0], imL.strides[0], imL.strides[1]))

#     leftStrided = as_strided(imL, 
#                             shape=((imL.shape[0] - window + 1), (imL.shape[1] - window + 1), window, window),
#                             strides=(imL.strides[0], imL.strides[1], imL.strides[0], imL.strides[1]))

#     rightStrided = as_strided(imR, 
#                         shape=((imR.shape[0] - window + 1), window, imR.shape[1]),
#                         strides=(imR.strides[0], imR.strides[0], imR.strides[1]))

#     cpus = mp.cpu_count()

#     # divide into sections for multiprocessing
#     leftSection = np.array_split(leftStrided, cpus)
#     rightSection = np.array_split(rightStrided, cpus)

#     # windows = [window] * cpus
#     with mp.Pool(processes=cpus) as pool:
#         disparity = pool.starmap(disparityHelper2, zip(leftSection,rightSection))
#     disparity = np.vstack(disparity)

#     # normalize so values 0 - 255
#     # let no-matchs = 0
#     disparity *= 255/np.max(disparity)
#     disparity[disparity < 0] = 0
#     disparity = disparity.astype(np.uint8)
#     return disparity
########################################################################################################
# # loops for everything
# def getDisparityMap3(imL,imR, window=3):
#     assert window % 2 == 1, "window not odd"
#     assert imL.shape[1] >= window, "window too big"

#     disparity = np.zeros((imL.shape[0] - window + 1, imL.shape[1] - window + 1))

#     for row in range(imL.shape[0] - window + 1):
#         for col in range(imL.shape[1] - window + 1):
#             match = cv2.matchTemplate(imL[row:row+window,col:col+window], 
#                                     imR[row:row+window:], method=cv2.TM_CCORR_NORMED)
#             _, _, _, max_loc = cv2.minMaxLoc(match)
#             dif = abs(max_loc[0] - col)
#             disparity[row,col] = dif if dif <= 14 else -1


#     # normalize so values 0 - 255
#     # let no-matchs = 0
#     disparity *= 255/np.max(disparity)
#     disparity[disparity < 0] = 0
#     disparity = disparity.astype(np.uint8)
#     return disparity
    
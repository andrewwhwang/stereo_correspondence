import cv2
import numpy as np
import multiprocessing as mp
import maxflow
import heapq
import utils

np.seterr(over='ignore')
VAR_ALPHA = -1
VAR_ABSENT = -2
CUTOFF = 30

# make it so you can't change OCCLUDED
class Final(object):
    OCCLUDED = 1<<30
    def __setattr__(self, name, value):
        return

class GraphCut:
    def __init__(self, imL, imR):
        self.imL = imL
        self.imR = imR
        self.occ = Final()
        self.maxIterations = 4
        self.alphaRange = 16
        self.imLMin, self.imLMax = utils.subPixel(self.imL)
        self.imRMin, self.imRMax = utils.subPixel(self.imR)
        self.K, self.lambda1, self.lambda2, self.denominator = self.getK()
        self.energy = 0
        self.disparity = np.full((self.imL.shape), self.occ.OCCLUDED, dtype=np.int64)
        self.vars0 = np.zeros((self.imL.shape), dtype=np.int64)
        self.varsA = np.zeros((self.imL.shape), dtype=np.int64)
        self.activePenalty = 0
        self.edgeThresh = 8

    def getDistance(self, p, q):
        ImP = self.imL[p]
        ImQ = self.imR[q]

        ImQMin = self.imRMin[q]
        ImQMax = self.imRMax[q]
        ImPMin = self.imLMin[p]
        ImPMax = self.imLMax[p]

        pDist = qDist = 0 

        if ImP < ImQMin:
            pDist = ImQMin - ImP
        elif ImP > ImQMax:
            pDist = ImP - ImQMax

        if ImQ < ImPMin:
            qDist = ImPMin - ImQ
        elif ImQ > ImPMax:
            qDist = ImQ - ImPMax

        d = min(pDist, qDist, CUTOFF)
        return d**2

    def getDispDistance(self, coordL, coordL2, disp):
        imLDist = abs(self.imL[coordL] - self.imL[coordL2])
        imRDist = abs(self.imR[coordL[0], coordL[1] + disp] - self.imR[coordL2[0], coordL2[1] + disp])

        if imLDist < self.edgeThresh and imRDist < self.edgeThresh:
            return self.lambda1
        return self.lambda2

    def getK(self):
        k = max(3, (self.alphaRange+3) // 4)
        totalKthSmallest = 0
        total = self.imL.shape[0] * (self.imL.shape[1] - self.alphaRange)

        for row, col in np.ndindex((self.imL.shape[0], self.imL.shape[1] - self.alphaRange)):
            # get kth smallest using minheap method
            heap = []
            for d in range(0, self.alphaRange+1):
                delta = -1 * self.getDistance((row, col), (row, col + d))
                heapq.heappush(heap, delta)
                if len(heap) -1 >= k:
                    heapq.heappop(heap)
            totalKthSmallest += -1 * heapq.heappop(heap)
        K = totalKthSmallest/total

        lambda2 = K / 5
        lambda1 = 3 * lambda2

        N = []
        for i in range(1,17):
            firstNumerator = round(i * K)
            first = abs(firstNumerator / (i * K) - 1)
            secondNumerator = round(i * lambda1)
            second = abs(secondNumerator / (i * lambda1) - 1)
            thirdNumerator = round(i * lambda2)
            third = abs(thirdNumerator / (i * lambda2) - 1)
            N.append((first+second+third, firstNumerator, secondNumerator, thirdNumerator, i))
            
        N = sorted(N)[0]
        _, K, lambda1, lambda2, denominator = N
        return K, lambda1, lambda2, denominator

    def addDataOccuTerms(self, g, label):
        for coordL, _ in np.ndenumerate(self.imL):

            disp = self.disparity[coordL]
            coordR = (coordL[0], coordL[1] + disp)


            if disp == label:
                self.vars0[coordL] = VAR_ALPHA
                self.varsA[coordL] = VAR_ALPHA
                penalty = self.denominator * self.getDistance(coordL, coordR) - self.K
                self.activePenalty += penalty
            else:
                #vars0
                if disp != self.occ.OCCLUDED:
                    penalty = self.denominator * self.getDistance(coordL, coordR) - self.K
                    node_id = g.add_nodes(1)[0]
                    g.add_tedge(node_id, 0, penalty)
                    self.vars0[coordL] = node_id
                else:
                    self.vars0[coordL] = VAR_ABSENT

                #varsA
                coordR = (coordL[0], coordL[1] + label)
                if utils.isValidLoc(coordR, self.imL.shape):
                    penalty = self.denominator * self.getDistance(coordL, coordR) - self.K
                    node_id = g.add_nodes(1)[0]
                    g.add_tedge(node_id, penalty, 0)
                    self.varsA[coordL] = node_id
                else:
                    self.varsA[coordL] = VAR_ABSENT


    def addSmoothnessTerms(self, g, label):
        for coordL, _ in np.ndenumerate(self.imL):
            for i in range(2):
                coordL2 = (coordL[0]+i, coordL[1]+i-1)
                if utils.isValidLoc(coordL2, self.imL.shape):

                    disp1 = self.disparity[coordL]
                    vars01 = self.vars0[coordL]

                    disp2 = self.disparity[coordL2]
                    vars02 = self.vars0[coordL2]

                    # pairwise assignment
                    if disp1 == disp2 and vars01 >= 0 and vars02 >= 0:
                        delta = self.getDispDistance(coordL, coordL2, disp1)
                        utils.addPairwise(g, vars01, vars02, 0, delta, delta, 0)

                    # vars01
                    if disp1 != disp2 and vars01 >= 0 and utils.isValidLoc((coordL2[0], coordL2[1] + disp1), self.imL.shape):
                        g.add_tedge(vars01, 0, self.getDispDistance(coordL, coordL2, disp1))

                    # vars02
                    if disp1 != disp2 and vars02 >= 0 and utils.isValidLoc((coordL[0], coordL[1] + disp2), self.imL.shape):
                        g.add_tedge(vars02, 0, self.getDispDistance(coordL, coordL2, disp2))
                        
                    varsA1 = self.varsA[coordL]
                    varsA2 = self.varsA[coordL2]
                    
                    # varsA1 and varsA2
                    if varsA1 != VAR_ABSENT and varsA2 != VAR_ABSENT:
                        delta = self.getDispDistance(coordL, coordL2, label)
                        if varsA1 != VAR_ALPHA:
                            if varsA2 != VAR_ALPHA:
                                utils.addPairwise(g, varsA1, varsA2, 0, delta, delta, 0)
                            else:
                                g.add_tedge(varsA1, 0, delta)
                        elif varsA2 != VAR_ALPHA:
                            g.add_tedge(varsA2, 0, delta)


############################change#######################################
    def addUniqueTerms(self, g, label):
        for coordL, _ in np.ndenumerate(self.imL):
            if self.vars0[coordL] >= 0:

                varA = self.varsA[coordL]
                if varA != VAR_ABSENT:
                    utils.forbid01(g, self.vars0[coordL], varA, self.occ.OCCLUDED)

                disp = self.disparity[coordL]
                coordL2 = (coordL[0], coordL[1] + disp - label)
                if utils.isValidLoc(coordL2, self.imL.shape):
                    varA = self.varsA[coordL2]
                    utils.forbid01(g, self.vars0[coordL], varA, self.occ.OCCLUDED)

    def mainLoop(self):

        dispSize = self.alphaRange + 1
        self.energy = 0

        done = np.full(dispSize, False)
        for i in range(self.maxIterations):
            for label in np.random.permutation(dispSize):
                if not done[label]:
                    self.activePenalty = 0
                    g = maxflow.Graph[int](2 * self.imL.size, 12 * self.imL.size)

                    self.addDataOccuTerms(g, label)
                    self.addSmoothnessTerms(g, label)
                    self.addUniqueTerms(g, label)
                    newEnergy = g.maxflow() + self.activePenalty
                    if newEnergy < self.energy:
                        self.updateDisp(g, label)
                        self.energy = newEnergy
                        done[:] = False
                    done[label] = True
                    if np.all(done):
                        return self.disparity
        return self.disparity

    def updateDisp(self, g, label):
        vecGetSegment = np.vectorize(g.get_segment)

        if self.vars0[self.vars0 >= 0].size > 0:
            vars0Segments = np.zeros(self.vars0.shape)
            vars0Segments[self.vars0 >= 0] = vecGetSegment(self.vars0[self.vars0 >= 0])
            self.disparity[vars0Segments == 1] = self.occ.OCCLUDED

        if self.varsA[self.varsA >= 0].size > 0:
            varsASegments = np.zeros(self.varsA.shape)
            varsASegments[self.varsA >= 0] = vecGetSegment(self.varsA[self.varsA >= 0])
            self.disparity[varsASegments == 1] = label


def helper(imL, imR):
    g = GraphCut(imL, imR)
    disparity = g.mainLoop()
    return disparity

def start(imR, imL, dispSize=16):
    cpus = max(1, mp.cpu_count()-1)
    leftSections = np.array_split(imL, cpus)
    rightSections = np.array_split(imR, cpus)

    with mp.Pool(processes=cpus) as pool:
        disparity = pool.starmap(helper, zip(leftSections,rightSections))
    disparity = np.vstack(disparity)

    im = np.zeros(imL.shape, dtype=np.uint8)
    # im = np.full(imL.shape, -1)
    occluded = disparity == 1<<30
    im[np.logical_not(occluded)] = disparity[np.logical_not(occluded)] * 255/dispSize

    im = cv2.applyColorMap(im, cv2.COLORMAP_PARULA)
    im[occluded] = np.array([0,0,0])
    return im
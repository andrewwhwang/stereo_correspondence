import cv2
import numpy as np
import multiprocessing as mp

MATCH = 0
SUB = -1
INDEL = -1
TOLERANCE = 2

# get matrix elements one-by-one
def getMatrixNaive(query, ref):
    m, n = query.shape[0], ref.shape[0]
    # Generate DP table and traceback path pointer matrix
    score = np.zeros((m+1, n+1))      # the DP table
   
    # Calculate DP table
    score[:,0] = INDEL * np.arange(m+1)
    score[0,:] = INDEL * np.arange(n+1)

    for i, j in np.ndindex((m, n)):
        corr = score[i,j] + getDiag(query[i], ref[j])
        insert = score[i+1,j] + INDEL
        delete = score[i,j+1] + INDEL
        score[i+1,j+1] = max(corr, delete, insert)
    return score

# get matrix using myer's bitparallel algo
def getMatrix(query, ref):
    
    mask = 2**len(query) - 1
    dpPos = np.zeros((len(ref),len(query)))
    dpNeg= np.zeros((len(ref),len(query)))

    # create a grid of scores
    length = len(query)
    scores = np.zeros((len(query)+1,len(ref)+1))
    scores[:,0] = np.arange(len(query)+1)
    scores[0,:] = np.arange(len(ref)+1)

    # preprocessing, query -> bitarray for each num position
    keys = list(range(256))
    values = [0] * len(keys)
    bitArrays = dict(zip(keys, values))

    for i, value in enumerate(query):
        for j in range(-TOLERANCE, TOLERANCE+1):
            if 0 <= value + j < len(keys):
                bitArrays[value + j] |= 1 << i

    # main algo
    vn = 0
    vp = (1 << length) - 1
    for i, value in enumerate(ref):
        X = bitArrays[value] | vn
        D0 = ((vp + (X & vp)) ^ vp) | X & mask
        hn = vp & D0 & mask
        hp = vn | ~(vp | D0) & mask
        X = hp << 1 | 1
        vn = X & D0 & mask
        vp = (hn << 1) | ~(X | D0) & mask

        dpPos[i] = np.array(list(np.binary_repr(hp).zfill(len(query))[::-1])).astype(int)
        dpNeg[i] = np.array(list(np.binary_repr(hn).zfill(len(query))[::-1])).astype(int)

    dpPos = np.cumsum(dpPos,axis=0,dtype=int) 
    dpNeg = np.cumsum(dpNeg,axis=0,dtype=int)

    firstCol = np.arange(1, length+1)
    scores[1:,1:] = np.subtract(firstCol, dpNeg - dpPos).T
    return scores * -1


def getDiag(num1, num2):
    return MATCH if -TOLERANCE <= int(num1) - int(num2) <= TOLERANCE else SUB

def backtrack(query, ref, score):
    # Traceback and compute the alignment 
    m, n = query.shape[0], ref.shape[0]

    aln = np.zeros(n)
    i,j = m,n # start from the bottom right cell
    while i > 0 and j > 0: # end touching the top or the left edge
        if score[i,j] == score[i-1,j-1] + getDiag(query[i-1], ref[j-1]):
            aln[j-1] = i - j 
            i -= 1
            j -= 1
        # score_left
        elif score[i,j] == score[i-1,j] + INDEL:
            i -= 1
        # score_up
        elif score[i,j] == score[i,j-1] + INDEL:
            aln[j-1] = aln[min(j, n-1)]
            j -= 1

    return aln

def needleman(leftSection, rightSection):
    res = np.zeros((leftSection.shape[0], leftSection.shape[1]))
    for i, (l, r) in enumerate(zip(leftSection, rightSection)):
        #bitparallel levenstein distance matrix
        score = getMatrix(l, r)

        # backtrack to get alignment 
        res[i,:] = backtrack(l, r, score)
    return res

def DP(imL, imR):
    # divide into sections for multiprocessing
    cpus = max(1, mp.cpu_count()-1)
    leftSections = np.array_split(imL, cpus)
    rightSections = np.array_split(imR, cpus)
    with mp.Pool(processes=cpus) as pool:
        disparity = pool.starmap(needleman, zip(leftSections,rightSections))
    disparity = np.vstack(disparity)

    # normalize so values are 0 - 255
    disparity = cv2.normalize(disparity,0,255,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    return disparity

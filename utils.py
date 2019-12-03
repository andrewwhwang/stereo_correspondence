import numpy as np

INPUT_DIR = "input"

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


def video(imDir):
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter("output/out.mp4", cv2.VideoWriter_fourcc(*'MP4V'), 30.0, (frameWidth,frameHeight))
    
    frameCounter = 1
    while(True):
        ret, frame = cap.read()
        if ret == True:
            print(frameCounter)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) / 255
            gray = cv2.GaussianBlur(gray,(3,3),1)
            
            u, v = ps4.hierarchical_lk(prev, gray, levels, k_size, k_type, 
                                        sigma, interpolation, border_mode)
                                        
            mask = quiver(u, v, scale=1, stride=13, color=(255,0,0))[:,:,0]
            frame[mask > 0] = (0,0,255)
        
            writer.write(frame)

            if frameCounter == 1:
                cv2.imwrite(os.path.join(output_dir, "ps4-6-a-1.png"), frame)
            elif frameCounter == 2:
                cv2.imwrite(os.path.join(output_dir, "ps4-6-a-2.png"), frame)

            prev = gray
            frameCounter += 1
        else: 
            break
        
    # When everything done, release the video capture object
    cap.release()
    writer.release()

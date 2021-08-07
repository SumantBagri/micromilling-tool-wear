def sortKeyFunc(s):
    import os
    return int(os.path.basename(s)[:-4])

def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

# Assuming that there ar only two contours which are disjoint
def patchContours(contour_list):
    import numpy as np
    if len(contour_list) != 2:
        print("Cant patch more than 2 disjoint contours!")
        return 0
    search_res = np.where(contour_list[0] == np.amax(contour_list[0],axis=0)[0][1])
    r_out_ind = search_res[0][np.amax(np.where(search_res[2] == 1))]
    r_out = contour_list[0][r_out_ind:]
    search_res = np.where(contour_list[1] == np.amax(contour_list[1],axis=0)[0][1])
    l_out_ind = search_res[0][np.amin(np.where(search_res[2] == 1))] + 1
    l_out = contour_list[1][:l_out_ind]
    new_cont_arr = l_out
    for val in np.arange(l_out[-1][0][0]+1,r_out[0][0][0]):
        patch_arr = [[[val,l_out[-1][0][1]]]]
        new_cont_arr = np.append(new_cont_arr, patch_arr, axis=0)
    new_cont_arr = np.append(new_cont_arr,r_out,axis=0)
    for val in np.arange(r_out[-1][0][0]-1,l_out[0][0][0],-1):
        patch_arr = [[[val,0]]]
        new_cont_arr = np.append(new_cont_arr, patch_arr, axis=0)
    return [new_cont_arr]

def getBoundingCircle(contours,drawing=None,wear_dict=None,ppm=1):
    import cv2
    import numpy as np
    if len(contours) > 1:
        contours = patchContours(contours)
        if contours == 0:
            print("Too many contours!!")
            return
    contours_poly = cv2.approxPolyDP(contours[0], 3, True)
    boundRect = cv2.boundingRect(contours_poly)
    center, cradius = cv2.minEnclosingCircle(contours_poly)
    radius = int(boundRect[2]/2)
    if isinstance(drawing, np.ndarray):
        color = (255, 0, 255)
        cv2.drawContours(drawing, contours_poly, -1, color)
        cv2.rectangle(drawing, (int(boundRect[0]), int(boundRect[1])), \
            (int(boundRect[0]+boundRect[2]), int(boundRect[1]+boundRect[3])), color, 2)
        cv2.circle(drawing, (int(center[0]), int(center[1])), int(cradius), color, 2)
        cv2.putText(drawing, "{:.2f}um".format((2*radius)/ppm),\
                    (int(center[0]), int(center[1])), cv2.FONT_HERSHEY_SIMPLEX, \
                    1, (255, 0, 0), 2)
        dRed = round(500 - (2*radius)/ppm,2)
        if wear_dict is not None:
            wear_dict['dRed'] = dRed
        return
    return center, cradius

def getPPM(path):
    import cv2
    imPath = path+'\\0.jpg'
    im0 = cv2.imread(imPath)
    im_gray = cv2.cvtColor(im0, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(im_gray, (61,61), 0)
    im_thresh = cv2.threshold(blur,91,255,cv2.THRESH_BINARY)[1]
    canny_output = cv2.Canny(im_thresh, 80, 100)
    edged = cv2.dilate(canny_output, None, iterations=10)
    edged = cv2.erode(edged, None, iterations=8)
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    _, radius = getBoundingCircle(contours)
    return (radius/250)

def euclideanAlign(im1, im2):
    import cv2
    import numpy as np
    # Find size of image1
    sz = im1.shape
    # Define the motion model
    warp_mode = cv2.MOTION_EUCLIDEAN
    # Define 2x3 matrix and initialize the matrix to identify
    warp_matrix = np.eye(2,3, dtype=np.float32)
    # Specify the number of iterations.
    number_of_iterations = 500;
    # Specify the threshold of the increment
    # in the correlation coefficient between two iterations
    termination_eps = 2e-8;
    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)
    # Run the ECC algorithm. The results are stored in warp_matrix.
    (cc, warp_matrix) = cv2.findTransformECC (im1,im2,warp_matrix, warp_mode, criteria,None,1)
    # Use warpAffine for Translation, Euclidean and Affine
    im2_aligned = cv2.warpAffine(im2, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);
    return im2_aligned

def getBinBaseIm(im0_path):
    import cv2
    orig = cv2.imread(im0_path)
    dst_base = cv2.fastNlMeansDenoisingColored(orig,None,10,10,7,21)
    im_gray_base = cv2.cvtColor(dst_base, cv2.COLOR_BGR2GRAY)
    im_thresh_base = cv2.threshold(im_gray_base,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    im_fin_base = cv2.dilate(im_thresh_base, None, iterations=20)
    im_fin_base = cv2.erode(im_fin_base, None, iterations=23)
    return im_fin_base

def getConnectivityMask(diff_im):
    import cv2
    import numpy as np
    from skimage import measure
    labels = measure.label(diff_im, connectivity=1)
    mask = np.zeros(diff_im.shape, dtype="uint8")
    # loop over the unique components
    for label in np.unique(labels):
        # if this is the background label, ignore it
        if label == 0:
            continue
        # otherwise, construct the label mask and count the
        # number of pixels
        labelMask = np.zeros(diff_im.shape, dtype="uint8")
        labelMask[labels == label] = 255
        numPixels = cv2.countNonZero(labelMask)
        # if the number of pixels in the component is sufficiently
        # large, then add it to our mask of "large blobs"
        if numPixels > 1000:
            mask = cv2.add(mask, labelMask)
    return mask

def updateOrient(cnts, edges, wear_dict):
    import numpy as np
    if edges == '1l':
        pixRange = [np.ptp(cnts[0],0)[0],0]
    elif edges == '1r':
        pixRange = [0, np.ptp(cnts[0],0)[0]]
    elif edges == '2':
        pixRange = [np.ptp(cnts[0],0)[0],np.ptp(cnts[1],0)[0]]
    else:
        pixRange = [0,0]
    # Update E1 orientation
    if isinstance(pixRange[0], np.ndarray):
        wear_dict['orE1'] = 2 if pixRange[0][0] > pixRange[0][1]+30 else 1
    else:
        wear_dict['orE1'] = 0
    # Update E2 orientation
    if isinstance(pixRange[1], np.ndarray):
        wear_dict['orE2'] = 2 if pixRange[1][0] > pixRange[1][1]+30 else 1
    else:
        wear_dict['orE2'] = 0

def wearDataHandler(im_fin_base, pixelsPerMetric, fileList, savefig):
    import os
    import cv2
    import numpy as np
    import pandas as pd
    import collections
    from scipy.spatial import distance as dist
    from skimage import measure
    from imutils import perspective, grab_contours
    import matplotlib.pyplot as plt
    wearDf = pd.DataFrame(columns=['slot', 'orE1', 'orE2', 'dRed', \
    'wearAE1', 'wear1E1', 'wear2E1', \
    'wearAE2', 'wear1E2', 'wear2E2'])
    wearDf['slot'] = wearDf['orE1'] = wearDf['orE2'] = wearDf['wear1E1'] =\
     wearDf['wear2E1'] = wearDf['wear1E2'] = wearDf['wear2E2'] = 0
    if len(fileList):
        bPath = os.path.dirname(fileList[0])+'\\wearData'
        for file in fileList:
            wear_dict = {}
            slot = int(file.split('\\')[-1][:-4])
            wear_dict['slot'] = slot
            #--------------Read in the current image file
            orig_inter = cv2.imread(file)
            dst = cv2.fastNlMeansDenoisingColored(orig_inter,None,10,10,7,21)
            #--------------Image Binarization
            im_gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
            im_thresh = cv2.threshold(im_gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
            im_thresh = cv2.dilate(im_thresh, None, iterations=22)
            im_thresh = cv2.erode(im_thresh, None, iterations=22)
            #--------------Remove unnecessary blobs from the binary image
            labels = measure.label(im_thresh, connectivity=1)
            labelMask = np.zeros(im_thresh.shape, dtype="uint8")
            count_dict = dict(collections.Counter(labels[0]))
            for i in range(1,labels.shape[0]):
                temp_dict = dict(collections.Counter(labels[i]))
                for key in temp_dict.keys():
                    if key not in count_dict.keys():
                        count_dict[key] = temp_dict[key]
                    else:
                        count_dict[key] += temp_dict[key]
            count_dict = {k: v for k, v in sorted(count_dict.items(), key=lambda item: item[1],reverse=True)}
            labelMask[labels == list(count_dict.keys())[1]] = 255
            im_thresh = labelMask
            #--------------Remove unwanted patches from the binary image
            im_fin = cv2.dilate(im_thresh, None, iterations=20)
            im_fin = cv2.erode(im_fin, None, iterations=23)
            #--------------Align the base and current images
            im_fin_al = euclideanAlign(im_fin_base,im_fin)
            im_fin_al = cv2.threshold(im_fin_al,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
            print('S{} Align Done!'.format(slot), end='|')
            #--------------Define drawing to be saved
            orig = cv2.absdiff(im_fin_base, im_fin_al)
            orig = cv2.cvtColor(orig,cv2.COLOR_GRAY2RGB)
            #--------------Get boundary edge of current image
            canny_output = cv2.Canny(im_fin_al, 80, 100)
            edged = cv2.dilate(canny_output, None, iterations=10)
            edged = cv2.erode(edged, None, iterations=8)
            #--------------Update wear data with dia reduction value
            bcnts = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            bcnts = grab_contours(bcnts)
            getBoundingCircle(bcnts,orig,wear_dict,pixelsPerMetric)
            #--------------Get difference with base image
            diff_im = cv2.absdiff(im_fin_base, im_fin_al)
            diff_im = cv2.erode(diff_im, None, iterations=1)
            mask = getConnectivityMask(diff_im)
            mask = cv2.erode(mask, None, iterations=5)
            print('S{} Mask Done!'.format(slot),end='|')
            #--------------Get boundary edges of wear regions
            edged = cv2.Canny(mask,1000,1200)
            edged = cv2.dilate(edged, None, iterations=2)
            edged = cv2.erode(edged, None, iterations=1)
            #--------------Get the contours for the blobs in edged
            cnts = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = grab_contours(cnts)
            #--------------Wear classification based on number of contours
            edges = '0'
            if len(cnts) >= 2:
                edges = '2'
                min_max_arr = np.array([[np.amin(cnts[i],0)[0][0], np.amax(cnts[i],0)[0][0]] \
                                        for i in range(len(cnts))])
                cnts = [cnts[np.argmin(min_max_arr,0)[0]],\
                        cnts[np.argmax(min_max_arr,0)[1]]]
                wear_dict['wearAE1'] = round(cv2.contourArea(cnts[0])/pow(pixelsPerMetric,2),2)
                wear_dict['wearAE2'] = round(cv2.contourArea(cnts[1])/pow(pixelsPerMetric,2),2)
            elif len(cnts) == 1:
                xMid = int((mask.shape[1])/2)
                #--------------E1 if xMax to the left of xMid
                if np.amax(cnts[0],0)[0][0] < xMid:
                    wear_dict['wear1E2'] = 0
                    wear_dict['wear2E2'] = 0
                    wear_dict['wearAE2'] = 0
                    wear_dict['wearAE1'] = round(cv2.contourArea(cnts[0])/pow(pixelsPerMetric,2),2)
                    edges = '1l'
                #--------------E2 if xMax to the right of xMid
                else:
                    wear_dict['wear1E1'] = 0
                    wear_dict['wear2E1'] = 0
                    wear_dict['wearAE1'] = 0
                    wear_dict['wearAE2'] = round(cv2.contourArea(cnts[1])/pow(pixelsPerMetric,2),2)
                    edges = '1r'
            else:
                print("0 contours detected!")
                wear_dict['wear1E1'] = 0
                wear_dict['wear2E1'] = 0
                wear_dict['wear1E2'] = 0
                wear_dict['wear2E2'] = 0
            #--------------Update the orientation of the wear on each edge
            updateOrient(cnts, edges, wear_dict)
            print('S{} Contour Done!'.format(slot),end='|')
            for i in range(len(cnts)):
                box = cv2.minAreaRect(cnts[i])
                box = cv2.boxPoints(box)
                box = np.array(box, dtype="int")
                box = perspective.order_points(box)
                _ = cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 3)
                for (x, y) in box:
                    cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)
                (tl, tr, br, bl) = box
                (tltrX, tltrY) = midpoint(tl, tr)
                (blbrX, blbrY) = midpoint(bl, br)
                (tlblX, tlblY) = midpoint(tl, bl)
                (trbrX, trbrY) = midpoint(tr, br)
                # draw the midpoints on the image
                _ = cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
                _ = cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
                _ = cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
                _ = cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)
                # draw lines between the midpoints
                _ = cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
                (0, 255, 0), 3)
                _ = cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
                            (0, 255, 0), 3)
                # compute the Euclidean distance between the midpoints
                dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
                dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
                # compute the size of the object
                dimA = dA / pixelsPerMetric
                dimB = dB / pixelsPerMetric
                # draw the object sizes on the image
                _ = cv2.putText(orig, "{:.2f}um".format(dimA),
                            (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (255, 0, 0), 2)
                _ = cv2.putText(orig, "{:.2f}um".format(dimB),
                            (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (255, 0, 0), 2)
                if edges == '2':
                    wear_dict['wear1E{}'.format(i+1)] = round(max(dimA,dimB),2)
                    wear_dict['wear2E{}'.format(i+1)] = round(min(dimA,dimB),2)
                elif edges == '1l':
                    wear_dict['wear1E1'.format(i+1)] = round(max(dimA,dimB),2)
                    wear_dict['wear2E1'.format(i+1)] = round(min(dimA,dimB),2)
                elif edges == '1r':
                    wear_dict['wear1E2'.format(i+1)] = round(max(dimA,dimB),2)
                    wear_dict['wear2E2'.format(i+1)] = round(min(dimA,dimB),2)
                else:
                    continue
            # Append data into DataFrame
            wearDf = wearDf.append(wear_dict, ignore_index=True)
            if savefig:
                # save the output image
                fname = '\\'+str(slot)+'wear.png'
                plt.imsave(bPath+fname,orig)
                print('S{} Im saved!'.format(slot))
        return wearDf
    else:
        return wearDf

import os
import glob
import time
import pickle
import traceback

import numpy as np
import pandas as pd
import ipyparallel as ipp

from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

from tqdm.notebook import tqdm
#from ipyparallel import Client


def sortKeyFunc(s):
    return int(os.path.basename(s).replace('_','')[:-4])

def getWCSS(X):
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    plt.plot(range(1, 11), wcss)
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()

def plot_mac_data(mac_data,savefig,key):
    fig, axs = plt.subplots(3,2,sharex=True,figsize=(15,15))

    axs[0,0].plot(mac_data.time,mac_data.fx)
    axs[0,0].set(xlabel='time(s)', ylabel='fx(N)')
    axs[0,0].grid()
    axs[1,0].plot(mac_data.time,mac_data.fy)
    axs[1,0].set(xlabel='time(s)', ylabel='fy(N)')
    axs[1,0].grid()
    axs[2,0].plot(mac_data.time,mac_data.fz)
    axs[2,0].set(xlabel='time(s)', ylabel='fz(N)')
    axs[2,0].grid()

    axs[0,1].plot(mac_data.time,mac_data.ax,mcolors.CSS4_COLORS['maroon'])
    axs[0,1].set(xlabel='time(s)', ylabel='ax(m/s^2)')
    axs[0,1].grid()
    axs[1,1].plot(mac_data.time,mac_data.ay,mcolors.CSS4_COLORS['maroon'])
    axs[1,1].set(xlabel='time(s)', ylabel='ay(m/s^2)')
    axs[1,1].grid()
    axs[2,1].plot(mac_data.time,mac_data.az,mcolors.CSS4_COLORS['maroon'])
    axs[2,1].set(xlabel='time(s)', ylabel='az(m/s^2)')
    axs[2,1].grid()

    if savefig == 'y':
        file_suffix = 'macForceAcc.png'
        plt.savefig(path+'\\logs\\'+key+file_suffix)
        
    plt.close()

def getSegmentedDataKmeans(data_inst,key,remove_thresh,savefig):
    data_req = data_inst.drop(np.where(abs(data_inst.fx) < remove_thresh)[0]).reset_index(drop=False)
    X = data_req[['time','fx']]
    
    kmeans = KMeans(n_clusters=10, init='k-means++', max_iter=300, n_init=10, random_state=0).fit(X)

    fig = plt.figure(figsize=(15,15))
    plt.plot(data_inst.time, data_inst.fx)
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=5000, c='red')

    if savefig == 'y':
        file_suffix = 'clusters.png'
        plt.savefig(path+'\\logs\\'+key+file_suffix)
    #plt.show()
    plt.close()

    _, idx = np.unique(kmeans.labels_,return_index=True)
    label_order_arr = kmeans.labels_[np.sort(idx)]

    mac_data_arr = np.where(kmeans.labels_ == label_order_arr[1])[0]
    for label in label_order_arr[2:-1]:
        mac_data_arr = np.union1d(mac_data_arr,np.where(kmeans.labels_ == label))

    si = data_req['index'][mac_data_arr[0]]
    ei = data_req['index'][mac_data_arr[-1]]

    mac_data = data_inst.iloc[si:ei].reset_index(drop=True)

    plot_mac_data(mac_data,savefig,key)

    sdd1 = data_req['index'][np.where(kmeans.labels_ == label_order_arr[0])[0][0]]
    edd1 = data_req['index'][np.where(kmeans.labels_ == label_order_arr[0])[0][-1]]

    sdd2 = data_req['index'][np.where(kmeans.labels_ == label_order_arr[-1])[0][0]]
    edd2 = data_req['index'][np.where(kmeans.labels_ == label_order_arr[-1])[0][-1]]

    dd1 = data_inst[sdd1:edd1]
    dd2 = data_inst[sdd2:edd2]

    #fig = plt.figure(figsize=(15,15))
    #plt.plot(dd1.time,dd1.fx)
    #plt.grid()

    #fig = plt.figure(figsize=(15,15))
    #plt.plot(dd2.time,dd2.fx)
    #plt.grid()
    
    return mac_data, dd1, dd2

def getSegmentedDataPeaks(data_inst):

    rc = ipp.Client(context=zmq.Context())
    dv = rc[:]

    aSync1 = dv.scatter('data_inst', data_inst) 

    aSync2 = dv.execute(
    r'''
    import os
    os.chdir(r'C:\Users\sbagr\Desktop\DDP_home\models')

    from functions import getExtremas

    ext = getExtremas(data_inst.fx)

    while True:
        temp = getExtremas(data_inst.fx[ext])    
        if len(temp) < 625:
            break
        else:
            ext = temp    
    '''
    )

    ext = dv.gather('ext', block=True)
    r"""
    ext = getExtremas(data_inst.fx)
    
    while True:
        temp = getExtremas(data_inst.fx[ext])    
        if len(temp) < 15000:
            break
        else:
            ext = temp 
    """
    jump_ind_lst = list(data_inst.fx.iloc[data_inst.fx.iloc[ext].index[np.where(abs(data_inst.fx.iloc[ext].diff()) > 0.6)]].index)
            
    si = np.where(np.diff(jump_ind_lst) > 10000)[0][0] + 1
    mac_si = jump_ind_lst[si]
    ei = np.where(np.diff(jump_ind_lst) > 10000)[0][-1]
    mac_ei = jump_ind_lst[ei]
    
    return mac_si, mac_ei

def initializeLogging(path): 
    if not os.path.exists(path+'\\logs'):
        os.mkdir(path+'\\logs')
        with open(path+'\\logs\\logfile.txt','w') as logf:
            pass
        
def getFileList(path,logger):
    fileList = glob.glob(path+r'\*.lvm')

    loop1_desc = path.split('\\')[-1].split('_')[0] + '_ren'
    
    for file in tqdm(fileList, desc=loop1_desc):
        try:
            logger.info("Renaming {}...".format(file))
            os.rename(file, path+'\\'+os.path.basename(file).replace('s',''))
        except:
            logger.info("Renaming already done!")
            continue

    fileList = glob.glob(path+r'\*.lvm')
    fileList.sort(key=sortKeyFunc)   

    return fileList

# Identifies all the peaks/valleys in the dataset and returns array with corresponding indices
def getExtremas(data):
    extremas_ind_lst = []
    idxLst = list(data.index)
    
    for i in range(len(idxLst)):
        if i == 0:
            if abs(data[idxLst[i+1]]) < abs(data[idxLst[i]]):
                extremas_ind_lst.append(idxLst[i])
        elif i == len(idxLst)-1:
            if abs(data[idxLst[i-1]]) < abs(data[idxLst[i]]):
                extremas_ind_lst.append(idxLst[i])
        else:
            if abs(data[idxLst[i-1]]) < abs(data[idxLst[i]]):
                if abs(data[idxLst[i+1]]) < abs(data[idxLst[i]]):
                    extremas_ind_lst.append(idxLst[i])
                else:
                    continue
            else:
                continue
    
    return extremas_ind_lst

# Plot the peak/valley (assumes data is of fx v/s time)
def plotExtremas(X, mac_si, mac_ei, path, key, savefig):
    fig = plt.figure(figsize=(15,15))

    plt.scatter(X.time,X.fx)
    plt.scatter(X.time.iloc[:int(mac_si/2)],X.fx.iloc[:int(mac_si/2)],color='g')
    plt.scatter(X.time.iloc[mac_si:mac_ei],X.fx.iloc[mac_si:mac_ei],color='r')
    plt.scatter(X.time.iloc[int((X.index[-1]+mac_ei)/2):],X.fx.iloc[int((X.index[-1]+mac_ei)/2):],color='g')

    plt.xlabel('time(s)')
    plt.ylabel('fx(N)')
    plt.grid()
    
    if savefig == 'y':
        file_suffix = 'macData.png'
        plt.savefig(path+'\\logs\\'+key+file_suffix)
    
    plt.close()


def segmentDataHandler(path,savefig,logger):
    remove_thresh = 0.25
    
    segmented_data_dict = {
                            'mac_data' : {},
                            'dd1' : {},
                            'dd2' : {}
                          }

    logf = open(path+'\\logs\\logfile.txt','w')
    bad_files = []
    loop_desc = path.split('\\')[-1].split('_')[0] + '_seg'
    fileList = getFileList(path,logger)

    columns = ['time','fx','fy','fz','ax','ay','az','vv']
    
    for file in tqdm(fileList, desc=loop_desc):
        key = str(os.path.basename(file)[:-4])
        logger.info("Importing raw data from {}".format(str(os.path.basename(file))))
        try:
            data_inst = pd.read_csv(file, header=None, names=columns,sep='\t')
            # Multiplication factor to get actual values
            data_inst.fx *= 5
            data_inst.fy *= 5
            data_inst.fz *= 5
            data_inst.ax *= 49
            data_inst.ay *= 49
            data_inst.az *= 49
            logger.info("Data import successful")
        except Exception as e:
            logf.write("Error in importing data\n : {}".format(str(e)))
            track = traceback.format_exc()
            logf.write("\n\t{}".format(track))
            logger.info("Error importing data. Check logfile...")
            return 0

        logger.info("Performing segmentation for file {0}".format(key+'.lvm'))

        #getWCSS(test_data.drop(np.where(abs(test_data.fx) < 0.1)[0])[['time','fx']])

        try:
            mac_si, mac_ei = getSegmentedDataPeaks(data_inst)
            plotExtremas(data_inst, mac_si, mac_ei, path, key, savefig)

            #mac_data,dd1,dd2 = getSegmentedDataKmeans(data_inst,key,remove_thresh,savefig)

            segmented_data_dict['mac_data'][key] = data_inst.iloc[mac_si:mac_ei]
            segmented_data_dict['dd1'][key] = data_inst[:int(mac_si/2)]
            segmented_data_dict['dd2'][key] = data_inst[int((data_inst.index[-1]+mac_ei)/2):]

            logf.write("Data Segmentation succeeded for {0}: {1}\n".format(str(key), 'OK')) 

        except Exception as e:
            logf.write("Data Segmentation Failed for {0}: {1}\n".format(str(key), str(e)))
            track = traceback.format_exc()
            logf.write("\n\t{}\n".format(track))
            bad_files.append(key)

    logf.close()

    logger.info("Pickling segmented data...!") 
    with open(path+'\\seg_data_dict.pickle', 'wb') as handle:
        pickle.dump(segmented_data_dict,handle)
        
    if len(bad_files):
        logger.info("Something nasty happened with these files : {}".format(bad_files))     
        return 0
    else:      
        return 1
import numpy as np
import matplotlib.pyplot as plt

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
def plotExtremas(data, extremas_ind_lst):
    fig = plt.figure(figsize=(15,15))
    plt.scatter(data.index[extremas_ind_lst],data.fx.iloc[extremas_ind_lst],color='r')
    plt.show()
# Plot the peak/valley (assumes data is of fx v/s time)
def plotExtremas(X, mac_si, mac_ei, ss_lst, se_lst, path, key, savefig):
    import matplotlib as mpl
    mpl.use('Agg')
    mpl.rcParams.update({'font.size': 30})
    from matplotlib import pyplot as plt
    _, axs = plt.subplots(2, 1, sharey=True, figsize=(15,15))
    axs[0].spines["right"].set_visible(False)
    axs[0].spines["top"].set_visible(False)
    axs[0].spines["bottom"].set_linewidth(2)
    axs[0].spines["left"].set_linewidth(2)
    axs[0].tick_params(length=6, width=2)
    axs[0].scatter(X.time,X.fx)
    axs[0].scatter(X.time.iloc[:int(mac_si/2)],X.fx.iloc[:int(mac_si/2)],color='g')
    axs[0].scatter(X.time.iloc[mac_si:mac_ei],X.fx.iloc[mac_si:mac_ei],color='r')
    axs[0].scatter(X.time.iloc[int((X.index[-1]+mac_ei)/2):],X.fx.iloc[int((X.index[-1]+mac_ei)/2):],color='g')
    axs[0].set_xlabel('time(s)\n(a)')
    axs[0].set_ylabel('fx(N)')
    #axs[0].grid()
    axs[1].spines["right"].set_visible(False)
    axs[1].spines["top"].set_visible(False)
    axs[1].spines["bottom"].set_linewidth(2)
    axs[1].spines["left"].set_linewidth(2)
    axs[1].tick_params(length=6, width=2)
    for i in range(len(ss_lst)):
        axs[1].scatter(X.time[ss_lst[i]:se_lst[i]],X.fx[ss_lst[i]:se_lst[i]])
    axs[1].set_xlabel('time(s)\n(b)')
    axs[1].set_ylabel('fx(N)')
    plt.subplots_adjust(hspace=0.3)
    #axs[1].grid()
    if savefig == 'y':
        file_suffix = 'macData.png'
        plt.savefig(path+'\\logs\\'+key+file_suffix)
    plt.close()

# Calculates the points per slot for given machining parameters
def getPPS(path):
    import numpy as np
    param_list = path.split('\\')[-1].split('_')
    rpm = int(path.split('\\')[-1].split('_')[1][:-1])*1000
    r = int(path.split('\\')[-1].split('_')[3][:-6]) # in um
    feed = int(param_list[4]) #fpt
    feed_rate = (2*feed*rpm)/(60000)
    pps = (np.pi*r)/(0.00005*feed_rate*1000)
    return pps

# Get threshold value for data slicing
def getThreshold(n_slots):
    import numpy as np
    if n_slots >= 15:
        return int(1700/(pow(n_slots,0.75)*np.sin(np.pi/(2*n_slots))))
    else:
        return n_slots*8500

# Correct for Drift Error
def rectifyDriftError(data_inst, mac_si, mac_ei, lines, key):
    from sklearn.linear_model import LinearRegression
    import pandas as pd
    dd = pd.concat((data_inst[:int(mac_si/2)],data_inst[int((data_inst.index[-1]+mac_ei)/2):]),axis=0)
    drift_corr_eps = 0.01 # Units in [Force:N/sec, Acc:m/sec^3, Vib:m/sec^2]
    ddp = key+','
    for col in data_inst.columns.drop('time'):
        lin_reg = LinearRegression().fit(dd.time.values.reshape(-1,1),dd[col].values.reshape(-1,1))
        err_line = lin_reg.predict(data_inst[mac_si:mac_ei+1].time.values.reshape(-1,1))
        if col == 'vv':
            ddp += str(lin_reg.coef_[0][0])+'\n'
        else:
            ddp += str(lin_reg.coef_[0][0])+','
        if abs(lin_reg.coef_) >= drift_corr_eps:
            if lin_reg.coef_ < 0:
                data_inst.loc[mac_si:mac_ei, col] = data_inst.loc[mac_si:mac_ei, col].values + err_line.flat
            else:
                data_inst.loc[mac_si:mac_ei, col] = data_inst.loc[mac_si:mac_ei, col].values - err_line.flat
        else:
            continue
    lines.append(ddp)

# Identifies all the peaks/valleys in the dataset and returns array with corresponding indices
def getExtremas(data,doPos,doNeg):
    pidxLst = list(data[(data > 0)].index)
    nidxLst = list(data[(data < 0)].index)
    ext = {'ppeaks':pidxLst,'npeaks':nidxLst}
    # For positive data points
    if doPos:
        ext['ppeaks'] = []
        for i in range(len(pidxLst)):
            if i == 0:
                if data[pidxLst[i+1]] < data[pidxLst[i]]:
                    ext['ppeaks'].append(pidxLst[i])
            elif i == len(pidxLst)-1:
                if data[pidxLst[i-1]] < data[pidxLst[i]]:
                    ext['ppeaks'].append(pidxLst[i])
            else:
                if (data[pidxLst[i-1]] < data[pidxLst[i]]) and (data[pidxLst[i+1]] < data[pidxLst[i]]):
                    ext['ppeaks'].append(pidxLst[i])
                else:
                    continue
    # For negative data points
    if doNeg:
        ext['npeaks'] = []
        for i in range(len(nidxLst)):
            if i == 0:
                if data[nidxLst[i+1]] > data[nidxLst[i]]:
                    ext['npeaks'].append(nidxLst[i])
            elif i == len(nidxLst)-1:
                if data[nidxLst[i-1]] > data[nidxLst[i]]:
                    ext['npeaks'].append(nidxLst[i])
            else:
                if (data[nidxLst[i-1]] > data[nidxLst[i]]) and (data[nidxLst[i+1]] > data[nidxLst[i]]):
                    ext['npeaks'].append(nidxLst[i])
                else:
                    continue
    return ext

def getConstAmp(data,n_slots):
    import pandas as pd
    ext = getExtremas(data.fx,1,1)
    thresh = getThreshold(n_slots)
    while True:
        cext = ext['ppeaks'] + ext['npeaks']
        if (len(ext['ppeaks']) <= thresh) and (len(ext['npeaks']) <= thresh):
            break
        elif len(ext['ppeaks']) <= thresh:
            ext = getExtremas(data.fx[cext],0,1)
        elif len(ext['npeaks']) <= thresh:
            ext = getExtremas(data.fx[cext],1,0)
        else:
            ext = getExtremas(data.fx[cext],1,1)
    nwin = int(0.025*len(ext['npeaks']))
    nwin = int(nwin/pow(10,len(str(nwin))-1))*pow(10,len(str(nwin))-1)
    pwin = int(0.025*len(ext['ppeaks']))
    pwin = int(pwin/pow(10,len(str(pwin))-1))*pow(10,len(str(pwin))-1)
    peak_idx_lst = []
    test_data = data.fx[ext['npeaks']].rolling(window=nwin).min().dropna()
    test_data = pd.DataFrame(test_data,columns=['fx'])
    midR = list((test_data.min() + test_data.max())/2)[0] if n_slots <= 15 else 0
    test_data = test_data[(test_data.fx < midR)]
    test_data['mav'] = test_data.fx
    gb = test_data.groupby(['mav']).agg(['count'])
    pList = list(gb[(gb[('fx', 'count')] >= nwin)][('fx', 'count')].index)
    for mav in pList:
        peak_idx_lst.append(test_data[test_data.mav == mav].index[0])
    test_data = data.fx[ext['ppeaks']].rolling(window=pwin).max().dropna()
    test_data = pd.DataFrame(test_data,columns=['fx'])
    midR = list((test_data.min() + test_data.max())/2)[0] if n_slots <= 15 else 0
    test_data = test_data[(test_data.fx > midR)]
    test_data['mav'] = test_data.fx
    gb = test_data.groupby(['mav']).agg(['count'])
    pList = list(gb[(gb[('fx', 'count')] >= pwin)][('fx', 'count')].index)
    for mav in pList:
        peak_idx_lst.append(test_data[test_data.mav == mav].index[0])
    peak_idx_lst.sort()
    return peak_idx_lst

def segmentDataHandler5(path,savefig,fileList,lines):
    import pandas as pd
    import numpy as np
    slot_data_dict = {}
    columns = ['time','fx','fy','fz','ax','ay','az','vv']
    pps = getPPS(path)
    for file in fileList:
        key = file.split('\\')[-1][:-4]
        print("Key-{}".format(key),end=" | ")
        n_slots = int(key.split('_')[1]) - int(key.split('_')[0]) + 1
        pad = 2000
        slot_data_dict[key] = []
        # Import raw data from target file
        data_inst = pd.read_csv(file, header=None, names=columns,sep='\t')
        # Multiplication factor to get actual values
        data_inst.fx *= 5
        data_inst.fy *= 5
        data_inst.fz *= 5
        data_inst.ax *= 49
        data_inst.ay *= 49
        data_inst.az *= 49
        data_inst.vv *= 1000
        # Identify machining data
        print("Data Imported",end=" | ")
        mu = data_inst.fx.mean()
        dNorm = data_inst[80000:-20000].fx - mu
        peak_thresh_um = np.quantile(dNorm[(dNorm > 0)],0.995) + mu
        peak_thresh_dm = np.quantile(dNorm[(dNorm < 0)],0.005) + mu
        idx_lst = list(data_inst[80000:-20000].fx[(data_inst.fx <= peak_thresh_dm)|(data_inst[80000:-20000].fx >= peak_thresh_um)].index)
        idx_lst.sort()
        mac_si = idx_lst[0] - 14000
        mac_ei = idx_lst[-1] + round(0.75*pps)
        print("MacData Done",end=" | ")
        # Rectify data for error due to drift in instrument
        rectifyDriftError(data_inst, mac_si, mac_ei, lines, key)
        print("Drift Rectified",end=" | ")
        peak_idx_lst = getConstAmp(data_inst[mac_si:mac_ei],n_slots)
        print("SlotData Done",end=" | ")
        ss_lst = [peak_idx_lst[0]-pad]
        se_lst = []
        for i in np.where(np.diff(peak_idx_lst) > 5000)[0]:
            ss_lst.append(peak_idx_lst[i+1]-pad)
            se_lst.append(peak_idx_lst[i]+pad)
        se_lst.append(peak_idx_lst[-1]+pad)
        for i in range(len(ss_lst)):
            slot_data_dict[key].append(data_inst[ss_lst[i]:se_lst[i]])
        plotExtremas(data_inst, mac_si, mac_ei, ss_lst, se_lst, path, key, savefig)
        print("Plot Saved",end=" | ")
        print("Done!!")
    return slot_data_dict

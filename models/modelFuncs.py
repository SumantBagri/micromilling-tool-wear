import matplotlib.pyplot as plt

import re
import numpy as np
import pandas as pd
import seaborn as sns
from paths import getTrainPathDict

import sklearn
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import OneHotEncoder

def saveRawData():
    import pickle
    paths_dict = getTrainPathDict()
    raw_data_path = r'C:\Users\sbagr\Desktop\DDP_home\data'

    # Get Raw Data
    raw_data = pd.DataFrame()
    for path in paths_dict.values():
        raw_data = pd.concat([raw_data, pd.read_pickle(path+'\\final_dataset.pickle')]).reset_index(drop=True)
    with open(raw_data_path+'\\raw_data.pickle','wb') as f:
        pickle.dump(raw_data,f)
    print("Pickled Raw Data Saved!")

def getPathData(fpath, key, for_dbn):
    data = pd.read_pickle(fpath)
    # Clean Raw Data
    data_set_dia,data_set_E1,data_set_E2,data_set_eotl = cleanData(data)
    # Get split_data_dict
    if key == 'dia':
        split_data_dict = getFeatLabSplit(data_set_dia,'diaRed_class',for_dbn)
    elif key == 'E1':
        split_data_dict = getFeatLabSplit(data_set_E1,'wearE1_class',for_dbn)
    elif key == 'E2':
        split_data_dict = getFeatLabSplit(data_set_E2,'wearE2_class',for_dbn)
    elif key == 'eotl':
        split_data_dict = getFeatLabSplit(data_set_eotl,'tteotl')
    else:
        print("Invalid Key ({})".format(key))
    # Normalize
    normalize(split_data_dict)
    return split_data_dict

def getFeatLabSplit(data,feature,for_dbn=None):
    split_data_dict = {}
    # For Classification
    if feature in ['diaRed_class','wearE1_class','wearE2_class']:
        if for_dbn:
            split_data_dict['labels'] = np.array(data.pop(feature)).reshape(-1).astype('float32')
        else:
            # One Hot Encode Labels
            ohe = OneHotEncoder(categories=[np.array([1,2,3],dtype='float64')], sparse=False)
            # Form np arrays of labels and features.
            split_data_dict['labels'] = np.array(ohe.fit_transform(np.array(data.pop(feature)).reshape(-1,1))).astype('float32')
        split_data_dict['features'] = np.array(data).astype('float32')
    else:
        split_data_dict['labels'] = np.array(data.pop(feature)).reshape(-1,1).astype('float32')
        split_data_dict['features'] = np.array(data).astype('float32')
    return split_data_dict


def getReqData(key, for_dbn):
    raw_data_fpath = r'C:\Users\sbagr\Desktop\DDP\DDP_repo\data\raw_data.pickle'
    raw_data = pd.read_pickle(raw_data_fpath)
    # Clean Raw Data
    data_set_dia,data_set_E1,data_set_E2,data_set_eotl = cleanData(raw_data)
    if key == 'dia':
        print("Fetching dRed data...")
        # Get Split Data
        split_data_dict = getClassDataSplits(data_set_dia,'diaRed_class',for_dbn)
    elif key == 'E1':
        print("Fetching E1 data...")
        # Get Split Data
        split_data_dict = getClassDataSplits(data_set_E1,'wearE1_class',for_dbn)
    elif key == 'E2':
        print("Fetching E2 data...")
        # Get Split Data
        split_data_dict = getClassDataSplits(data_set_E2,'wearE2_class',for_dbn)
    elif key == 'eotl':
        print("Fetching EoTL data...")
        # Get Split Data
        split_data_dict = getRegDataSplits(data_set_eotl,'tteotl')
    else:
        print("Invalid Key ({})".format(key))
    # Normalize
    normalize(split_data_dict, key)
    return split_data_dict

def plot_dbn_metrics(history,metrics):
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    roman_num = [ "(I)", "(II)", "(III)", "(IV)" ]
    for n, metric in enumerate(metrics):
        name = metric.replace("_"," ").capitalize()
        x_data = np.arange(1, len(history.__dict__[metric])+1, 1)
        ax = plt.subplot(2,2,n+1)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_linewidth(2)
        ax.spines["left"].set_linewidth(2)
        ax.tick_params(length=6, width=2)
        ax.plot(x_data,  history.__dict__[metric], color=colors[0], label='Train')
        ax.plot(x_data, history.__dict__['val_'+metric],
                 color=colors[0], linestyle="--", label='Val')
        ax.set_xlabel('Epoch\n{}'.format(roman_num[n]))
        ax.set_ylabel(name)
        if metric == 'loss' or metric == 'mean_squared_error' or metric == 'mean_absolute_error':
            plt.ylim([0, plt.ylim()[1]])
        elif metric == 'auc':
            plt.ylim([0.8,1])
        else:
            plt.ylim([0,1])
        ax.legend()
        plt.subplots_adjust(wspace=0.4)


def plot_metrics(history,metrics):
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    roman_num = [ "(I)", "(II)", "(III)", "(IV)" ]
    for n, metric in enumerate(metrics):
        name = metric.replace("_"," ").capitalize()
        ax = plt.subplot(2,2,n+1)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_linewidth(2)
        ax.spines["left"].set_linewidth(2)
        ax.tick_params(length=6, width=2)
        ax.plot(history.epoch,  history.history[metric], color=colors[0], label='Train')
        ax.plot(history.epoch, history.history['val_'+metric],
                 color=colors[0], linestyle="--", label='Val')
        ax.set_xlabel('Epoch\n{}'.format(roman_num[n]))
        ax.set_ylabel(name)
        if metric == 'loss' or metric == 'mean_squared_error' or metric == 'mean_absolute_error':
            plt.ylim([0, plt.ylim()[1]])
        elif metric == 'auc':
            plt.ylim([0.8,1])
        else:
            plt.ylim([0,1])
        ax.legend()
        plt.subplots_adjust(wspace=0.4, hspace=0.4)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    import itertools
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    #else:
        #print('Confusion matrix, without normalization')

    #print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

def plot_cm(labels, predictions, p=0.5):
    cm_arr = multilabel_confusion_matrix(labels, predictions > p)
    for i in range(3):
        cm = cm_arr[i]
        plt.figure(figsize=(5,5))
        sns.heatmap(cm, annot=True, fmt="d")
        plt.title('Confusion matrix @{:.2f} for label {}'.format(p,i))
        plt.ylabel('Actual label')
        plt.xlabel('Predicted label')

        print('Legitimate Transactions Detected (True Negatives): ', cm[0][0])
        print('Legitimate Transactions Incorrectly Detected (False Positives): ', cm[0][1])
        print('Fraudulent Transactions Missed (False Negatives): ', cm[1][0])
        print('Fraudulent Transactions Detected (True Positives): ', cm[1][1])
        print('Total Fraudulent Transactions: ', np.sum(cm[1]))

def plot_roc(name, labels, predictions, **kwargs):
    fp, tp, _ = sklearn.metrics.roc_curve(labels, predictions)
    plt.plot(100*fp, 100*tp, label=name, linewidth=2, **kwargs)
    plt.xlabel('False positives [%]')
    plt.ylabel('True positives [%]')
    plt.xlim([-0.5,20])
    plt.ylim([80,100.5])
    plt.grid(True)
    ax = plt.gca()
    ax.set_aspect('equal')

def fix_time(df):
    anomaly_list = [0]
    for i in range(1,len(df)):
        if df.time[i] < df.time[i-1]:
            anomaly_list.append(i)
    anomaly_list.append(len(df))
    for i in range(len(anomaly_list)-1):
        df.time[anomaly_list[i]:anomaly_list[i+1]] -= df.time[anomaly_list[i]]
        df.matR[anomaly_list[i]:anomaly_list[i+1]] -= df.matR[anomaly_list[i]]
    for i in range(1,len(anomaly_list)-1):
        df.time[anomaly_list[i]:anomaly_list[i+1]] += df.time[anomaly_list[i]-1] + 0.74
        df.matR[anomaly_list[i]:anomaly_list[i+1]] += df.matR[anomaly_list[i]-1] + 0.74

def cleanData(data_set):
    # Clean Data
    data_set['adoc'] = pd.to_numeric(data_set['adoc'])
    data_set['feed'] = pd.to_numeric(data_set['feed'])
    data_set['rpm'] = pd.to_numeric(data_set['rpm'])
    data_set['tpr'] = pd.to_numeric(data_set['tpr'])
    data_set.drop(['diaRed', 'wearE1', 'wearE2'], axis=1, inplace=True)
    # Three different output class for each model
    # One for regression model
    data_set_dia = data_set.drop(['wearE1_class', 'wearE2_class', 'tteotl'], axis=1) # For Classification
    data_set_E1 = data_set.drop(['diaRed_class', 'wearE2_class', 'tteotl'], axis=1) # For Classification
    data_set_E2 = data_set.drop(['diaRed_class', 'wearE1_class', 'tteotl'], axis=1) # For Classification
    data_set_eotl = data_set.drop(['diaRed_class', 'wearE1_class', 'wearE2_class'], axis=1) # For Regression
    return data_set_dia,data_set_E1,data_set_E2,data_set_eotl

# Returns the labels and features for train-val-test splits
def getClassDataSplits(data,feature,for_dbn):
    # Train Test Valid Split
    train1, test1 = train_test_split(data, test_size=0.3, stratify=data[feature])
    train_df = pd.DataFrame(train1)
    test_df = pd.DataFrame(test1)
    train2, valid1 = train_test_split(train_df, test_size=0.2, stratify=train_df[feature])
    train_df = pd.DataFrame(train2)
    valid_df = pd.DataFrame(valid1)
    """
    train1, test1 = train_test_split(data.iloc[np.where(data[feature] == 1)[0]], test_size=0.3)
    train2, test2 = train_test_split(data.iloc[np.where(data[feature] == 2)[0]], test_size=0.3)
    train3, test3 = train_test_split(data.iloc[np.where(data[feature] == 3)[0]], test_size=0.3)
    train_df = pd.concat([train1, train2, train3])
    test_df = pd.concat([test1, test2, test3])
    # Train Validation Split
    train1, valid1 = train_test_split(train_df.iloc[np.where(train_df[feature] == 1)[0]], test_size=0.2)
    train2, valid2 = train_test_split(train_df.iloc[np.where(train_df[feature] == 2)[0]], test_size=0.2)
    train3, valid3 = train_test_split(train_df.iloc[np.where(train_df[feature] == 3)[0]], test_size=0.2)
    train_df = pd.concat([train1, train2, train3])
    valid_df = pd.concat([valid1, valid2, valid3])
    """
    # Shuffle after concat
    train_df = sklearn.utils.shuffle(train_df)
    valid_df = sklearn.utils.shuffle(valid_df)
    test_df = sklearn.utils.shuffle(test_df)
    split_data_dict = {'train':{}, 'val':{}, 'test':{}}
    if for_dbn:
        # Form np arrays of labels and features.
        split_data_dict['train']['train_labels'] = np.array(train_df.pop(feature)).reshape(-1).astype('float32')
        split_data_dict['val']['val_labels'] = np.array(valid_df.pop(feature)).reshape(-1).astype('float32')
        split_data_dict['test']['test_labels'] = np.array(test_df.pop(feature)).reshape(-1).astype('float32')
    else:
        # One Hot Encode Labels
        ohe = OneHotEncoder(sparse=False)
        # Form np arrays of labels and features.
        split_data_dict['train']['train_labels'] = np.array(ohe.fit_transform(np.array(train_df.pop(feature)).reshape(-1,1))).astype('float32')
        split_data_dict['val']['val_labels'] = np.array(ohe.fit_transform(np.array(valid_df.pop(feature)).reshape(-1,1))).astype('float32')
        split_data_dict['test']['test_labels'] = np.array(ohe.fit_transform(np.array(test_df.pop(feature)).reshape(-1,1))).astype('float32')
    split_data_dict['train']['train_features'] = np.array(train_df).astype('float32')
    split_data_dict['val']['val_features'] = np.array(valid_df).astype('float32')
    split_data_dict['test']['test_features'] = np.array(test_df).astype('float32')
    return split_data_dict

def getRegDataSplits(data,feature):
    train_df, test_df = train_test_split(data, test_size=0.2)
    train_df, valid_df = train_test_split(train_df, test_size=0.2)
    # Shuffle after concat
    train_df = sklearn.utils.shuffle(train_df)
    valid_df = sklearn.utils.shuffle(valid_df)
    test_df = sklearn.utils.shuffle(test_df)
    split_data_dict = {'train':{}, 'val':{}, 'test':{}}
    # Form np arrays of labels and features.
    split_data_dict['train']['train_labels'] = np.array(train_df.pop(feature)).reshape(-1)
    split_data_dict['val']['val_labels'] = np.array(valid_df.pop(feature)).reshape(-1)
    split_data_dict['test']['test_labels'] = np.array(test_df.pop(feature)).reshape(-1)
    split_data_dict['train']['train_features'] = np.array(train_df)
    split_data_dict['val']['val_features'] = np.array(valid_df)
    split_data_dict['test']['test_features'] = np.array(test_df)
    return split_data_dict

# Normalize the dataset
def normalize(split_data_dict, feature='na'):
    # Normalization
    scaler = MinMaxScaler()
    if len(split_data_dict) == 3:
        split_data_dict['train']['train_features'] = scaler.fit_transform(split_data_dict['train']['train_features'])
        split_data_dict['val']['val_features'] = scaler.fit_transform(split_data_dict['val']['val_features'])
        split_data_dict['test']['test_features'] = scaler.fit_transform(split_data_dict['test']['test_features'])
        # Clipping
        #split_data_dict['train']['train_features'] = np.clip(split_data_dict['train']['train_features'], -5, 5)
        #split_data_dict['val']['val_features'] = np.clip(split_data_dict['val']['val_features'], -5, 5)
        #split_data_dict['test']['test_features'] = np.clip(split_data_dict['test']['test_features'], -5, 5)
        # If output label for regression then normalize
        if feature == 'eotl':
            print("Normalizing labels...")
            split_data_dict['train']['train_labels'] = scaler.fit_transform(split_data_dict['train']['train_labels'].reshape(-1, 1))
            split_data_dict['val']['val_labels'] = scaler.fit_transform(split_data_dict['val']['val_labels'].reshape(-1, 1))
            split_data_dict['test']['test_labels'] = scaler.fit_transform(split_data_dict['test']['test_labels'].reshape(-1, 1))
        # Printing Stats
        print('Training labels shape:', split_data_dict['train']['train_labels'].shape)
        print('Validation labels shape:', split_data_dict['val']['val_labels'].shape)
        print('Test labels shape:',split_data_dict['test']['test_labels'].shape)
        print('Training features shape:', split_data_dict['train']['train_features'].shape)
        print('Validation features shape:', split_data_dict['val']['val_features'].shape)
        print('Test features shape:', split_data_dict['test']['test_features'].shape)
    else:
        split_data_dict['features'] = scaler.fit_transform(split_data_dict['features'])
        #split_data_dict['features'] = np.clip(split_data_dict['features'], -5, 5)
        # Printing Stats
        print('Labels shape:', split_data_dict['labels'].shape)
        print('Features shape:', split_data_dict['features'].shape)

def denormalize(data, norm_data):
    scaler = MinMaxScaler()
    _ = scaler.fit_transform(data.reshape(-1,1))
    denorm_data = scaler.inverse_transform(norm_data)
    return denorm_data

# Get CV results table
def getCvTable(cv_results, n_splits, n_iter):
    result_dict = {}
    pattern = re.compile(r'param(?:_.[a-z]*)+')
    param_key_list = re.findall(pattern, ' '.join(list(cv_results.keys())))
    for it in range(n_iter):
        for key in param_key_list:
            if key not in result_dict.keys():
                result_dict[key] = []
            if isinstance(cv_results[key][it], list):
                result_dict[key].append(cv_results[key][it][0])
            else:
                result_dict[key].append(cv_results[key][it])
        for split_n in range(n_splits):
            key = 'split{}_test_score'.format(split_n)
            if key not in result_dict.keys():
                result_dict[key] = []
            result_dict[key].append(cv_results[key][it])
        if 'mean_test_score' not in result_dict:
            result_dict['mean_test_score'] = []
        result_dict['mean_test_score'].append(cv_results['mean_test_score'][it])
        if 'mean_fit_time' not in result_dict:
            result_dict['mean_fit_time'] = []
        result_dict['mean_fit_time'].append(cv_results['mean_fit_time'][it])
        if 'mean_score_time' not in result_dict:
            result_dict['mean_score_time'] = []
        result_dict['mean_score_time'].append(cv_results['mean_score_time'][it])
    return pd.DataFrame(result_dict).sort_values(by=['mean_test_score'], ascending=False).reset_index(drop=True)

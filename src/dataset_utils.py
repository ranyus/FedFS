from operator import sub
import pandas as pd
import sklearn as skl
import pickle as pkl
import numpy as np
from datetime import datetime 
import random as rnd

def load_voxel(return_Xy=False):
    raw_data = pd.read_csv(
        "../data/voxel/DataSruct.csv", delimiter=' ')
    if return_Xy:
        # separate patterns and labels

        X_all = raw_data.iloc[:, :-4]
        y_all = raw_data.iloc[:, [0, -4]]
        return X_all,y_all
    return raw_data




def load_wesad(decentralised=False, train_sbj=['S5', 'S14'], test_sbj=['S7'], sub_size=3000000, shuffled=False):
    '''
    Returns the dataset as dictionary where each record is of the form:
    Subject:{patterns:labels}
    '''
    dirs = ['S17', 'S15', 'S5', 'S14', 'S8', 'S2',
     'S16', 'S11', 'S9', 'S4', 'S10', 'S6', 'S3', 'S13', 'S7']
    
    
    train = None
    test = None    
    test_ok = set(test_sbj)
    train_ok = set(train_sbj)
    test_first = True
    train_first = True
    # test_idx = []
    # train_idx= [] 
    # test_count = 0    
    train_idx = None

    train_count = 0
    for usr in dirs:
        seed  = datetime.now().microsecond
        with open(f'/home/lorenzo/data/WESAD/{usr}.pkl', 'rb') as fd:
            data = pkl.load(fd)
        N = data[usr]['patterns'].shape[0]
        if usr in test_ok:
            
            if test_first:
                test = np.copy(np.c_[data[usr]['patterns'],data[usr]['labels']])
                # skip 5,6,7 labels as stated in the dataset documentation
                #                 
                test = test[test[:,-1]<5]
                
                
                test_first = False
                print('new test on {}'.format(
                    usr))
                
            else:
                tmp = np.c_[data[usr]['patterns'],data[usr]['labels']]
                tmp = tmp[tmp[:,-1] < 5]
                test = np.vstack((test,tmp))                    
                print('add test on {}'.format(usr))
                

        elif usr in train_ok:
            if train_first:
                train = np.copy(np.c_[data[usr]['patterns'], data[usr]['labels']])
                train = train[train[: ,-1] < 5]
                train = np.copy(train[:sub_size, :])
                idxs = np.arange(train_count, train_count +
                                 train.shape[0], dtype=int)
                if shuffled:
                    rnd.Random(seed).shuffle(idxs)
                    rnd.Random(seed).shuffle(train)
                
                train_first = False
                train_idx= np.copy(idxs)
                train_count = train.shape[0]
                print('new train on {}'.format(usr))
                
            else:
                tmp = np.c_[data[usr]['patterns'], data[usr]['labels']]
                tmp = tmp[tmp[:, -1] < 5]
                tmp = tmp[:sub_size, :]
                idxs = np.arange(train_count, train_count +
                                 tmp.shape[0], dtype=int)
                if shuffled:
                    rnd.Random(seed).shuffle(idxs)
                    rnd.Random(seed).shuffle(train)
                
                # N, d = data[usr]['patterns'].shape
                train = np.vstack((train, tmp))
                # train_idx.append(idxs)
                train_idx = np.vstack((train_idx,idxs))
                train_count = train.shape[0]
                print('add train on {}'.format(usr))
                
        else:
            print('Skipping {}'.format(usr))
            continue
        # fd.close()
    if decentralised:
        return  train, test, train_idx
    return train, test

    # else:
    #     train_data = {}
    #     test_data = {}
    #     test_ok = set(test_sbj)
    #     for usr in dirs:
    #         with open(f'/home/lorenzo/data/WESAD/{usr}.pkl','rb') as fd:
    #             data = pkl.load(fd)
    #             if usr in test_ok:
    #                 test_data[usr] = data[usr]
    #             else:
    #                 train_data[usr] = data[usr]

        

    #     return [train_data,test_data]

def main(): 
    # [train,test,idx] = load_wesad(decentralised=True)
    # train,test= load_wesad(train_sbj=['S17'],test_sbj=['S14'])
    train, test, idx = load_wesad(decentralised=True,
        train_sbj=['S17', 'S15'], test_sbj=['S14','S16'])
    
    print('{} {}'.format(train.shape,test.shape))
    print('labels {}'.format(np.unique(train[:,-1])))
    print ('idx len: {}'.format(([e.shape[0] for e in idx])))

if __name__ == "__main__":
    main()
    

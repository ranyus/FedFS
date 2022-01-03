import csv

import numpy as np

from sklearn import neural_network as nn
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline





# #####
# UTILS
# #####

# ML / STATS


def train_net(X, y, X_te, y_te, net=[300, 100], reps=10):
    '''
    Trains a NN for <reps> times. Returns the score. 
    '''
    print(f'Net{net}')
    score = []
    mlp=[]
    for run in range(reps):
        print('Run {}'.format(run))
        mlp.append(make_pipeline(StandardScaler(),nn.MLPClassifier(hidden_layer_sizes=net,solver='adam', activation='relu')))
        mlp[0].fit(X,y)                       
        score.append(mlp[0].score(X_te, y_te))
        # del mlp


    return score


def train_svm(X, y, X_te, y_te, reps=2):
    '''
    Trains a SVC for <reps> times. Returns the score. 
    '''
    print(f'Training SVM')
    score = []
    
    for run in range(reps):
        print('Run {}'.format(run))
        svc = make_pipeline(
            StandardScaler(), svm.SVC(gamma='scale', tol=1e-5))
        svc = svc.fit(X, y)
        score.append(svc.score(X_te, y_te))
        # del svc

    return score


def compute_95ci(data):
    ci = 1.833*(np.std(data)/np.sqrt(len(data)))
    avg = np.mean(data)
    return [avg-ci, avg, avg+ci]

# INPUT/OUTPUT


def write_tocsv(fname, data, delim=',', header=None):
    '''
    Saves to file list of list structure of numbers
    '''

    with open(fname, 'w') as fd:
        wr = csv.writer(fd, delimiter=delim)
        if header is not None:
            #             print('writing header')
            wr.writerow(i for i in header)
#         print ("writing data")
        for row in data:
            wr.writerow(row)

    fd.close()
    print(f'{fname} saved.')

def read_fromcsv(fname, delim=','):
    with open(fname, 'r') as fd:
        rd = csv.reader(fd,delimiter=delim)
        table = []
        for row in rd:
            table.append(row)
        
    return table

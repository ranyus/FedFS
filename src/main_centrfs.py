
import os

import numpy as np
from numpy.lib.npyio import load
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer, fetch_covtype

import crossentropy as ce
import custom_utils as cut



def main():
    
    # BREAST CANCER, COVTYPE
    X_all,y_all=load_breast_cancer(return_X_y=True)
    # X_all, y_all = fetch_covtype(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.20, random_state=123)


    num_classes = np.unique(y_train).shape[0]
    print('num classes {}'.format(num_classes))
    
    # configuration
    ce_sel_prob = 0.99
    ce_max_samples = 20

    # -1 FOR |\gamma_t - \gamma_{t-d}|>\ce_tol
    # > 0 for number of predefined steps
    ce_max_steps = 150
    ce_tol = 1e-7
    # smoothing paramenter for ce update. Semantics: (1-alpha) * old + (alpha)* new.
    # starting point
    alpha_smooth = 1.
    # decreasing step
    alpha_step = (alpha_smooth)/ce_max_steps
    # alpha_step = 0/ce_max_steps
    estop= False
    # parallel jobs
    njobs = 1
    
    
    
    
    log_path = '../stats/centrfs-bc/'
    CHECK_DIR = os.path.exists(log_path)
    if not CHECK_DIR:        
        os.mkdir(log_path)
        print('Created log dir: {}'.format(log_path))
    else:
        print('{} already exists'.format(CHECK_DIR))

    print('Running Crossentropy with alpha {}'.format(alpha_smooth))
    fs, prob, _, _, conv_round, _ = ce.get_features_ce(X_train, y_train, classes=num_classes, selection_probability=ce_sel_prob, max_sample_ce=ce_max_samples,max_steps_ce=ce_max_steps, alpha=alpha_smooth, epsilon_tolerance=ce_tol, alpha_step=alpha_step, verbose=True, early_stop=estop, njobs=njobs, subsample_perc=0.20)


    print('Saving logs...')
    sfx = f'-alpha({alpha_smooth}).csv'

    cut.write_tocsv(log_path+'global_fs'+sfx, map(lambda x: [x], fs))
    cut.write_tocsv(log_path+'global_prob'+sfx, map(lambda x: [x], prob))
    cut.write_tocsv(log_path+'global_running_stats'+sfx, [[conv_round, ce_tol, alpha_smooth, ce_max_samples, ce_max_steps]], 
                    header=['Convergence Round', 'CE Tolerance', 'Alpha', 'CE max samples', 'CE max steps'])


if __name__ == "__main__":
    main()

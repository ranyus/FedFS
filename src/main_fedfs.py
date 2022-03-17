
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_covtype, load_breast_cancer

import custom_utils as cut
import federated_feature_selection as ffs


def main():

    # +++ BREAST CANCER, COVTYPE
    # X_all,y_all=load_breast_cancer(return_X_y=True)
    # # X_all, y_all = fetch_covtype(return_X_y=True)
    # X_train, _, y_train, _ = train_test_split(
    #     X_all, y_all, test_size=0.20, random_state=123)
    # num_classes = np.unique(y_train).shape[0]
    # +++

    # +++ VOXEL (MAV) DATASET PREPARATION
    # X_all, y_all = dut.load_voxel(return_Xy=True)

    # # split train test
    # Xtrain, Xtest, ytrain, ytest = train_test_split(
    #     X_all, y_all, test_size=0.20)

    # # restore time
    # y_train = np.array(ytrain.sort_values(by='timestamps').iloc[:, 1])
    # y_test = np.array(ytest.sort_values(by='timestamps').iloc[:, 1])
    # X_train = np.array(Xtrain.sort_values(by='timestamps').iloc[:, 1:])
    # X_test = np.array(Xtest.sort_values(by='timestamps').iloc[:, 1:])
    # num_classes = np.unique(y_train).shape[0]
    # +++

    # Distribute dataset on the workers. IID data partitioning [FOR ALL THE ABOVE DATASETS]
    num_workers = 10
    n_train = X_train.shape[0]
    nums = [n_train // num_workers] * num_workers
    # nums[-1] += n_train % num_workers
    # Create <num_workers> edge nodes
    idxs = np.array([np.random.choice(np.arange(n_train),
                                       num, replace=False) for num in nums])

    print("{} workers holding {} patterns, respectively".format(
        idxs.shape[0], idxs.shape[1]))

    # Experimental settings Settings

    perc_nodes = [1.]
    perc_data = [1.]

    # stopping condition: -1 goas through all the comm. rounds. 0< x < 1 is the tolerance
    tolerance = -1
    
    ce_max_samples = 20
    # -1 FOR |\gamma_t - \gamma_{t-d}|>\ce_tol
    # > 0 for number of predefined steps
    ce_max_steps = 1
    ce_tol = 1e-7
    selection_prob = 0.99
    # smoothing paramenter for ce update. Semantics: alpha* old + (1-alpha)* new.  -1 for 1/(max_sample_ce*iteration). Default 1e-1. 
    alpha_smooth = 1.0
    MAX_ROUND = 300
    alpha_step = (alpha_smooth)/(MAX_ROUND*ce_max_steps)

    # NN architecture
    njobs = 1

    # log_path = '../stats/fedfs-bc/'
    log_path = '../stats/fedfs-mav/'
    if not os.path.isdir(log_path):
        print('Creating log dir: {}'.format(log_path))
        os.mkdir(log_path)
    print('Running Federated Crossentropy with alpha {}'.format(alpha_smooth))
    for frac_nodes in perc_nodes:

        for frac_data in perc_data:
            print(
                '*** New Configuration: [nodes {}, data {}]'.format(frac_nodes, frac_data))
            fs, net, prob, conv_round, fed_running_fs, fed_running_prob, dwl_data_before_conv, max_rounds, alpha_per_round = ffs.federated_feature_selection(X_train, y_train, idxs,fraction_selected_workers=frac_nodes,fraction_local_subsample=frac_data,early_stop_tolerance=tolerance, ce_selection_prob=selection_prob,ce_tolerance=ce_tol, alpha=alpha_smooth, alpha_step=alpha_step,max_sample_ce=ce_max_samples, max_step_ce=ce_max_steps, fed_max_round=MAX_ROUND,verbose=True, verbose_scenario=False,njobs=njobs)
            
            print('Saving logs...')
            sfx = '-nodes({})_data({})-alpha_({}).csv'.format(frac_nodes, frac_data, alpha_smooth)
            cut.write_tocsv(log_path+'fed_fs'+sfx, map(lambda x: [x], fs))
            cut.write_tocsv(log_path+'fed_prob'+sfx, map(lambda x: [x], prob))
            cut.write_tocsv(log_path+'fed_dwl'+sfx,
                            map(lambda x: [x], dwl_data_before_conv))
            cut.write_tocsv(log_path+'fed_running_prob'+sfx, fed_running_prob)
            cut.write_tocsv(log_path+'fed_running_fs'+sfx, fed_running_fs)
    
            cut.write_tocsv(log_path+'fed_running_alpha' +
                            sfx, map(lambda x: [x], alpha_per_round))
            cut.write_tocsv(log_path+'fed_running_stats'+sfx, [[conv_round, net, max_rounds, tolerance, ce_max_samples, ce_max_steps, int(num_workers*frac_nodes)]],
                            header=['Convergence Round', 'Net. Overhead', 'Max Rounds', 'Tolerance', 'CE max samples', 'CE max steps', 'Active workers'])
            


if __name__ == "__main__":
    main()

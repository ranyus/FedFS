from os import replace, times
from posix import times_result
import numpy as np
import pymit as mit
from scipy import stats
from joblib import Parallel, delayed
# import joblib.pool as jbp 

import arrayoperations as ao


def get_features_ce(X, Y, init_probability=[0.5], classes=2, selection_percentile=25, selection_probability=0.90, max_sample_ce=10, max_steps_ce=1, alpha=1, epsilon_tolerance=1e-5, current_step=1, subsample_perc=-1, alpha_step=0, verbose=False, early_stop=False,njobs=2):


    # np.random.seed(times_result.time())

    N,num_features = X.shape

    # set the initial probability
    if len(init_probability) == 1:
        prob_dist_features = np.ones(num_features)*init_probability[0]
    else:
        prob_dist_features = init_probability
    
    # auxiliary structures
    selected_features_samples = []

    old_prob = np.zeros_like(prob_dist_features)
    kstest_curr = 0
    kstest_prev = 0
    is_converged = False
    obj_fun = np.zeros((max_sample_ce, 2))


    for num_steps in np.arange(max_steps_ce):
        if verbose:
            print('Step: {}'.format(num_steps))
            print('Alpha: {}'.format(alpha))
            print('Num Feats: {}'.format(len(prob_dist_features[prob_dist_features>selection_probability])))
            print('Prob Feats: {}'.format(prob_dist_features))
        
        features_samples = (np.random.binomial(n=1, p=prob_dist_features, size=[
                            max_sample_ce, num_features])).astype(int)
        if subsample_perc > 0:
            ssize = np.int(X.shape[0]*subsample_perc)
            data_idx = np.random.choice(X.shape[0],size=ssize,replace=True)
            XX = X[data_idx,:]
            YY = Y[data_idx]
        else:
            XX=X
            YY=Y
        if njobs > 1:
            Parallel(n_jobs=njobs, verbose=0)(delayed(
                parallel_obj_computation)(
                    obj_fun[i, :], XX[:, features_samples[i, :] > 0], YY, features_samples[i,:]>0, classes)
                for i in np.arange(features_samples.shape[0]))
        else: 
            for i in np.arange(features_samples.shape[0]):            
                fst = features_samples[i, :] > 0
                if np.any(fst):
                        
                    outvec, maxVal = ao.merge_multiple_arrays(
                        XX[:, fst])
                    obj_fun[i, :] = [i, mit.H_cond(YY, outvec, bins=[classes, maxVal])]
        np.sum(obj_fun[:,0]) # barrier
        # compute percentile
        perc = np.percentile(obj_fun[:, 1], selection_percentile)
        # select samples whose obj_fub is below percentile
        selected_features_samples = features_samples[obj_fun[obj_fun[:, 1] <= perc, 0].astype(
            int), :]
        
        np.copyto(old_prob,prob_dist_features)
        # update the probability distribution of features 
        
        new_p = np.sum(selected_features_samples, axis=0)/selected_features_samples.shape[0]
        
        
        if np.all(np.isreal(new_p)) and np.all(new_p >= 0) and np.all(new_p <= 1):
            prob_dist_features = (1-alpha) * prob_dist_features + alpha*new_p
            np.clip(prob_dist_features,1e-10,1-1e-10,out=prob_dist_features)
        
            
        
        # evaluate stopping condition
        if early_stop:
            kstest_prev = kstest_curr
            kstest_curr = stats.ks_2samp(old_prob, prob_dist_features)[1]
            var = np.abs(kstest_curr-kstest_prev)
            if var <= epsilon_tolerance and kstest_curr > .995 and not is_converged:
                print("Convergence at {}!!!".format(num_steps))
                print('Var = {} ; KS p-value = {}'.format(var, kstest_curr))
                convergence_round = num_steps 
                is_converged = True

                break

        alpha -= alpha_step
    selected_feats = np.where(prob_dist_features >= selection_probability)[0]

#     print ("Found {} features in {} seconds ".format(selected_feats.shape[0],elapsed))
#     print("CE stopped after {} steps with diff: {}".format(num_steps,obj_diff))
    return selected_feats, prob_dist_features, np.sum(selected_features_samples, axis=0), selected_features_samples.shape[0], num_steps, alpha


def parallel_obj_computation(obj_fun, X, Y, features_samples,classes,i):
    # print(obj_fun, id(obj_fun), 'arrays in workers')
    fst = features_samples[i, :] > 0
    if np.sum(fst)>0:
        outvec, maxVal = ao.merge_multiple_arrays(
            X[:, fst])
        if (Y.ndim != 1 or outvec.ndim != 1):
            print('Y.shape:{}\toutvec.shape{}'.format(Y.shape,outvec.shape))

        obj_fun[i, :] = [i, mit.H_cond(
            Y, outvec, bins=[classes, maxVal])]


def parallel_obj_computation(obj_fun, X, Y, features_samples, classes):
    # print(obj_fun, id(obj_fun), 'arrays in workers')
    if np.sum(features_samples) > 0:
        outvec, maxVal = ao.merge_multiple_arrays(
            X)
        if (Y.ndim != 1 or outvec.ndim != 1):
            print('Y.shape:{}\toutvec.shape{}'.format(Y.shape, outvec.shape))

        obj_fun = mit.H_cond(
            Y, outvec, bins=[classes, maxVal])

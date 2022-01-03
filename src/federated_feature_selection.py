
import math
import numpy as np

from scipy import stats
import crossentropy as ce


def federated_feature_selection(patterns: np.array, labels: np.array, data_parition_idxs: list,
                                fraction_selected_workers: float = 1.0, fraction_local_subsample=1.0,
                                early_stop_tolerance=1e-1, max_sample_ce=10, max_step_ce=1, ce_selection_prob=0.9, alpha=1, alpha_step=0, ce_tolerance=-1, fed_max_round=-1,
                                verbose=False, verbose_scenario=False, njobs=2):
    '''
    This function executes the federated feature selection exchanging a probability vector of dimension d per device. 
    Each element of the probability vector is the probability of the i-th feature. 
    '''
    # Init
    X_train = patterns
    y_train = labels

    N, d = X_train.shape
    num_classes = np.unique(y_train).shape[0]

    # numero di data partitions
    idxs = data_parition_idxs
    # size di ogni data partition
    nums = [len(v) for v in data_parition_idxs]
    # numero di workers (= idxs)
    num_workers = len(data_parition_idxs)
    # number of workers to use at each round. Max usable workers is 'num_workers'
    subset_size = int(
        np.fmin(np.fmax(1, fraction_selected_workers*num_workers), num_workers))

    # auxiliary structures
    gdist = np.zeros_like(d, dtype=np.float)
    init_prob = np.ones(d)*0.5
    old_prob = np.zeros_like(init_prob)
    local_prob_workers = np.ones((num_workers, d))*0.5

    bitmap_size = (d//32)+1
    print("bitmap size: {} byte".format(bitmap_size))
    # (local dataset size) / (number of nodes * total size of messages ul + dl)
    if fed_max_round == -1:
        max_rounds = int((N*d)/(subset_size*(2*d+1+2*bitmap_size)))
    else:
        max_rounds = fed_max_round

    # structures for statistics
    alpha_per_round = []
    features_per_round = []
    features_prob_per_round = []
    downloaded_data = []

    print("Using {} workers. Max rounds {}".format(subset_size, max_rounds))

    net_cost = 0

    # selezione i worker da usare
    selected_workers = np.random.choice(
        np.arange(num_workers), size=(max_rounds, subset_size), replace=True)

    convergence_round = max_rounds
    is_converged = False

    kstest_curr = 0
    kstest_prev = 0
    # loop for global sync
    ce_steps = 1
    for comround in np.arange(max_rounds):
        if verbose:
            print('Round {}\nFS distribution: {}'.format(comround, init_prob))
        if verbose_scenario:
            print('Round {}'.format(comround))

        # LOCAL COMPUTATION
        local_data_size = []
        alpha_per_round.append(alpha)
        # loop for local data collection and local computation
        i = 0
        for worker in selected_workers[comround]:
            # initial update of local worker with global information
            np.copyto(local_prob_workers[worker], init_prob)

            # local subsample with replacement to simulate data collection through time.

            subsample_size = math.floor(
                len(idxs[worker])*fraction_local_subsample)

            # collect data
            local_idxs = np.random.choice(
                idxs[worker], size=subsample_size, replace=True)

            # get data for CE
            X_loc = np.array(X_train[local_idxs, :])
            y_loc = np.array(y_train[local_idxs])

            # perform CE
            _, fsdist, _, _, ce_steps, cur_alpha = ce.get_features_ce(X_loc, y_loc, 
                classes=num_classes, init_probability=local_prob_workers[worker],
                 max_sample_ce=max_sample_ce, max_steps_ce=max_step_ce, alpha=alpha, 
                 epsilon_tolerance=ce_tolerance, current_step=ce_steps, alpha_step=alpha_step, njobs=njobs)

            # save current probability vector and dataset size
            np.copyto(local_prob_workers[worker], fsdist)
            local_data_size.append(len(local_idxs))
            # stats: i) network overhead stats, ii)ce steps
            net_cost += np.count_nonzero(fsdist)+1+bitmap_size

        # update alpha for all
        alpha = cur_alpha
        # CENTRAL ENTITY COMPUTATION (Implementation of Eq 14)
        global_denominator = np.sum(local_data_size)
        i = 0
        for worker in selected_workers[comround]:
            gdist = gdist + \
                local_prob_workers[worker] * \
                (local_data_size[i]/global_denominator)
            i += 1

        # STATS
        # collecting net cost
        net_cost += subset_size * (np.count_nonzero(init_prob)+bitmap_size)
        #
#         global_ce_steps.append(local_ce_steps)
        downloaded_data.append(np.sum(local_data_size))

        features_per_round.append(np.where(init_prob >= ce_selection_prob)[0])
        features_prob_per_round.append(fsdist)

        # EXIT condition management
        np.copyto(old_prob, init_prob)
        # np.copyto(init_prob,np.clip(alpha_smooth*init_prob+(1-alpha_smooth)*gdist,0.01,.99))
        np.copyto(init_prob, gdist)

        kstest_prev = kstest_curr
        kstest_curr = stats.ks_2samp(old_prob, init_prob)[1]
        var = np.abs(kstest_curr-kstest_prev)

        if verbose:
            print('Current alpha {} | {}'.format(alpha, cur_alpha))
            print(" diff: {}".format(var))
            print('Current num features {}'.format(
                (features_per_round[-1].size)))
            print('KS p-value = {}'.format(kstest_curr))

        if var <= early_stop_tolerance and kstest_curr > .995 and not is_converged:
            print("Convergence at {}!!!".format(comround))
            print('Var = {} ; KS p-value = {}'.format(var, kstest_curr))
            convergence_round = comround
            is_converged = True

            break

        # RESET before next round
        gdist.fill(0)
        glen = 0

    # RETURN select final features
    fed_final_fs = np.where(init_prob >= ce_selection_prob)[0]
    return fed_final_fs, net_cost, init_prob, convergence_round, features_per_round, features_prob_per_round, downloaded_data, max_rounds, alpha_per_round

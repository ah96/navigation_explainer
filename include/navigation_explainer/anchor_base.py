"""Base anchor functions"""
from __future__ import print_function
import numpy as np
import operator
import copy
import sklearn
import collections


def matrix_subset(matrix, n_samples):
    if matrix.shape[0] == 0:
        return matrix
    n_samples = min(matrix.shape[0], n_samples)
    return matrix[np.random.choice(matrix.shape[0], n_samples, replace=False)]


class AnchorBaseBeam(object):
    def __init__(self):
        pass

    @staticmethod
    def kl_bernoulli(p, q):
        p = min(0.9999999999999999, max(0.0000001, p))
        q = min(0.9999999999999999, max(0.0000001, q))
        return (p * np.log(float(p) / q) + (1 - p) *
                np.log(float(1 - p) / (1 - q)))

    @staticmethod
    def dup_bernoulli(p, level):
        lm = p
        um = min(min(1, p + np.sqrt(level / 2.)), 1)
        qm = (um + lm) / 2.
#         print 'lm', lm, 'qm', qm, kl_bernoulli(p, qm)
        if AnchorBaseBeam.kl_bernoulli(p, qm) > level:
            um = qm
        else:
            lm = qm
        return um

    @staticmethod
    def dlow_bernoulli(p, level):
        um = p
        lm = max(min(1, p - np.sqrt(level / 2.)), 0)
        qm = (um + lm) / 2.
#         print 'lm', lm, 'qm', qm, kl_bernoulli(p, qm)
        if AnchorBaseBeam.kl_bernoulli(p, qm) > level:
            lm = qm
        else:
            um = qm
        return lm

    @staticmethod
    def compute_beta(n_features, t, delta):
        alpha = 1.1
        k = 405.5
        temp = np.log(k * n_features * (t ** alpha) / delta)
        return temp + np.log(temp)

    @staticmethod
    def lucb(sample_fns, initial_stats, epsilon, delta, batch_size, top_n,
             verbose=False, verbose_every=1):
        # initial_stats must have n_samples, positive
        n_features = len(sample_fns)
        n_samples = np.array(initial_stats['n_samples'])
        positives = np.array(initial_stats['positives'])
        ub = np.zeros(n_samples.shape)
        lb = np.zeros(n_samples.shape)
        #print('\nlucb')
        #print('n_features = ', n_features)
        #print('n_samples = ', n_samples)
        #print('positives = ', positives)
        #print('top_n = ', top_n)
        for f in np.where(n_samples == 0)[0]:
            n_samples[f] += 1
            positives[f] += sample_fns[f](1)
        #print('n_samples = ', n_samples)
        #print('positives = ', positives)    
        
        # ja zakomentarisao: da ukljuci sve features sa istim precision
        #if n_features == top_n:
        #    return range(n_features)
        # ja zakomentarisao
        
        means = positives / n_samples
        #print('means = ', means)
        t = 1

        def update_bounds(t, top_n):
            #print('\nupdate_bounds')
            # dodatak: da ukljuci sve features sa istim precision
            if top_n > 1:
                sorted_means = np.sort(means)
                #print('sorted_means = ', sorted_means)
                top_n = np.count_nonzero(sorted_means == sorted_means[-1])
                print('top_n = ', top_n)
            sorted_means = np.argsort(means)
            #print('sorted_means = ', sorted_means)
            beta = AnchorBaseBeam.compute_beta(n_features, t, delta)
            #print('beta = ', beta)
            J = sorted_means[-top_n:]
            #print('J = ', J)
            not_J = sorted_means[:-top_n]
            #print('not_J = ', not_J)
            for f in not_J:
                ub[f] = AnchorBaseBeam.dup_bernoulli(means[f], beta /
                                                     n_samples[f])
            for f in J:
                lb[f] = AnchorBaseBeam.dlow_bernoulli(means[f],
                                                      beta / n_samples[f])
            #print('ub = ', ub)
            #print('lb = ', lb)
            #print('\nub[not_J] = ', ub[not_J])
            #print('np.argmax(ub[not_J]) = ', np.argmax(ub[not_J]))
            #print('\nlb[J] = ', lb[J])
            #print('np.argmin(lb[J]) = ', np.argmin(lb[J]))
            ut = not_J[np.argmax(ub[not_J])]
            lt = J[np.argmin(lb[J])]
            return ut, lt, top_n
        ut, lt, top_n = update_bounds(t, top_n)
        #print('(ut, lt) = ', (ut, lt))
        #print('ub = ', ub)
        #print('lb = ', lb)
        B = ub[ut] - lb[lt]
        verbose_count = 0
        epsilon = 0.8
        print('B = ', B)
        while B > epsilon:
            verbose_count += 1
            #print('\nBB = ', B)
            #print('\nepsilon = ', epsilon)
            if verbose and verbose_count % verbose_every == 0:
                print('Best: %d (mean:%.10f, n: %d, lb:%.4f)' %
                      (lt, means[lt], n_samples[lt], lb[lt]), end=' ')
                print('Worst: %d (mean:%.4f, n: %d, ub:%.4f)' %
                      (ut, means[ut], n_samples[ut], ub[ut]), end=' ')
                print('B = %.2f' % B)
            n_samples[ut] += batch_size
            positives[ut] += sample_fns[ut](batch_size)
            means[ut] = positives[ut] / n_samples[ut]
            n_samples[lt] += batch_size
            positives[lt] += sample_fns[lt](batch_size)
            means[lt] = positives[lt] / n_samples[lt]
            t += 1
            ut, lt, top_n = update_bounds(t, top_n)
            B = ub[ut] - lb[lt]
        sorted_means = np.argsort(means)
        #print('\nsorted_means = ', sorted_means)
        return sorted_means[-top_n:]

    @staticmethod
    def make_tuples(previous_best, state):
        # alters state, computes support for new tuples
        normalize_tuple = lambda x: tuple(sorted(set(x)))  # noqa #funkcija
        all_features = range(state['n_features'])
        coverage_data = state['coverage_data']
        current_idx = state['current_idx']
        data = state['data'][:current_idx]
        labels = state['labels'][:current_idx]
        #print('\nmake_tuples')
        #print('\nall_features = ', all_features)
        #print('\ncoverage_data.shape = ', coverage_data.shape)
        #print('\ncurrent_idx = ', current_idx)
        #print('\ndata.shape = ', data.shape)
        #print('\nlabels.shape = ', labels.shape)
        if len(previous_best) == 0:
            #print('\nmake_tuples')
            #print('\nUSAO!!!')
            tuples = [(x, ) for x in all_features]
            #print('\ntuples = ', tuples)
            for x in tuples:
                #print('\nx = ', x)
                pres = data[:, x[0]].nonzero()[0]
                #print('data[:, x[0]].nonzero() = ', data[:, x[0]].nonzero())
                #print('pres = ', pres)
                #print('set(pres) = ', set(pres))
                #print('float(len(pres)) = ', float(len(pres)))
                #print('float(labels[pres].sum()) = ', float(labels[pres].sum()))
                # NEW
                state['t_idx'][x] = set(pres)
                state['t_nsamples'][x] = float(len(pres))
                state['t_positives'][x] = float(labels[pres].sum())
                state['t_order'][x].append(x[0])
                # NEW
                state['t_coverage_idx'][x] = set(
                    coverage_data[:, x[0]].nonzero()[0])
                state['t_coverage'][x] = (
                    float(len(state['t_coverage_idx'][x])) /
                    coverage_data.shape[0])
                #print('state[t_coverage_idx][x] = ', state['t_coverage_idx'][x])
                #print('state[t_coverage][x] = ', state['t_coverage'][x])    
            #print('\nmake_tuples')        
            return tuples
        new_tuples = set()
        for f in all_features:
            for t in previous_best:
                new_t = normalize_tuple(t + (f, ))
                if len(new_t) != len(t) + 1:
                    continue
                if new_t not in new_tuples:
                    new_tuples.add(new_t)
                    state['t_order'][new_t] = copy.deepcopy(state['t_order'][t])
                    state['t_order'][new_t].append(f)
                    state['t_coverage_idx'][new_t] = (
                        state['t_coverage_idx'][t].intersection(
                            state['t_coverage_idx'][(f,)]))
                    state['t_coverage'][new_t] = (
                        float(len(state['t_coverage_idx'][new_t])) /
                        coverage_data.shape[0])
                    t_idx = np.array(list(state['t_idx'][t]))
                    t_data = state['data'][t_idx]
                    present = np.where(t_data[:, f] == 1)[0]
                    state['t_idx'][new_t] = set(t_idx[present])
                    idx_list = list(state['t_idx'][new_t])
                    state['t_nsamples'][new_t] = float(len(idx_list))
                    state['t_positives'][new_t] = np.sum(
                        state['labels'][idx_list])
        return list(new_tuples)

    @staticmethod
    def get_sample_fns(sample_fn, tuples, state):
        # each sample fn returns number of positives
        sample_fns = []
        def complete_sample_fn(t, n):
            #print('\n(t, n) = ', (t, n))
            raw_data, data, labels = sample_fn(list(t), n)
            #print('raw_data.shape = ', raw_data.shape)
            #print('data.shape = ', data.shape)
            #print('labels.shape = ', labels.shape)
            current_idx = state['current_idx']
            # idxs = range(state['data'].shape[0], state['data'].shape[0] + n)
            idxs = range(current_idx, current_idx + n)

            if '<U' in str(raw_data.dtype):
                # String types: make sure both string types are of maximum length 
                # to avoid string truncation. E.g., '<U308', '<U290' -> '<U308'
                max_dtype = max(str(state['raw_data'].dtype), str(raw_data.dtype))
                state['raw_data'] = state['raw_data'].astype(max_dtype)
                raw_data = raw_data.astype(max_dtype)

            state['t_idx'][t].update(idxs)
            state['t_nsamples'][t] += n
            state['t_positives'][t] += labels.sum()
            state['data'][idxs] = data
            state['raw_data'][idxs] = raw_data
            #print('\nlabels.shape = ', labels.shape)
            labels = labels.reshape((labels.shape[0],)) # added by Amar
            #print('\nlabels.shape = ', labels.shape)
            state['labels'][idxs] = labels
            state['current_idx'] += n
            if state['current_idx'] >= state['data'].shape[0] - max(1000, n):
                prealloc_size = state['prealloc_size']
                current_idx = data.shape[0]
                state['data'] = np.vstack(
                    (state['data'],
                     np.zeros((prealloc_size, data.shape[1]), data.dtype)))
                state['raw_data'] = np.vstack(
                    (state['raw_data'],
                     np.zeros((prealloc_size, raw_data.shape[1]),
                              raw_data.dtype)))
                state['labels'] = np.hstack(
                    (state['labels'],
                     np.zeros(prealloc_size, labels.dtype)))
            # This can be really slow
            # state['data'] = np.vstack((state['data'], data))
            # state['raw_data'] = np.vstack((state['raw_data'], raw_data))
            # state['labels'] = np.hstack((state['labels'], labels))
            return labels.sum()
        for t in tuples:
            sample_fns.append(lambda n, t=t: complete_sample_fn(t, n))
        return sample_fns


    @staticmethod
    def get_initial_statistics(tuples, state):
        stats = {
            'n_samples': [],
            'positives': []
        }
        '''
        print('\nget_initial_statistics')
        print('\ntuples = ', tuples)
        print('\nstats = ', stats)
        print('\nstate = ', state)
        print('\nget_initial_statistics')
        '''
        for t in tuples:
            stats['n_samples'].append(state['t_nsamples'][t])
            stats['positives'].append(state['t_positives'][t])
        return stats

    @staticmethod
    def get_anchor_from_tuple(t, state):
        # TODO: This is wrong, some of the intermediate anchors may not exist.
        anchor = {'feature': [], 'mean': [], 'precision': [],
                  'coverage': [], 'examples': [], 'all_precision': 0}
        anchor['num_preds'] = state['data'].shape[0]
        normalize_tuple = lambda x: tuple(sorted(set(x)))  # noqa
        current_t = tuple()
        for f in state['t_order'][t]:
            current_t = normalize_tuple(current_t + (f,))

            mean = (state['t_positives'][current_t] /
                    state['t_nsamples'][current_t])
            anchor['feature'].append(f)
            anchor['mean'].append(mean)
            anchor['precision'].append(mean)
            anchor['coverage'].append(state['t_coverage'][current_t])
            raw_idx = list(state['t_idx'][current_t])
            raw_data = state['raw_data'][raw_idx]
            covered_true = (
                state['raw_data'][raw_idx][state['labels'][raw_idx] == 1])
            covered_false = (
                state['raw_data'][raw_idx][state['labels'][raw_idx] == 0])
            exs = {}
            exs['covered'] = matrix_subset(raw_data, 10)
            exs['covered_true'] = matrix_subset(covered_true, 10)
            exs['covered_false'] = matrix_subset(covered_false, 10)
            exs['uncovered_true'] = np.array([])
            exs['uncovered_false'] = np.array([])
            anchor['examples'].append(exs)
        return anchor

    @staticmethod
    def anchor_beam(sample_fn, delta=0.05, epsilon=0.1, batch_size=10,
                    min_shared_samples=0, desired_confidence=1, beam_size=1,
                    verbose=True, epsilon_stop=0.05, min_samples_start=256,
                    max_anchor_size=None, verbose_every=1,
                    stop_on_first=False, coverage_samples=256):
        anchor = {'feature': [], 'mean': [], 'precision': [],
                  'coverage': [], 'examples': [], 'all_precision': 0}
        _, coverage_data, _ = sample_fn([], coverage_samples, compute_labels=False)
        #print('\ncoverage_data = ', coverage_data)
        #print('\ncoverage_data.shape = ', coverage_data.shape)
        raw_data, data, labels = sample_fn([], max(1, min_samples_start), compute_labels=True) #([], 1)
        #print('\nraw_data = ', raw_data)
        #print('\nraw_data.shape = ', raw_data.shape)
        #print('\ndata = ', data)
        #print('\ndata.shape = ', data.shape)
        #print('\nlabels = ', labels)
        #print('\nlabels.shape = ', labels.shape)
        
        mean = labels.mean() # coverage
        beta = np.log(1. / delta)
        #print('\nmean, beta = ', mean, beta)
        
        lb = AnchorBaseBeam.dlow_bernoulli(mean, beta / data.shape[0])
        #print('\nlb = ', lb)
        #print('desired_confidence = ', desired_confidence)
        #print('desired_confidence - epsilon = ', desired_confidence - epsilon)
        
        while mean > desired_confidence and lb < desired_confidence - epsilon:
            print('USAO_1!!!')
            nraw_data, ndata, nlabels = sample_fn([], batch_size)
            data = np.vstack((data, ndata))
            raw_data = np.vstack((raw_data, nraw_data))
            labels = np.hstack((labels, nlabels))
            mean = labels.mean()
            lb = AnchorBaseBeam.dlow_bernoulli(mean, beta / data.shape[0])
        if lb > desired_confidence:
            print('USAO_2!!!')
            anchor['num_preds'] = data.shape[0]
            anchor['all_precision'] = mean
            return anchor
        
        prealloc_size = batch_size * 10000
        current_idx = data.shape[0]
        data = np.vstack((data, np.zeros((prealloc_size, data.shape[1]),
                                         data.dtype)))
        raw_data = np.vstack((raw_data, np.zeros((prealloc_size, raw_data.shape[1]), raw_data.dtype)))
        labels = np.asarray([x.item() for x in labels])
        labels = np.hstack((labels, np.zeros(prealloc_size, labels.dtype)))
        n_features = data.shape[1]
        #print('\nprealloc_size, current_idx, data, raw_data, labels, n_features = ', prealloc_size, current_idx, data, raw_data, labels, n_features)
        
        state = {'t_idx': collections.defaultdict(lambda: set()),
                 't_nsamples': collections.defaultdict(lambda: 0.),
                 't_positives': collections.defaultdict(lambda: 0.),
                 'data': data,
                 'prealloc_size': prealloc_size,
                 'raw_data': raw_data,
                 'labels': labels,
                 'current_idx': current_idx,
                 'n_features': n_features,
                 't_coverage_idx': collections.defaultdict(lambda: set()),
                 't_coverage': collections.defaultdict(lambda: 0.),
                 'coverage_data': coverage_data,
                 't_order': collections.defaultdict(lambda: list())
                 }
        #print('\nstate = ', state)         
        
        current_size = 1 #1
        best_of_size = {0: []}
        best_coverage = -1
        best_tuple = ()
        t = 1
        
        if max_anchor_size is None:
            max_anchor_size = n_features
        
        while current_size <= max_anchor_size:
            print('\n(current_size, max_anchor_size) = ', (current_size, max_anchor_size))
            tuples = AnchorBaseBeam.make_tuples(best_of_size[current_size - 1], state)
            #print('\ntuples_before = ', tuples)
            tuples = [x for x in tuples
                      if state['t_coverage'][x] > best_coverage]
            #print('\ntuples_after = ', tuples)
            #print('\nlen(tuples_after) = ', len(tuples))
            #print('\nbest_coverage = ', best_coverage)
            if len(tuples) == 0:
                break
            
            sample_fns = AnchorBaseBeam.get_sample_fns(sample_fn, tuples, state)
            #print('\nlen(sample_fns) = ', len(sample_fns))
            
            initial_stats = AnchorBaseBeam.get_initial_statistics(tuples, state)
            print('\ninitial_stats = ', initial_stats)
            # print tuples, beam_size
            
            chosen_tuples = AnchorBaseBeam.lucb(sample_fns, initial_stats, epsilon, delta, batch_size, max(beam_size, len(tuples)),
                verbose=verbose, verbose_every=verbose_every)
            print('\nchosen_tuples = ', chosen_tuples)    
            best_of_size[current_size] = [tuples[x] for x in chosen_tuples]
            #print('\nbest_of_size = ', best_of_size)
            #return 0         
            
            if verbose:
                print('Best of size ', current_size, ':')
            # print state['data'].shape[0]
            #print('\nbest_of_size[current_size] = ', best_of_size[current_size])
            
            stop_this = False
            tuple_counter = 0
            best_tuples = []
            for i, t in zip(chosen_tuples[::-1], best_of_size[current_size][::-1]):
                print('\n(i, t) = ', (i, t))
                # I can choose at most (beam_size - 1) tuples at each step,
                # and there are at most n_feature steps
                beta = np.log(1. / (delta / (1 + (beam_size - 1) * n_features)))
                # beta = np.log(1. / delta)
                # if state['t_nsamples'][t] == 0:
                #     mean = 1
                # else:
                mean = state['t_positives'][t] / state['t_nsamples'][t]
                lb = AnchorBaseBeam.dlow_bernoulli(
                    mean, beta / state['t_nsamples'][t])
                ub = AnchorBaseBeam.dup_bernoulli(
                    mean, beta / state['t_nsamples'][t])
                coverage = state['t_coverage'][t]
                if verbose:
                    print(i, mean, lb, ub)
                if tuple_counter >= 0:    
                    while ((mean >= desired_confidence and
                        lb < desired_confidence - epsilon_stop) or
                        (mean < desired_confidence and
                            ub >= desired_confidence + epsilon_stop)):
                        #print('\nwhile loop')
                        # print mean, lb, state['t_nsamples'][t]
                        sample_fns[i](batch_size)
                        mean = state['t_positives'][t] / state['t_nsamples'][t]
                        lb = AnchorBaseBeam.dlow_bernoulli(
                            mean, beta / state['t_nsamples'][t])
                        ub = AnchorBaseBeam.dup_bernoulli(
                            mean, beta / state['t_nsamples'][t])
                if verbose:
                    print('%s mean = %.2f lb = %.2f ub = %.2f coverage: %.2f n: %d' % (t, mean, lb, ub, coverage, state['t_nsamples'][t]))
                if mean >= desired_confidence and lb > desired_confidence - epsilon_stop:
                    if verbose:
                        print('Found eligible anchor ', t, 'Coverage:',
                              coverage, 'Is best?', coverage > best_coverage)
                    if coverage > best_coverage:
                        # best coverage se azurira samo ako je nadjen tuple sa
                        # dovoljno visokom preciznoscu
                        #print('\nbest_coverage = ', best_coverage)
                        best_coverage = coverage
                        best_tuple = t
                        best_tuples = [best_tuple]
                        if best_coverage == 1 or stop_on_first:
                            stop_this = True
                    elif coverage == best_coverage and tuple_counter > 0:
                        best_tuples.append(t)
                tuple_counter += 1
            if stop_this:
                break
            current_size += 1
        if best_tuple == ():
            # Could not find an anchor, will now choose the highest precision
            # amongst the top K from every round
            if verbose:
                print('Could not find an anchor, now doing best of each size')
            tuples = []
            for i in range(0, current_size):
                tuples.extend(best_of_size[i])
            # tuples = best_of_size[current_size - 1]
            sample_fns = AnchorBaseBeam.get_sample_fns(sample_fn, tuples,
                                                       state)
            initial_stats = AnchorBaseBeam.get_initial_statistics(tuples,
                                                                  state)
            # print tuples, beam_size
            chosen_tuples = AnchorBaseBeam.lucb(
                sample_fns, initial_stats, epsilon, delta, batch_size,
                1, verbose=verbose)
            best_tuple = tuples[chosen_tuples[0]]
        # return best_tuple, state
        return AnchorBaseBeam.get_anchor_from_tuple(best_tuple, state), best_tuples
4
3
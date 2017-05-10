from collections import defaultdict
import codecs
from random import random
from copy import copy
import datetime
from math import exp
import numpy as np
import pickle
from numba import jit
import os
import os.path as path

'''
Count number of translation probabilities that changed by more than p between
iterations
'''
def dictdiff(x, y, p):
    c = 0
    for k in x.keys(): # assumes x and y share same keys
        if abs(x[k] - y[k]) > p:
            c += 1
    return c

'''
Calculate translation probabilities
'''
def train_ibm(sentences_zipped, initial_translation_probs, max_iters, precision, ibm2=False):
    # sentences_zipped is a list of tuples of two aligned sentences
    # we call these e and f locally for english and foreign, but this will actually be run in both directions
    # both sentences should start with a null word

    # we are modeling generating foreign words from english here

    # translation probs t[(e,f)] is p(f|e)
    # if doing ibm model 2, we also have q[(e_idx,f_idx,e_len,f_len)] is p(e_idx|f_idx,e_len,f_len)
    # the idea is that the probability of a foreign word f is the probability of being aligned to
    # the word at e_idx (from q) times the probability of the english word generating f (from t)

    t = initial_translation_probs
    if ibm2:
        q = defaultdict(random)

    for itr in range(max_iters):
        # count dictionaries
        cef = defaultdict(float)
        ce = defaultdict(float)
        cjilm = defaultdict(float)
        cilm = defaultdict(float)

        #calculate counts
        for (es, fs) in sentences_zipped:
            le, lf = len(es), len(fs)
            for (f_idx, f_word) in enumerate(fs):
                if f_idx == 0:
                    continue # skip null word

                # calculate probabilities of f_word coming from each e_word
                if ibm2:
                    probs = [t[(e_word, f_word)] * q[(e_idx, f_idx, le, lf)] for e_idx,e_word in enumerate(es)]
                else:
                    probs = [t[(e_word, f_word)] for e_idx,e_word in enumerate(es)]
                                
                sum_p = sum(probs)
                probs = [p/sum_p for p in probs] # normalize probabilites

                # update counts
                for e_idx, (e_word, p) in enumerate(zip(es, probs)):
                    if p != 0:
                        cef[(e_word, f_word)] += p
                        ce[e_word] += p
                        cjilm[(e_idx, f_idx, le, lf)] += p
                        cilm[(f_idx, le, lf)] += p

        #save translation probabilities of previous iteration
        tprev = t
        t = copy(t)

        #calculate translation probabilities
        for (e, f) in cef.keys():
            t[(e, f)] = cef[(e, f)] / ce[e]
        if ibm2:
            q = copy(q)

            for (e_idx, f_idx, le, lf) in cjilm.keys():
                q[(e_idx, f_idx, le, lf)] = cjilm[(e_idx, f_idx, le, lf)] / cilm[(f_idx, le, lf)]

        #check change between iterations
        print(itr,'number of changed probabilites:',dictdiff(tprev, t, precision))

    #return translation probabilities
    if ibm2:
        return t,q
    else:
        return t

def align_ibm(e_sent, f_sent, t, q=None):
    # creates alignment variables for each word in f_sent
    # the values point to a location in e_sent
    # if q is None, we will use ibm1, otherwise ibm2
    alignments = []
    le, lf = len(e_sent), len(f_sent)
    for f_idx, f_word in enumerate(f_sent):
        if f_idx == 0:
            alignments.append(0) # for null word
            continue

        if q is not None:
            probs = [t[(e_word, f_word)] * q[(e_idx, f_idx, le, lf)] for e_idx,e_word in enumerate(e_sent)]
        else:
            probs = [t[(e_word, f_word)] for e_idx,e_word in enumerate(e_sent)]

        a = max(range(len(probs)), key=lambda x: probs[x]) # argmax probability
        alignments.append(a)
    return alignments

def align2_ibm(e_sent, f_sent, m, q):
    # like align_ibm, but translation probabilities t have been moved to a matrix m indexed by position
    # creates alignment variables for each position in f_sent
    alignments = []
    le, lf = len(e_sent), len(f_sent)
    for f_idx in range(len(f_sent)):
        if f_idx == 0:
            alignments.append(0) # for null word
            continue

        if q is not None:
            probs = [m[e_idx, f_idx] * q[(e_idx, f_idx, le, lf)] for e_idx,e_word in enumerate(e_sent)]
        else:
            probs = [m[e_idx, f_idx] for e_idx,e_word in enumerate(e_sent)]

        a = max(range(len(probs)), key=lambda x: probs[x]) # argmax probability
        alignments.append(a)
    return alignments

def combine_union(alignments1, alignments2):
    return list(set(alignments1).union(set(alignments2)))

def combine_intersect(alignments1, alignments2):
    return [(a,b) for a,b in alignments1 if (a,b) in alignments2]

@jit
def update_probs(real_probs, us, forward_direction, alpha=1):
    # takes transition probabilities real_probs and adjusts them based on the dual variables (us)
    # forward_direction is a boolean used to indicate which of the two directions we are updating
    # the update is slightly different for each direction
    # alpha is a parameter that adjusts how much we like alignments that are close but not exact

    result = np.zeros_like(real_probs)
    for i in range(1,real_probs.shape[0]):
        for j in range(1,real_probs.shape[1]):
            tmp = 0
            for offset in [-1,1]:
                if j + offset >= 1 and j + offset < real_probs.shape[1]:
                    if forward_direction:
                        tmp += max(us[i, j + offset] - alpha, 0)
                    else:
                        tmp += max(-us[i, j + offset] - alpha, 0)
            if forward_direction:
                result[i, j] = real_probs[i, j] * exp(us[i, j] + tmp)
            else:
                result[i, j] = real_probs[i, j] * exp(-us[i, j] + tmp)
    return result

def update_us(us, alignments_ef, alignments_fe, learning_rate):
    # us are updated by adding learning_rate*(difference in alignments)
    # a positive u indicates alignments_fe has an alignment that alignments_ef does not
    # and positive u means the opposite

    # alignments_ef has an alignment for each foreign word to an english one
    # alignments_fi has an alignment for each enlgish word to a foreign one
    # they have both been converted into the format (e_idx, f_idx)

    # calculate (c^b-c^a)
    diff_cs = np.zeros((len(alignments_fe), len(alignments_ef)))
    for a,b in alignments_ef:
        diff_cs[a,b] -= 1
    for a,b in alignments_fe:
        diff_cs[a,b] += 1

    return us + learning_rate * diff_cs, (diff_cs == 0).all() # the second return value indicates convergence

def align_dual(es, fs, t_ef, t_fe, q_ef, q_fe, alpha=1, iterations=250):
    # for the two languages e and f, es and fs and the sentences, t is the word translations, and
    # q is the position translation probabilities

    # align using dual decomposition to encourage agreement

    converged = False

    #initilaize the lagrange multipliers u[e_idx,j_idx]
    us = np.zeros((len(es), len(fs)))

    # real_probs is a matrix that holds the transition probabilities t by word index
    real_probs_ef = np.zeros((len(es), len(fs)))
    real_probs_fe = np.zeros((len(fs), len(es)))
    for ei, ew in enumerate(es):
        for fi, fw in enumerate(fs):
            if ei == 0 and fi == 0:
                continue
            real_probs_ef[ei, fi] = t_ef[(ew, fw)]
            real_probs_fe[fi, ei] = t_fe[(fw, ew)]

    for i in range(iterations):
        learning_rate = 1 / (i + 1)

        translation_probs_ef = update_probs(real_probs_ef, us, True, alpha=alpha)
        translation_probs_fe = update_probs(real_probs_fe, np.transpose(us),
                                    False, alpha=alpha)

        # alignments will be ordered (e_idx, f_idx)
        # alignments_ef has an alignment for each foreign word to a english one
        alignments_ef = [(b,a) for a,b in enumerate(align2_ibm(es, fs, translation_probs_ef, q_ef))]
        # alignments_fe has an alignment for each enlgish word to a foreign one
        alignments_fe = [(a,b) for a,b in enumerate(align2_ibm(fs, es, translation_probs_fe, q_fe))]

        # adjust us
        us, done = update_us(us, alignments_ef, alignments_fe, learning_rate)

        if done:
            converged = True
            break
    intersection = combine_intersect(alignments_ef, alignments_fe)
    union = combine_union(alignments_ef, alignments_fe)
    return intersection, union, converged

def load_sentences(filename, max_n=None):
    sents_with_null = []
    with codecs.open(filename, encoding='utf-8') as f:
        for line in f:
            sent = line.lower().replace('\n', '').split(' ')
            sents_with_null.append(["null_align"] + sent)
            if max_n is not None and len(sents_with_null) >= max_n:
                break

    return sents_with_null

def count_words(sentences):
    return len(set(word for sent in sentences for word in sent))

def train_complete(e_sentences, f_sentences, save_file=None, ibm2=True):
    # models generation of f_sentences from e_sentences
    # expects a null word at beginning of all sentences

    n_f = count_words(f_sentences)
    initial_probs = defaultdict(lambda: 1./n_f)

    sentences = list(zip(f_sentences, f_sentences))

    # t_ef[(e,f)] is p(f|e) for foreign word f and english word e
    t_ef = train_ibm(sentences, initial_probs, max_iters=10, precision=1e-3, ibm2=False)

    if ibm2:
        if save_file is not None: # save the ibm1 model first
            with open(save_file+'.ibm1', 'wb') as fp:
                pickle.dump((dict(t_ef),None,n_f), fp)

        t_ef, q_ef =  train_ibm(sentences, t_ef, max_iters=5, precision=1e-3, ibm2=True)
    else:
        q_ef = None # so we have something to return

    if save_file is not None:
        with open(save_file, 'wb') as fp:
            pickle.dump((dict(t_ef),dict(q_ef) if q_ef is not None else None,n_f), fp)
    
    return t_ef, q_ef

def load_model(filename):
    with open(filename, 'rb') as fp:
        (t, q, n) = pickle.load(fp)
    #return defaultdict(lambda: 1e-5,t), defaultdict(lambda: 1e-5,q if q is not None else [])
    return defaultdict(lambda: 1./n,t), defaultdict(random, q) if q is not None else None

def filter_null_alignments(alignments):
    return [(a,b) for a,b in alignments if a != 0 and b != 0]

def save_alignments(alignments, filename, reverse_direction=False):
    with open(filename, 'w') as output:
        for alignment in alignments:
            for ew, fw in alignment:
                if ew != 0 and fw != 0:
                    if reverse_direction:
                        output.write(str(fw - 1) + '-' + str(ew - 1) + ' ')
                    else:
                        output.write(str(ew - 1) + '-' + str(fw - 1) + ' ')
            output.write('\n')

def eval_alignments(filename, gold_filename):
    os.system('perl measure-alignment-error.pl %s %s'%(gold_filename,filename))


'''
Train aligner on train and dev sets and align on test set
'''
def main(language = 'french'):
    load_model_from_file = True
    max_train = 10000 # for quicker debugging
    ibm2 = True

    print('loading data', datetime.datetime.now())

    if language == 'japanese':
        train_files = {'en':'data/kyoto-train.cln.en', 'f':'data/kyoto-train.cln.ja'}
        dev_files = {'en':'data/english-dev.txt', 'f':'data/japanese-dev.txt'}
        gold_file = 'data/align-dev.txt'
        reverse_output = True
        model_folder = 'models_jap'
    elif language == 'french':
        train_files = {'en':'fr_data/europarl-v6.fr-en.en', 'f':'fr_data/europarl-v6.fr-en.fr'}
        dev_files = {'en':'fr_data/en-fr.en', 'f':'fr_data/en-fr.fr'}
        gold_file = 'fr_data/en-fr.align'
        reverse_output = False
        model_folder = 'models_fr'
    else:
        assert False, 'language not found'

    if load_model_from_file:
        t_ef, q_ef = load_model(path.join(model_folder,'ef_model.pkl.ibm1'))
        t_fe, q_fe = load_model(path.join(model_folder,'fe_model.pkl.ibm1'))
    else:

        #prepare train sentences
        en_sent = load_sentences(train_files['en'], max_train)
        f_sent = load_sentences(train_files['f'], max_train)

        print('training direction 1', datetime.datetime.now())
        t_ef, q_ef = train_complete(en_sent, f_sent, path.join(model_folder,'ef_model.pkl'), ibm2=ibm2)
        print('training direction 2', datetime.datetime.now())
        t_fe, q_fe = train_complete(f_sent, en_sent, path.join(model_folder,'fe_model.pkl'), ibm2=ibm2)


    #prepare test sentences
    en_sent = load_sentences(dev_files['en'])
    f_sent = load_sentences(dev_files['f'])

    dir1_alignments = []
    dir2_alignments = []
    intersection_alignments = []
    union_alignments = []
    num_intersection = 0
    num_union = 0
    for i in range(len(en_sent)):
        # aligns each foreign word to the english one
        align_ef = align_ibm(en_sent[i], f_sent[i], t_ef, q_ef)
        align_ef = filter_null_alignments([(ei,fi) for fi,ei in enumerate(align_ef)])
        # aligns each english word to a foreign one
        align_fe = align_ibm(f_sent[i], en_sent[i], t_fe, q_fe)
        align_fe = filter_null_alignments([(ei,fi) for ei,fi in enumerate(align_fe)])

        dir1_alignments.append(align_ef)
        dir2_alignments.append(align_fe)

        intersection = combine_intersect(align_ef, align_fe)
        union = combine_union(align_ef, align_fe)
        num_intersection += len(intersection)
        num_union += len(union)
        intersection_alignments.append(intersection)
        union_alignments.append(union)
    print('intersection over union', num_intersection / num_union)

    save_alignments(dir1_alignments, 'ibm-align-dir1.txt', reverse_output)
    save_alignments(dir2_alignments, 'ibm-align-dir2.txt', reverse_output)
    save_alignments(intersection_alignments, 'ibm-align-intersection.txt', reverse_output)
    save_alignments(union_alignments, 'ibm-align-union.txt', reverse_output)
    eval_alignments('ibm-align-dir1.txt', gold_file)
    eval_alignments('ibm-align-dir2.txt', gold_file)
    eval_alignments('ibm-align-intersection.txt', gold_file)

    #align test sentences with dual decomp
    alpha = 1
    mas = []
    num_converged = 0
    num_intersection = 0
    num_union = 0
    for i in range(len(en_sent)):
        intersection, union, converged = align_dual(en_sent[i], f_sent[i], t_ef, t_fe, q_ef, q_fe, alpha)
        intersection = filter_null_alignments(intersection)
        union = filter_null_alignments(union)

        num_intersection += len(intersection)
        num_union += len(union)
        mas.append(intersection)
        if converged:
            num_converged += 1
    print(num_intersection / num_union)
    print(float(num_converged) / len(en_sent))

    save_alignments(mas, 'dual-decomp-intersection.txt', reverse_output)
    eval_alignments('dual-decomp-intersection.txt', gold_file)

    #print time at end
    print('finished',datetime.datetime.now())

if __name__ == '__main__':
    main()

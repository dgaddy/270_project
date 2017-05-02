from collections import defaultdict
import codecs
from random import random
from copy import copy
import datetime
from math import exp
import numpy as np
import pickle
from numba import jit

def forward(e, f, I, J, t, q):
    a = [[0] * J for _ in range(I)]
    for i in range(I):
        a[i][0] = q[(i, -1, I)] * t[(e[i], f[0])]
    s = sum([a[i][0] for i in range(I)])
    for i in range(I):
        a[i][0] /= s
    qh = [[0] * I for _ in range(I)]
    for i in range(I):
        for ip in range(I):
            qh[i][ip] = q[(i, ip, I)]
    for j in range(1, J):
        for i in range(I):
            th = t[(e[i], f[j])]
            for ip in range(I):
                a[i][j] += a[ip][j - 1] * qh[i][ip] * th
        s = sum([a[i][j] for i in range(I)])
        for i in range(I):
            a[i][j] /= s
    return a

def backward(e, f, I, J, t, q):
    b = [[0] * J for _ in range(I)]
    for ip in range(I):
        b[ip][J - 1] = 1.
    s = sum([b[ip][J - 1] for ip in range(I)])
    for ip in range(I):
        b[ip][J - 1] /= s
    qh = [[0] * I for _ in range(I)]
    for i in range(I):
        for ip in range(I):
            qh[i][ip] = q[(i, ip, I)]
    for j in range(J - 2, -1, -1):
        for i in range(I):
            th = t[(e[i], f[j + 1])]
            for ip in range(I):
                b[ip][j] += b[i][j + 1] * qh[i][ip] * th
        s = sum([b[ip][j] for ip in range(I)])
        for ip in range(I):
            b[ip][j] /= s
    return b

'''
Count number of translation probabilities that changed by more than p between
iterations
'''
def dictdiff(x, y, p):
    c = 0
    for k in x.keys():
        if abs(x[k] - y[k]) > p:
            c += 1
    return c

'''
Calculate translation probabilities
'''
def train_ibm1(ej, t, m, p):
    c = 0
    while True:
        cef = defaultdict(float)
        ce = defaultdict(float)
        cjilm = defaultdict(float)
        cilm = defaultdict(float)
        dd = defaultdict(float)

        #calculate counts
        for (es, js) in ej:
            le = len(es)
            lj = len(js)
            for (i, f) in enumerate(js):
                dd[i] = 0
                for (j, e) in enumerate(es):
                    dd[i] += t[(e, f)]
            for (i, f) in enumerate(js):
                ddt = dd[i]
                for (j, e) in enumerate(es):
                    if t[(e, f)] != 0:
                        d = t[(e, f)] / ddt
                        cef[(e, f)] += d
                        ce[e] += d
                        cjilm[(i, j, le, lj)] += d
                        cilm[(i, le, lj)] += d

        #save translation probabilities of previous iteration
        tprev = copy(t)

        #calculate translation probabilities
        for (e, j) in cef.keys():
            t[(e, j)] = cef[(e, j)] / ce[e]
        c += 1

        #check if no change between iterations or m iterations reached
        if dictdiff(tprev, t, p) == 0 or c == m:
            break

    #print number of iterations run
    print(c)

    #return translation probabilities
    return t

def train_ibm2(ej, t, q, m, p):
    c = 0
    while True:
        cef = defaultdict(float)
        ce = defaultdict(float)
        cjilm = defaultdict(float)
        cilm = defaultdict(float)
        dd = defaultdict(float)

        #calculate counts
        for (es, js) in ej:
            le = len(es)
            lj = len(js)
            for (i, f) in enumerate(js):
                dd[i] = 0
                for (j, e) in enumerate(es):
                    dd[i] += t[(e, f)] * q[(i, j, le, lj)]
            for (i, f) in enumerate(js):
                ddt = dd[i]
                for (j, e) in enumerate(es):
                    if t[(e, f)] * q[(i, j, le, lj)] != 0:
                        d = t[(e, f)] * q[(i, j, le, lj)] / ddt
                        cef[(e, f)] += d
                        ce[e] += d
                        cjilm[(i, j, le, lj)] += d
                        cilm[(i, le, lj)] += d

        #save translation and transition probabilities of previous iteration
        tprev = copy(t)
        qprev = copy(q)

        #calculate translation and transition probabilities
        for (e, j) in cef.keys():
            t[(e, j)] = cef[(e, j)] / ce[e]
        for (i, j, le, lj) in cjilm.keys():
            q[(i, j, le, lj)] = cjilm[(i, j, le, lj)] / cilm[(i, le, lj)]
        c += 1

        #check if no change between iterations or m iterations reached
        if dictdiff(tprev, t, p) == 0 and dictdiff(qprev, q, p) == 0 or c == m:
            break

    #print number of iterations run
    print(c)

    #return translation probabilities
    return t, q

def train_hmm(ej, t, q, m, p):
    c = 0
    while True:
        cef = defaultdict(float)
        ce = defaultdict(float)
        ci_1 = defaultdict(float)
        cii_l = defaultdict(float)
        for (es, js) in ej:
            le = len(es)
            lj = len(js)
            a = forward(es, js, le, lj, t, q)
            b = backward(es, js, le, lj, t, q)
            for (j, e) in enumerate(es):
                for (i, f) in enumerate(js):
                    if a[j][i] * b[j][i] != 0:
                        d = a[j][i] * b[j][i] / sum([a[k][lj - 1] for k in range(le)])
                        cef[(e, f)] += d
                        ce[e] += d
                        if i == 0:
                            ci_1[(j + 1, le)] += d
            for j in range(le):
                for i in range(1, lj):
                    for jp in range(le):
                        if a[jp][i - 1] * q[(j, jp, le)] * t[(es[j], js[i])] * b[j][i] != 0:
                            d = a[jp][i - 1] * q[(j, jp, le)] * t[(es[j], js[i])] * b[j][i] / sum([a[k][lj - 1] for k in range(le)])
                            cii_l[(j - jp, le)] += d
        tprev = copy(t)
        qprev = copy(q)
        for (e, j) in cef.keys():
            t[(e, j)] = cef[(e, j)] / ce[e]
        for (d, le) in ci_1.keys():
            for j in range(le):
                if j + 1 == d:
                    q[(j, -1, le)] = ci_1[(d, le)] / sum([ci_1[(k + 1, le)] for k in range(le)])
        for (d, le) in cii_l.keys():
            for j in range(le):
                for jp in range(le):
                    if j - jp == d:
                        q[(j, jp, le)] = cii_l[(d, le)] / sum([cii_l[(k - jp, le)] for k in range(le)])
        c += 1
        if dictdiff(tprev, t, p) == 0 and dictdiff(qprev, q, p) == 0 or c == m:
            break
    print(c)
    return (t, q)

def train_hmm_joint(ej, t, q, t2, q2, m, p, pd):
    c = 0
    while True:
        cef = defaultdict(float)
        ce = defaultdict(float)
        cef2 = defaultdict(float)
        ce2 = defaultdict(float)
        cii_l = defaultdict(float)
        cii_l2 = defaultdict(float)
        for (es, js) in ej:
            le = len(es)
            lj = len(js)
            a = forward(es, js, le, lj, t, q)
            b = backward(es, js, le, lj, t, q)
            a2 = forward(js, es, lj, le, t2, q2)
            b2 = backward(js, es, lj, le, t2, q2)
            for (j, e) in enumerate(es):
                for (i, f) in enumerate(js):
                    if a[j][i] * b[j][i] != 0:
                        d = a[j][i] * b[j][i] / sum([a[k][lj - 1] for k in range(le)])
                        cef[(e, f)] += d
                        ce[e] += d
                    if a2[i][j] * b2[i][j] != 0:
                        d = a2[i][j] * b2[i][j] / sum([a2[k][le - 1] for k in range(lj)])
                        cef2[(f, e)] += d
                        ce2[f] += d
            for j in range(le):
                if pd == 'p':
                    if a[j][0] * b[j][0] * a2[0][j] * b2[0][j] != 0:
                        if j == 0:
                            d = a[j][0] * b[j][0] / sum([a[k][lj - 1] for k in range(le)]) * a2[0][j] * b2[0][j] / sum([a2[k][le - 1] for k in range(lj)])
                else:
                    if a[j][0] * b[j][0] + a2[0][j] * b2[0][j] != 0:
                        if j == 0:
                            d = (a[j][0] * b[j][0] / sum([a[k][lj - 1] for k in range(le)]) + a2[0][j] * b2[0][j] / sum([a2[k][le - 1] for k in range(lj)])) / 2
                cii_l[(le, le)] += d
                for i in range(1, lj):
                    for jp in range(le):
                        if pd == 'p':
                            if a[jp][i - 1] * q[(j, jp, le)] * t[(es[j], js[i])] * b[j][i] * a2[i][j] * b2[i][j] != 0:
                                d = a[jp][i - 1] * q[(j, jp, le)] * t[(es[j], js[i])] * b[j][i] / sum([a[k][lj - 1] for k in range(le)]) + a2[i][j] * b2[i][j] / sum([a2[k][le - 1] for k in range(lj)])
                        else:
                            if a[jp][i - 1] * q[(j, jp, le)] * t[(es[j], js[i])] * b[j][i] + a2[i][j] * b2[i][j] != 0:
                                d = (a[jp][i - 1] * q[(j, jp, le)] * t[(es[j], js[i])] * b[j][i] / sum([a[k][lj - 1] for k in range(le)]) + a2[i][j] * b2[i][j] / sum([a2[k][le - 1] for k in range(lj)])) / 2
                        cii_l[(j - jp, le)] += d
            for j in range(lj):
                if pd == 'p':
                    if a2[j][0] * b2[j][0] * a[0][j] * b[0][j] != 0:
                        if j == 0:
                            d = a2[j][0] * b2[j][0] / sum([a2[k][le - 1] for k in range(lj)]) * a[0][j] * b[0][j] / sum([a[k][lj - 1] for k in range(le)])
                else:
                    if a2[j][0] * b2[j][0] + a[0][j] * b[0][j] != 0:
                        if j == 0:
                            d = (a2[j][0] * b2[j][0] / sum([a2[k][le - 1] for k in range(lj)]) + a[0][j] * b[0][j] / sum([a[k][lj - 1] for k in range(le)])) / 2
                cii_l2[(lj, lj)] += d
                for i in range(1, le):
                    for jp in range(lj):
                        if pd == 'p':
                            if a2[jp][i - 1] * q2[(j, jp, lj)] * t2[(js[j], es[i])] * b2[j][i] * a[i][j] * b[i][j] != 0:
                                d = a2[jp][i - 1] * q2[(j, jp, lj)] * t2[(js[j], es[i])] * b2[j][i] / sum([a2[k][le - 1] for k in range(lj)]) * a[i][j] * b[i][j] / sum([a[k][lj - 1] for k in range(le)])
                        else:
                            if a2[jp][i - 1] * q2[(j, jp, lj)] * t2[(js[j], es[i])] * b2[j][i] + a[i][j] * b[i][j] != 0:
                                d = (a2[jp][i - 1] * q2[(j, jp, lj)] * t2[(js[j], es[i])] * b2[j][i] / sum([a2[k][le - 1] for k in range(lj)]) + a[i][j] * b[i][j] / sum([a[k][lj - 1] for k in range(le)])) / 2
                        cii_l2[(j - jp, lj)] += d
        tprev = copy(t)
        qprev = copy(q)
        t2prev = copy(t2)
        q2prev = copy(q2)
        for (e, j) in cef.keys():
            t[(e, j)] = cef[(e, j)] / ce[e]
            t2[(j, e)] = cef2[(j, e)] / ce2[j]
        for (d, le) in cii_l.keys():
            for j in range(le):
                if d == le:
                    q[(j, -1, le)] = cii_l[(d, le)]
                else:
                    for jp in range(le):
                        if j - jp == d:
                            q[(j, jp, le)] = cii_l[(d, le)] / sum([cii_l[(k - jp, le)] for k in range(le)])
        for (d, lj) in cii_l2.keys():
            for j in range(lj):
                if d == lj:
                    q2[(j, -1, lj)] = cii_l2[(d, lj)]
                else:
                    for jp in range(lj):
                        if j - jp == d:
                            q2[(j, jp, lj)] = cii_l2[(d, lj)] / sum([cii_l2[(k - jp, lj)] for k in range(lj)])
        c += 1
        if dictdiff(tprev, t, p) == 0 and dictdiff(qprev, q, p) == 0 and dictdiff(t2prev, t2, p) == 0 and dictdiff(q2prev, q2, p) == 0 or c == m:
            break
    print(c)
    return (t, q, t2, q2)

'''
Align sentence using translation probabilities t
'''
def align(es, js, t, q):
    #ma = defaultdict(int)
    le = len(es)
    lj = len(js)
    ma = [0] * le
    for (j, e) in enumerate(es):
        cma = (0, -1)
        for (i, f) in enumerate(js):
            va = t[(e, f)] * q[(i, j, le, lj)]
            if cma[1] < va:
                cma = (i, va)
        ma[j] = int(cma[0])
    return ma

'''
Align sentence using translation probabilities t
'''
def align2(es, js, m, q):
    # ma = defaultdict(int)
    le = len(es)
    lj = len(js)
    ma = [0] * le
    for j in range(len(es)):
        cma = (0, -1)
        for i in range(len(js)):
            va = m[j, i] * q[(i, j, le, lj)]
            if cma[1] < va:
                cma = (i, va)
        ma[j] = cma[0]
    return ma

@jit
def update_probs(real_probs, us, ej, alpha=1):
    result = np.zeros_like(real_probs)
    for i in range(1,real_probs.shape[0]):
        for j in range(1,real_probs.shape[1]):
            tmp = 0
            for offset in [-1,1]:
                if j + offset >= 1 and j + offset < real_probs.shape[1]:
                    if ej:
                        tmp += max(us[i, j + offset] - alpha, 0)
                    else:
                        tmp += max(-us[i, j + offset] - alpha, 0)
            if ej:
                result[i, j] = real_probs[i, j] * exp(us[i, j] + tmp)
            else:
                result[i, j] = real_probs[i, j] * exp(-us[i, j] + tmp)
    return result

def update_us(us, alignments_ej, alignments_je, learning_rate):
    # calculate (c^b-c^a)
    diff_cs = np.zeros((len(alignments_ej), len(alignments_je)))
    for i, a in enumerate(alignments_ej):
        diff_cs[i, a] -= 1
    for i, a in enumerate(alignments_je):
        diff_cs[a, i] += 1

    return us + learning_rate * diff_cs, (diff_cs == 0).all()

def combine(alignments_ej, alignments_je):
    # alignments = []
    # for i, v in enumerate(alignments_ej):
    #     alignments.append((i, v))
    # for i, v in enumerate(alignments_je):
    #     alignments.append((v, i))
    # return list(set(alignments))
    return list(set(list(enumerate(alignments_ej))
            + [(v, i) for i, v in list(enumerate(alignments_je))]))

def combine_intersect(alignments_ej, alignments_je):
    # tmp = list(enumerate(alignments_ej))
    # alignments = []
    # for i, v in enumerate(alignments_je):
    #     if (v, i) in tmp:
    #         alignments.append((v, i))
    # return alignments
    return [(v, i) for i, v in list(enumerate(alignments_je))
            if (v, i) in list(enumerate(alignments_ej))]

def align_dual(es, js, t_ej, t_je, q_ej, q_je, alpha=1):
    '''
    returns list of tuples (i,j) for aligning english index i to japanese index
    j
    '''

    converged = False

    #a = ej, b = je
    #initilaize us
    us = np.zeros((len(es), len(js)))

    real_probs_ej = np.zeros((len(es), len(js)))
    real_probs_je = np.zeros((len(js), len(es)))
    for m, ew in enumerate(es):
        for n, jw in enumerate(js):
            if m == 0 and n == 0:
                continue
            real_probs_ej[m, n] = t_ej[(ew, jw)]
            real_probs_je[n, m] = t_je[(jw, ew)]

    for i in range(250):
        learning_rate = 1 / (i + 1)

        translation_probs_ej = update_probs(real_probs_ej, us, True, alpha=alpha)
        translation_probs_je = update_probs(real_probs_je, np.transpose(us),
                                    False, alpha=alpha)

        alignments_ej = align2(es, js, translation_probs_ej, q_ej)
        alignments_je = align2(js, es, translation_probs_je, q_je)

        # adjust us

        us, done = update_us(us, alignments_ej, alignments_je, learning_rate)

        if done:
            converged = True
            break
    intersection = combine_intersect(alignments_ej, alignments_je)
    union = combine(alignments_ej, alignments_je)
    return intersection, union, converged
    # return combine(alignments_ej, alignments_je), converged
    # return combine_intersect(alignments_ej, alignments_je), converged

'''
Train aligner on train and dev sets and align on test set
'''
if __name__ == "__main__":
    #print time at start
    print(datetime.datetime.now())

    #prepare train sentences
    e = []
    en = []
    j = []
    jn = []
    # ews = []
    # jws = []
    for line in codecs.open('data/english-dev.txt', encoding='utf-8'):
        # e.append(["null_align"] + line.lower().replace('\n', '').split(' '))
        # ews += line.lower().replace('\n', '').split(' ')
        e.append(line.lower().replace('\n', '').split(' '))
        en.append(["null_align"] + line.lower().replace('\n', '').split(' '))
    # for line in codecs.open('data/kyoto-train.cln.en', encoding='utf-8'):
    # #    e.append(["null_align"] + line.lower().replace('\n', ' ').split(' '))
    #     # ews += line.lower().replace('\n', '').split(' ')
    #     e.append(line.lower().replace('\n', '').split(' '))
    #     en.append(["null_align"] + line.lower().replace('\n', '').split(' '))
    for line in codecs.open('data/japanese-dev.txt', encoding='utf-8'):
        # j.append(["null_align"] + line.replace('\n', '').split(' '))
        # jws += line.replace('\n', '').split(' ')
        j.append(line.replace('\n', '').split(' '))
        jn.append(["null_align"] + line.replace('\n', '').split(' '))
    # for line in codecs.open('data/kyoto-train.cln.ja', encoding='utf-8'):
    # #    j.append(line.replace('\n', ' ').split(' '))
    #     # jws += line.replace('\n', '').split(' ')
    #     j.append(line.replace('\n', '').split(' '))
    #     jn.append(["null_align"] + line.replace('\n', '').split(' '))
    #print(len(ews), len(jws))
    # ej = zip(e, j)
    ej = zip(en, j)
    ej = list(ej)
    # je = zip(j, e)
    je = zip(jn, e)
    je = list(je)

    #calculate initial translation probabilities
    lj = len(set([w for js in j for w in js]))
    f = lambda: 1. / lj
    t_ej = defaultdict(f)
    q_ej = defaultdict(random)
    le = len(set([w for es in e for w in es]))
    g = lambda: 1. / le
    t_je = defaultdict(g)
    q_je = defaultdict(random)

    #set max number of iterations
    m = 2

    #set precision
    p = 0.1

    #train initial aligner with ibm1

    '''
    t_ej = train_ibm1(ej, t_ej, m, p)
    print(datetime.datetime.now())

    #continue training aligner with ibm2
    t_ej, q_ej = train_ibm2(ej, t_ej, q_ej, m, p)
    print(datetime.datetime.now())

    # #continue training aligner with hmm
    # t_ej, q_ej = train_hmm(ej, t_ej, q_ej, m, p)
    # print(datetime.datetime.now())

    #train initial aligner with ibm1
    t_je = train_ibm1(je, t_je, m, p)
    print(datetime.datetime.now())

    #continue training aligner with ibm2
    t_je, q_je = train_ibm2(je, t_je, q_je, m, p)
    print(datetime.datetime.now())
    '''

    '''
    file_pattern = 'pkl_ibm15ibm25/ibm15ibm25%s.pkl'
    items = []
    for item_name in ['t_ej','q_ej','t_je','q_je']:
        with open(file_pattern%item_name,'rb') as f:
            items.append(pickle.load(f))
    t_ej, q_ej, t_je, q_je = items
    '''

    # #continue training aligner with hmm
    # t_je, q_je = train_hmm(je, t_je, q_je, m, p)
    # print(datetime.datetime.now())

    #continue training aligner with hmm joint
    # t_ej, q_ej, t_je, q_je = train_hmm_joint(ej, t_ej, q_ej, t_je, q_je, m, p, True)
    # print(datetime.datetime.now())

    # t_ej = dict(t_ej)
    # q_ej = dict(q_ej)
    #
    # with open('pkl/ibm15hmm5t_ej.pkl', 'wb') as fp:
    #     pickle.dump(t_ej, fp)
    #
    # with open('pkl/ibm15hmm5q_ej.pkl', 'wb') as fp:
    #     pickle.dump(q_ej, fp)
    #
    # t_je = dict(t_je)
    # q_je = dict(q_je)
    #
    # with open('pkl/ibm15hmm5t_je.pkl', 'wb') as fp:
    #     pickle.dump(t_je, fp)
    #
    # with open('pkl/ibm15hmm5q_je.pkl', 'wb') as fp:
    #     pickle.dump(q_je, fp)
    #
    t_ej = pickle.load(open('pkl/ibm15ibm25t_ej.pkl', 'rb'))
    t_ej = defaultdict(f, t_ej)
    print(datetime.datetime.now())
    q_ej = pickle.load(open('pkl/ibm15ibm25q_ej.pkl', 'rb'))
    q_ej = defaultdict(random, q_ej)
    print(datetime.datetime.now())
    t_je = pickle.load(open('pkl/ibm15ibm25t_je.pkl', 'rb'))
    t_je = defaultdict(g, t_je)
    print(datetime.datetime.now())
    q_je = pickle.load(open('pkl/ibm15ibm25q_je.pkl', 'rb'))
    q_je = defaultdict(random, q_je)
    print(datetime.datetime.now())

    #prepare test sentences
    e = []
    en = []
    j = []
    jn = []
    for line in codecs.open('data/english-dev.txt', encoding='utf-8'):
        e.append(line.lower().replace('\n', '').split(' '))
        en.append(["null_align"] + line.lower().replace('\n', '').split(' '))
    for line in codecs.open('data/japanese-dev.txt', encoding='utf-8'):
        j.append(line.replace('\n', '').split(' '))
        jn.append(["null_align"] + line.replace('\n', '').split(' '))

    mas = []
    # num_intersection = 0
    # num_union = 0
    for i in range(len(e)):
        align_eji = align(en[i], j[i], t_ej, q_ej)
        align_jei = align(jn[i], e[i], t_je, q_je)
        # align_eji = align(en[i], jn[i], t_ej, q_ej)
        # align_jei = align(jn[i], en[i], t_je, q_je)
        # intersection = combine_intersect(align_eji, align_jei)
        # union = combine(align_eji, align_jei)
        # num_intersection += len(intersection)
        # num_union += len(union)
        # print(len(intersection) / len(union))
        # mas.append(union)
        mas.append((align_eji, align_jei))
    # print(num_intersection / num_union)

    #save alignments
    output = open('data/align-dev_union_ibm12ibm22.txt', 'w')
    # for ma in mas:
    #     for ew, jw in ma:
    #         if ew != 0 and jw != 0:
    #             output.write(str(jw - 1) + '-' + str(ew - 1) + ' ')
    #     output.write('\n')
    # output.close()


    if False:
        num_intersection = 0
        num_union = 0
        for align_eji, align_jei in mas:
            intersection = 0
            union = 0
            list_union = []
            for key, value in enumerate(align_eji):
                if key != 0:
                    union += 1
                    num_union += 1
                    list_union.append((value, key - 1))
                    if (value + 1, key - 1) in enumerate(align_jei):
                        intersection += 1
                        num_intersection += 1
                    output.write(str(value) + '-' + str(key - 1) + ' ')
            for key, value in enumerate(align_jei):
                if key != 0:
                    if (key - 1, value) not in list_union:
                        union += 1
                        num_union += 1
                    output.write(str(key - 1) + '-' + str(value) + ' ')
            #print(intersection / union)
            output.write('\n')
        output.close()
        print(num_intersection / num_union)

    for alpha in [8, 16, 32, 64]:
        print('alpha:', alpha)
        #align test sentences
        mas = []
        num_converged = 0
        num_intersection = 0
        num_union = 0
        for i in range(len(en)):
            # alignment, converged = align_dual(en[i], jn[i], t_ej, t_je, q_ej, q_je)
            intersection, union, converged = align_dual(en[i], jn[i], t_ej, t_je, q_ej, q_je, alpha)
            intersection = [(ew, jw) for ew, jw in intersection if ew != 0 and jw != 0]
            union = [(ew, jw) for ew, jw in union if ew != 0 and jw != 0]
            #print(len(intersection) / len(union))
            num_intersection += len(intersection)
            num_union += len(union)
            # mas.append(alignment)
            mas.append(union)
            if converged:
                num_converged += 1
        print(num_intersection / num_union)
        print(float(num_converged) / len(en))

        #save alignments
        output = open('output-%s.txt'%alpha, 'w')
        for ma in mas:
            for ew, jw in ma:
                if ew != 0 and jw != 0:
                    output.write(str(jw - 1) + '-' + str(ew - 1) + ' ')
            output.write('\n')
        output.close()

        #print time at end
        print(datetime.datetime.now())

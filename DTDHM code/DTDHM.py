# -*-coding:gb2312-*-
import numpy as np
import pysam
import sys
import pandas as pd
from pyod.models.knn import KNN
import matplotlib.pyplot as plt
from numba import njit
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import subprocess
import time
import os


def read_ref_file(filename, ref):
    # read reference file
    # input: fasta file and ref array.
    # output: ref array
    if os.path.exists(filename):
        print("Read reference file: " + str(filename))
        with open(filename, 'r') as f:
            line = f.readline()
            for line in f:
                linestr = line.strip()
                ref += linestr
    else:
        print("Warning: can not open " + str(filename) + '\n')
    return ref


def get_RCmapq(filename, ReadCount, mapq):
    samfile = pysam.AlignmentFile(filename, "rb")
    for line in samfile:
        if line.reference_name:
            chr = line.reference_name.strip('chr')
            if chr.isdigit() and chr == chrnumb:
                posList = line.positions
                ReadCount[posList] += 1
                mapq[posList] += line.mapq
    return ReadCount, mapq


def ReadDepth(mapq, ReadCount, binNum, ref, binSize):
    # get read depth
    '''
       1. compute the mean of rc in each bin;
       2. count the number of 'N' in ref. If there is a 'N' in a bin£¬the rd is not counted;
       3. GC bias
    '''
    RD = np.full(binNum, 0.0)
    GC = np.full(binNum, 0)
    MQ = np.full(binNum, 0.0)
    pos = np.arange(1, binNum+1)
    for i in range(binNum):
        RD[i] = np.mean(ReadCount[i*binSize:(i+1)*binSize])
        MQ[i] = np.mean(mapq[i*binSize:(i+1)*binSize])
        cur_ref = ref[i*binSize:(i+1)*binSize]
        N_count = cur_ref.count('N') + cur_ref.count('n')
        if N_count == 0:
            gc_count = cur_ref.count('C') + cur_ref.count('c') + cur_ref.count('G') + cur_ref.count('g')
        else:
            RD[i] = -10000
            gc_count = 0
        GC[i] = int(round(gc_count / binSize, 3) * binSize)
    index = RD > 0
    RD = RD[index]
    GC = GC[index]
    MQ = MQ[index]
    pos = pos[index]
    RD = gc_correct(RD, GC)
    return pos, RD, MQ, GC


def gc_correct(RD, GC):
    # correcting gc bias
    bincount = np.bincount(GC)
    global_rd_ave = np.mean(RD)
    for i in range(len(RD)):
        if bincount[GC[i]] < 2:
            continue
        mean = np.mean(RD[GC == GC[i]])
        RD[i] = global_rd_ave * RD[i] / mean
    return RD


def TV_smoothnoise(RD, MQ):
    res = prox_tv1d(alpha, RD)
    RD = res
    res = prox_tv1d(alpha, MQ)
    MQ = res
    # res = prox_tv1d(alpha, GC)
    # GC = res
    return RD, MQ


def prox_tv1d(step_size: float, w: np.ndarray) -> np.ndarray:
    """
    Computes the proximal operator of the 1-dimensional total variation operator.
    This solves a problem of the form
         argmin_x TV(x) + (1/(2 stepsize)) ||x - w||^2
    where TV(x) is the one-dimensional total variation
    Parameters
    ----------
    w: array
        vector of coefficients
    step_size: float
        step size (sometimes denoted gamma) in proximal objective function
    References
    ----------
    Condat, Laurent. "A direct algorithm for 1D total variation denoising."
    IEEE Signal Processing Letters (2013)
    """
    if w.dtype not in (np.float32, np.float64):
        raise ValueError('argument w must be array of floats')
    w = w.copy()
    output = np.empty_like(w)
    _prox_tv1d(step_size, w, output)
    return output


@njit
def _prox_tv1d(step_size, input, output):
    """low level function call, no checks are performed"""
    width = input.size + 1
    index_low = np.zeros(width, dtype=np.int32)
    slope_low = np.zeros(width, dtype=input.dtype)
    index_up  = np.zeros(width, dtype=np.int32)
    slope_up  = np.zeros(width, dtype=input.dtype)
    index     = np.zeros(width, dtype=np.int32)
    z         = np.zeros(width, dtype=input.dtype)
    y_low     = np.empty(width, dtype=input.dtype)
    y_up      = np.empty(width, dtype=input.dtype)
    s_low, c_low, s_up, c_up, c = 0, 0, 0, 0, 0
    y_low[0] = y_up[0] = 0
    y_low[1] = input[0] - step_size
    y_up[1] = input[0] + step_size
    incr = 1
    for i in range(2, width):
        y_low[i] = y_low[i-1] + input[(i - 1) * incr]
        y_up[i] = y_up[i-1] + input[(i - 1) * incr]
    y_low[width-1] += step_size
    y_up[width-1] -= step_size
    slope_low[0] = np.inf
    slope_up[0] = -np.inf
    z[0] = y_low[0]
    for i in range(1, width):
        c_low += 1
        c_up += 1
        index_low[c_low] = index_up[c_up] = i
        slope_low[c_low] = y_low[i]-y_low[i-1]
        while (c_low > s_low+1) and (slope_low[max(s_low, c_low-1)] <= slope_low[c_low]):
            c_low -= 1
            index_low[c_low] = i
            if c_low > s_low+1:
                slope_low[c_low] = (y_low[i]-y_low[index_low[c_low-1]]) / (i-index_low[c_low-1])
            else:
                slope_low[c_low] = (y_low[i]-z[c]) / (i-index[c])
        slope_up[c_up] = y_up[i]-y_up[i-1]
        while (c_up > s_up+1) and (slope_up[max(c_up-1, s_up)] >= slope_up[c_up]):
            c_up -= 1
            index_up[c_up] = i
            if c_up > s_up + 1:
                slope_up[c_up] = (y_up[i]-y_up[index_up[c_up-1]]) / (i-index_up[c_up-1])
            else:
                slope_up[c_up] = (y_up[i]-z[c]) / (i-index[c])
        while (c_low == s_low+1) and (c_up > s_up+1) and (slope_low[c_low] >= slope_up[s_up+1]):
            c += 1
            s_up += 1
            index[c] = index_up[s_up]
            z[c] = y_up[index[c]]
            index_low[s_low] = index[c]
            slope_low[c_low] = (y_low[i]-z[c]) / (i-index[c])
        while (c_up == s_up+1) and (c_low>s_low+1) and (slope_up[c_up]<=slope_low[s_low+1]):
            c += 1
            s_low += 1
            index[c] = index_low[s_low]
            z[c] = y_low[index[c]]
            index_up[s_up] = index[c]
            slope_up[c_up] = (y_up[i]-z[c]) / (i-index[c])
    for i in range(1, c_low - s_low + 1):
        index[c+i] = index_low[s_low+i]
        z[c+i] = y_low[index[c+i]]
    c = c + c_low-s_low
    j, i = 0, 1
    while i <= c:
        a = (z[i]-z[i-1]) / (index[i]-index[i-1])
        while j < index[i]:
            output[j * incr] = a
            output[j * incr] = a
            j += 1
        i += 1
    return


@njit
def prox_tv1d_cols(stepsize, a, n_rows, n_cols):
    """apply prox_tv1d along columns of the matri a
    """
    A = a.reshape((n_rows, n_cols))
    out = np.empty_like(A)
    for i in range(n_cols):
        _prox_tv1d(stepsize, A[:, i], out[:, i])
    return out.ravel()


@njit
def prox_tv1d_rows(stepsize, a, n_rows, n_cols):
    """apply prox_tv1d along rows of the matri a
    """
    A = a.reshape((n_rows, n_cols))
    out = np.empty_like(A)
    for i in range(n_rows):
        _prox_tv1d(stepsize, A[i, :], out[i, :])
    return out.ravel()


def Read_seg_file():
    seg_start = []
    seg_end = []
    seg_count = []
    seg_len = []
    with open("seg", 'r') as f:
        for line in f:
            linestrlist = line.strip().split('\t')
            if len(linestrlist) > 5:
                start = int(linestrlist[2]) - 1
                end = int(linestrlist[3]) - 1
                seg_start.append(start)
                seg_end.append(end)
                seg_len.append(int(linestrlist[4]))
                seg_count.append(float(linestrlist[5]))
    seg_start = np.array(seg_start)
    seg_end = np.array(seg_end)
    seg_count = np.array(seg_count)
    return seg_start, seg_end, seg_count, seg_len


def seg_RDposMQ(RD, binHead, MQ, seg_start, seg_end, seg_count, binSize):
    seg_RD = np.full(len(seg_count), 0.0)
    seg_MQ = np.full(len(seg_count), 0.0)
    seg_Start = np.full(len(seg_count), 0)
    seg_End = np.full(len(seg_count), 0)
    for i in range(len(seg_RD)):
        seg_RD[i] = np.mean(RD[seg_start[i]:seg_end[i]])
        seg_MQ[i] = np.mean(MQ[seg_start[i]:seg_end[i]])
        seg_Start[i] = binHead[seg_start[i]] * binSize + 1
        if seg_end[i] == len(binHead):
            seg_end[i] = len(binHead) - 1
        seg_End[i] = binHead[seg_end[i]] * binSize + binSize
    return seg_RD, seg_MQ, seg_Start, seg_End


def filter_range(mode, threshold, svstart, svend, svRD, svMQ, averscores):
    SVRD = []
    SVMQ = []
    SVstart = []
    SVend = []
    SVscores = []
    type = []
    for i in range(len(svRD)):
        if svRD[i] > mode and svMQ[i] > modeMQ and averscores[i] > threshold and svend[i] > svstart[i]:
            SVRD.append(svRD[i])
            SVMQ.append(svMQ[i])
            SVstart.append(svstart[i])
            SVend.append(svend[i])
            SVscores.append(averscores[i])
            type.append("TANDUP")
    return SVRD, SVMQ, SVstart, SVend, SVscores, type


def combiningSV(SVRD, SVMQ, SVstart, SVend, SVscores):
    # SV_chr = seg_chr[index]
    SV_Start = []
    SV_End = []
    SV_RD = []
    SV_MQ = []
    SV_scores = []
    SVtype = np.full(len(SVRD), 1)
    for i in range(len(SVRD) - 1):
        if SVend[i] + 1 == SVstart[i + 1]:
            SVstart[i + 1] = SVstart[i]
            SVRD[i + 1] = np.mean(SVRD[i:i + 1])
            SVMQ[i + 1] = np.mean(SVMQ[i:i + 1])
            SVscores[i + 1] = np.mean(SVscores[i:i + 1])
            SVtype[i] = 0
    for i in range(len(SVRD)):
        if SVtype[i] == 1:
            SV_Start.append(SVstart[i])
            SV_End.append(SVend[i])
            SV_RD.append(SVRD[i])
            SV_MQ.append(SVMQ[i])
            SV_scores.append(SVscores[i])
    return SV_RD, SV_MQ, SV_Start, SV_End, SV_scores


def get_exact_position(chrname, binSize, bam, str1, SVstart, SVend, type):
    # str1=100M
    # bp_exact_position = []
    # maxbin = 5
    SVStart = []
    SVEnd = []
    SVlen = np.full(len(SVstart), 0)
    SVtype = []

    discordantrange = np.empty(len(SVstart), dtype=object)
    dbf = pysam.AlignmentFile(drbam, 'rb')
    for i in range(len(SVstart)):
        discordantrange[i] = []
        SVlen[i] = abs(SVend[i] - SVstart[i])
        for r in dbf.fetch(chrname, SVstart[i] - maxbin * binSize,
                          SVend[i] + maxbin * binSize):
            if r.tlen != 0:
                discordantrange[i].append([r.reference_name, r.pos, r.cigarstring, r.pnext, r.tlen, SVlen[i], r.flag])
    discordantresult = "_range_discordant.txt"
    with open(bam + discordantresult, 'w') as f1:
        for i in range(len(discordantrange)):
            f1.write("\nthis is " + str(i) + " discordant range:\n")
            f1.writelines(str(discordantrange[i]))

    not100Mperrange = np.empty(len(SVstart), dtype=object)
    bf = pysam.AlignmentFile(bam, 'rb')
    # allcigar = np.empty(len(SVstart), dtype=object)
    for i in range(len(SVstart)):
        not100Mperrange[i] = []
        # allcigar[i] = []
        SVlen[i] = abs(SVend[i] - SVstart[i])
        if SVstart[i] - maxbin * binSize > SVend[i] + maxbin * binSize:
            continue
        for r in bf.fetch(chrname, SVstart[i] - maxbin * binSize,
                          SVend[i] + maxbin * binSize):
            if r.cigarstring != str1 and r.cigarstring != None and r.tlen != 0:
                not100Mperrange[i].append([r.reference_name, r.pos, r.cigarstring, r.pnext, r.tlen, SVlen[i], r.flag & 64, r.flag & 128, r.flag])

    cigarresult = "_range_ciagr.txt"
    with open(bam + cigarresult, 'w') as f1:
        for i in range(len(not100Mperrange)):
            f1.write("\nthis is " + str(i) + " big range:\n")
            f1.writelines(str(not100Mperrange[i]))
            start = 0
            end = 0
            startlist = []
            endlist = []
            for j in range(len(not100Mperrange[i])):
                pos = not100Mperrange[i][j][2].index('M')
                if pos == 4 or pos == 5:
                    if SVstart[i] - maxbin * binSize < not100Mperrange[i][j][1] < SVstart[i] + maxbin * binSize:
                        startlist.append(not100Mperrange[i][j][1])
                if pos == 1 or pos == 2:
                    if SVend[i] - maxbin * binSize < not100Mperrange[i][j][1] < SVend[i] + maxbin * binSize:
                        endlist.append(not100Mperrange[i][j][1] + (int)(not100Mperrange[i][j][2][0: pos]) - 1)
            if len(startlist) != 0:
                start = max(set(startlist), key=startlist.count)
                SVlen[i] = SVend[i] - start
            if len(endlist) != 0:
                end = max(set(endlist), key=endlist.count)
                SVlen[i] = end - SVstart[i]
            if len(startlist) != 0 and len(endlist) != 0:
                SVlen[i] = end - start
            dr_pos = []
            dr_pnext = []
            for l in range(len(discordantrange[i])):
                if abs(SVlen[i] - abs(discordantrange[i][l][4])) < (maxbin+2)*binSize \
                        and abs(discordantrange[i][l][1] - SVstart[i]) < maxbin*binSize:
                    dr_pos.append(discordantrange[i][l][1])
                    dr_pnext.append(discordantrange[i][l][3])
            if start == 0:
                if dr_pos:
                    start = dr_pos[0]
            if end == 0:
                if dr_pnext:
                    end = dr_pnext[-1] + 100
            if end > start and start != 0 and len(dr_pos) != 0:
                SVStart.append(start)
                SVEnd.append(end)
                SVtype.append(type[i])
    return SVStart, SVEnd, SVtype


def Write_step1_file(SV_RD, SV_MQ, SV_Start, SV_End, SV_scores):
    """
    write svdata file
    svRD, svMQ, svstart, svend, averscores
    """
    output = open(bam + '_merge_step1.csv', "w")
    output.write(
        "SV_RD" + '\t' + "SV_MQ" + '\t' + "SV_Start" + '\t' + "SV_End" + '\t' + "SV_averscores" + '\n')
    for i in range(len(SV_RD)):
        output.write(
            str(SV_RD[i]) + '\t' + str(SV_MQ[i]) + '\t' + str(SV_Start[i]) +
            '\t' + str(SV_End[i]) + '\t' + str(SV_scores[i]) + '\t' + '\n')


def Write_step2_file(SV_RD, SV_MQ, SV_Start, SV_End, SV_scores):
    """
    write svdata file
    svRD, svMQ, svstart, svend, averscores
    """
    output = open(bam + '_filter_step2.csv', "w")
    output.write(
        "SV_RD" + '\t' + "SV_MQ" + '\t' + "SV_Start" + '\t' + "SV_End" + '\t' + "SV_scores" + '\n')
    for i in range(len(SV_RD)):
        output.write(
            str(SV_RD[i]) + '\t' + str(SV_MQ[i]) + '\t' + str(SV_Start[i]) +
            '\t' + str(SV_End[i]) + '\t' + str(SV_scores[i]) + '\n')


def Write_data_file(chr, seg_start, seg_end, seg_count, seg_mq, scores, labels):
    """
    write knn data file
    chr, start, end, rd, mq, score, label
    """
    output = open('_KNN_Score.csv', "w")
    output.write(
        "chr" + '\t' + "start" + '\t' + "end" + '\t' + "read depth" + '\t' + "map quality" + '\t' + "score" + '\t' + "labels" +'\n')
    for i in range(len(scores)):
        output.write(
            str(chr[i]) + '\t' + str(seg_start[i]) + '\t' + str(seg_end[i]) +
            '\t' + str(seg_count[i]) + '\t' + str(seg_mq[i]) + '\t' + str(scores[i]) + '\t' + str(labels[i]) + '\n')


def boxplot(averscores):
    four = pd.Series(averscores).describe()
    Q1 = four['25%']
    Q3 = four['75%']
    IQR = Q3 - Q1
    upper = Q3 + 0.6 * IQR
    # lower = Q1 - 0.8 * IQR
    return upper


# def plotRDMQ(RD, MQ):
#     # plot RD and MQ
#     plt.scatter(RD, MQ, s=3, c="black")
#     plt.xlabel("RD")
#     plt.ylabel("MQ")
#     plt.show()
#
#
# def plotscores(scores):
#     y = np.arange(1, len(scores) + 1, 1)
#     plt.scatter(scores, y, s=3)
#     plt.xlim(0, max(scores) + min(scores))
#     plt.ylim(1, len(scores) + 1)
#     plt.xlabel("scores")
#     plt.savefig(bam + '_scores.jpg')
#     plt.show()


start = time.time()
binSize = 1000
alpha = 0.25
refpath = sys.argv[1]
bam = sys.argv[2]
drbam = sys.argv[3]
str1 = sys.argv[4]

reffile = refpath.split("/")[-1]  # chr21.fa
chrname = reffile.split('.fa')[0]  # chr21
chrnumb = chrname.strip('chr')  # 21

refList = []
refList = read_ref_file(refpath, refList)
chrLen = len(refList)
print("chrLen:" + str(chrLen))
print("Read bam file:", bam)

ReadCount = np.full(chrLen, 0)
mapq = np.full(chrLen, 0)
ReadCount, mapq = get_RCmapq(bam, ReadCount, mapq)
binNum = int(chrLen / binSize) + 1
pos, RD, MQ, GC = ReadDepth(mapq, ReadCount, binNum, refList, binSize)
RD, MQ = TV_smoothnoise(RD, MQ)
mode = np.mean(RD)
modeMQ = np.mean(MQ)
print("modeRD:" + str(mode))
print("modeMQ:" + str(modeMQ))
with open('RD', 'w') as file:
    for i in range(len(RD)):
        file.write(str(RD[i]) + '\n')
# segment
subprocess.call('Rscript CBS_data.R ', shell=True)
seg_start, seg_end, seg_count, seg_len = Read_seg_file()

if len(seg_start) != 0:
    seg_RD, seg_MQ, seg_Start, seg_End = seg_RDposMQ(RD, pos, MQ, seg_start, seg_end, seg_count, binSize)
    seg_chr = []
    seg_chr.extend(chrnumb for i in range(len(seg_RD)))
    seg_chr = np.array(seg_chr)
    RDMQ = np.full((len(seg_RD), 2), 0.0)
    RDMQ[:, 0] = seg_RD
    RDMQ[:, 1] = seg_MQ
    scler = StandardScaler()
    rdmq_scaler = scler.fit_transform(RDMQ)
    RDMQ_scaler = pd.DataFrame(rdmq_scaler)
    # KNN
    n_neighbors = int(len(seg_RD) * 0.2)
    print("n_neighbors:", n_neighbors)
    clf = KNN(n_neighbors=n_neighbors, method='mean')
    clf.fit(RDMQ_scaler)
    labels = clf.labels_
    scores = clf.decision_scores_
    # Write_data_file(seg_chr, seg_Start, seg_End, seg_RD, seg_MQ, scores, labels)

    maxbin = 2
    threshold = boxplot(scores)
    print("threshold:" + str(threshold))
    SVRD, SVMQ, SVstart, SVend, SVscores, type = filter_range(mode, threshold, seg_Start, seg_End, seg_RD, seg_MQ, scores)
    # Write_step2_file(SVRD, SVMQ, SVstart, SVend, SVscores)
    SVRD, SVMQ, SVstart, SVend, SVscores = combiningSV(SVRD, SVMQ, SVstart, SVend, SVscores)
    SV_range = np.full((len(SVstart), 2), 0)
    SV_range[:, 0] = SVstart
    SV_range[:, 1] = SVend
    print("filter_range:" + str(len(SV_range)))

    SVStart, SVEnd, SVtype = get_exact_position(chrname, binSize, bam, str1, SVstart, SVend, type)
    print("SR_range:" + str(len(SVStart)))
    output = open(bam + '_result.txt', "w")
    for i in range(len(SVStart)):
        output.write(
            str(chrname) + '\t' + str(SVStart[i]) + '\t' + str(SVEnd[i]) + '\t' + str(SVtype[i]) +
            '\t' + '\n')
        print(chrname, SVStart[i], SVEnd[i], SVtype[i])
else:
    output0 = open(bam + '_result.txt', "w")
    print("no result")

end = time.time()
print(" ** the run time of is: ", end - start, " **")

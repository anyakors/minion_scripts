# Implementation of algorithm from https://stackoverflow.com/a/22640362/6029703
import numpy as np
import pylab

def thresholding_algo(y, lag, threshold, influence):
    signals = np.zeros(len(y))
    filteredY = np.array(y)
    avgFilter = [0]*len(y)
    stdFilter = [0]*len(y)
    avgFilter[lag - 1] = np.mean(y[0:lag])
    stdFilter[lag - 1] = np.std(y[0:lag])
    for i in range(lag, len(y)):
        if abs(y[i] - avgFilter[i-1]) > threshold * stdFilter [i-1]:
            if y[i] > avgFilter[i-1]:
                signals[i] = 1
            else:
                signals[i] = -1

            filteredY[i] = influence * y[i] + (1 - influence) * filteredY[i-1]
            avgFilter[i] = np.mean(filteredY[(i-lag+1):i+1])
            stdFilter[i] = np.std(filteredY[(i-lag+1):i+1])
        else:
            signals[i] = 0
            filteredY[i] = y[i]
            avgFilter[i] = np.mean(filteredY[(i-lag+1):i+1])
            stdFilter[i] = np.std(filteredY[(i-lag+1):i+1])

    return dict(signals = np.asarray(signals),
                avgFilter = np.asarray(avgFilter),
                stdFilter = np.asarray(stdFilter))


def peak_frequency(signals):
    k = 0
    last = 0
    periods = []
    for i in range(0,len(signals)):
        if signals[i]!=last:
            periods.append(i-k)
            k = i
            last = signals[i]
    if len(periods)>5:
        frequency = np.mean(periods)
    return frequency


def drop_occurence(signals, freq_threshold):
    k = 0 #last change
    last = 0 #last value
    drops = []
    for i in range(0,len(signals)):
        if signals[i]==last and last==-1:
            current = i-k #length of constant value
            if current<freq_threshold:
                drops.append(0)
            else:
                drops.append(1)
        else:
            current = 0
            last = signals[i]
            k = i
            drops.append(0)
    return drops
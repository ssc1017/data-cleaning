import pandas as pd
import numpy as np
import heapq
import sys
from sklearn.cluster import KMeans
from sklearn.cluster import Birch

cluster_num = 100
dataset = pd.read_csv("dj4e-3xrn.tsv", sep='\t')


class MBR:
    def __init__(self, points):
        #(rot_angle, area, width, height, center_point, corner_points) = minBoundingRect(points)
        self.points = points
        self.num = points.shape[0]
        self.lower = sys.maxint
        self.upper = sys.maxint
        self.is_candidate = 0
        self.endpoint_max = np.max(points,axis=0)
        self.endpoint_min = np.min(points,axis=0)


class dataPoint:
    def __init__(self, data):
        self.DkDist = sys.maxint
        self.data = data


def dist(firstData, secondData):
    return (((firstData - secondData) ** 2).sum()) ** 0.5


def MINDIST(P, Q):
    P_max = P.endpoint_max.tolist()
    P_min = P.endpoint_min.tolist()
    Q_max = Q.endpoint_max.tolist()
    Q_min = Q.endpoint_min.tolist()
    x = []

    for i in range(len(P_max)):
        if Q_max[i] < P_min[i]:
            x.append(P_min[i] - Q_max[i])
        elif P_max[i] < Q_min[i]:
            x.append(Q_min[i] - P_max[i])
        else:
            x.append(0)

    x = map(lambda x: x**2, x)
    return reduce(lambda x, y: x + y, x)


def MAXDIST(P, Q):
    P_max = P.endpoint_max.tolist()
    P_min = P.endpoint_min.tolist()
    Q_max = Q.endpoint_max.tolist()
    Q_min = Q.endpoint_min.tolist()
    x = []

    for i in range(len(P_max)):
        x.append(max(abs(Q_max[i]-P_min[i]), abs(P_max[i]-Q_min[i])))

    x = map(lambda x: x**2, x)
    return reduce(lambda x, y: x + y, x)


def heapNumPoints(heap):
    sum = 0
    for i in range(len(heap)):
        sum += heap[i][2].num
    return sum


def computeLowerUpper(pset, P, k, minDkDist, self_num):
    lowerHeap=[] #maxheap
    upperHeap=[] #maxheap
    for i in range(len(pset)):
        if i == self_num:
            continue

        mindist = MINDIST(P, pset[i])
        if mindist < P.lower:
            heapq.heappush(lowerHeap, (-mindist, i, pset[i]))
            lowerHeapNumPoints = heapNumPoints(lowerHeap)

            topNum = lowerHeap[0][2].num
            while (lowerHeapNumPoints - topNum) >= k:
                top = heapq.heappop(lowerHeap)
                lowerHeapNumPoints = lowerHeapNumPoints - topNum
                topNum = lowerHeap[0][2].num
            if heapNumPoints(lowerHeap) >= k:
                P.lower = MINDIST(P, lowerHeap[0][2])

        maxdist = MAXDIST(P, pset[i])
        if maxdist < P.upper:
            heapq.heappush(upperHeap, (-maxdist, i, pset[i]))
            upperHeapNumPoints = heapNumPoints(upperHeap)

            topNum = upperHeap[0][2].num
            while (upperHeapNumPoints - topNum) >= k:
                top = heapq.heappop(upperHeap)
                upperHeapNumPoints = upperHeapNumPoints - topNum
                topNum = upperHeap[0][2].num
            if heapNumPoints(upperHeap) >=k:
                P.upper = MAXDIST(P, lowerHeap[0][2])
            if P.upper <= minDkDist:
                return


def computeCandidatePartitions(pset, k, n):
    minDkDist = 0
    partHeap = [] #minheap
    for i in range(len(pset)):
        computeLowerUpper(pset, pset[i], k, minDkDist, i)
        if pset[i].lower > minDkDist:
            heapq.heappush(partHeap, (pset[i].lower, i, pset[i]))
            sum = heapNumPoints(partHeap)

            topNum = partHeap[0][2].num
            while (sum - topNum) >= n:
                top=heapq.heappop(partHeap)
                sum = sum-topNum
                topNum = partHeap[0][2].num

            if heapNumPoints(partHeap) >= n:
                minDkDist = partHeap[0][2].lower

    candSet = []
    for i in range(len(pset)):
        if pset[i].upper >= minDkDist:
            candSet.append(pset[i].points)  # only keep the points of MBR
            pset[i].is_candidate = 1

    print 'the number of candidates: ' + str(len(candSet))

    combined_candSet = reduce(lambda x, y: np.concatenate((x, y), axis=0), candSet)

    print 'the number of the rest tuples: ' + str(combined_candSet.shape[0])
    print('finished')

    return combined_candSet


def getPartitionsBirch(data):
    print('doing partition-based algorithm ...')
    num_samples = data.shape[0]

    num_clusters = int(float(num_samples) / 5)
    birch_model = Birch(threshold=0.1, n_clusters=num_clusters)
    birch_model.fit(data)
    labels = birch_model.labels_
    n_clusters = np.unique(labels).size
    print 'the number of partitions: ' + str(n_clusters)

    ls = [[] for i in range(num_clusters)]
    for i, l in enumerate(labels):
        ls[l].append(data[i])

    all_MBR=[]
    for i in range(num_clusters):
        all_MBR.append(MBR(np.array(ls[i])))

    return all_MBR


def getPartitionsKmenas(data):
    kmeans_model = KMeans(n_clusters=cluster_num).fit(data)
    ls = [[] for i in range(cluster_num)]
    for i, l in enumerate(kmeans_model.labels_):
        ls[l].append(data[i])

    all_MBR=[]
    for i in range(cluster_num):
        all_MBR.append(MBR(np.array(ls[i])))

    return all_MBR

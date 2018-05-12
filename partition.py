import pandas as pd
import numpy as np
import heapq
import sys
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
#from birch import Birch
from sklearn.cluster import Birch


cluster_num = 100
dataset = pd.read_csv("dj4e-3xrn.tsv", sep='\t')
scalar = StandardScaler()

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
        #print (i)
        #print (minDkDist)
        computeLowerUpper(pset, pset[i], k, minDkDist, i)
        if pset[i].lower > minDkDist:
            heapq.heappush(partHeap, (pset[i].lower, i, pset[i]))
            sum = heapNumPoints(partHeap)

            topNum = partHeap[0][2].num
            while (sum - topNum) >= n:
                top=heapq.heappop(partHeap)
                sum = sum-topNum
                topNum = partHeap[0][2].num

            #print (heapNumPoints(partHeap))
            #print (partHeap[0][2].lower)
            if heapNumPoints(partHeap) >= n:
                minDkDist = partHeap[0][2].lower

    candSet = []
    for i in range(len(pset)):
        #print (pset[i].upper)
        if pset[i].upper >= minDkDist:
            candSet.append(pset[i])
            pset[i].is_candidate=1
        #print (pset[i].is_candidate)

    cnt=0
    for i in range(len(pset)):
        if pset[i].is_candidate==1:
            cnt +=1
    print cnt
    return candSet

def getPartitionsBirch():
    data = dataset.dropna(axis=0, how='any')[['peer_index_', 'overall_score', 'progress_category_score']].values

    num_samples = data.shape[0]
    indices = np.arange(num_samples)

    num_clusters = int(float(num_samples) / 5)
    print (num_clusters)
    birch_model = Birch(threshold=0.1, n_clusters=num_clusters)
    birch_model.fit(data)
    labels = birch_model.labels_
    n_clusters = np.unique(labels).size
    print (labels)

    ls = [[] for i in range(num_clusters)]
    for i, l in enumerate(labels):
        ls[l].append(data[i])

    all_MBR=[]
    for i in range(num_clusters):
        all_MBR.append(MBR(np.array(ls[i])))

    a = all_MBR[3]
    # print a.points
    # print a.endpoint_min
    # print a.endpoint_max
    # print a.num

    return all_MBR

def getPartitionsKmenas():
    data = dataset.dropna(axis=0, how='any')[['peer_index_', 'overall_score', 'progress_category_score']].values
    #data = dataset.dropna(axis=0, how='any')[['peer_index_', 'overall_score']].values
    #data = scalar.fit_transform(data)
    kmeans_model = KMeans(n_clusters=cluster_num).fit(data)
    ls = [[] for i in range(cluster_num)]
    for i, l in enumerate(kmeans_model.labels_):
        ls[l].append(data[i])
    #print(np.array(ls[10]))
    all_MBR=[]
    for i in range(cluster_num):
        all_MBR.append(MBR(np.array(ls[i])))

    return all_MBR

if __name__ == '__main__':
    N = 20
    k = 100
    #birch()
    computeCandidatePartitions(getPartitionsBirch(), k, N)
    # heap = computeOutliersIndex(N, k)
    # for i in range(len(heap)):
    #     print scalar.inverse_transform(heap[i][2].data)

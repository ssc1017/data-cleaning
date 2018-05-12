import pandas as pd
import numpy as np
import heapq
import sys
from sklearn.preprocessing import StandardScaler

# dataset = pd.read_csv("1.csv", sep=',')


dataset = pd.read_csv("dj4e-3xrn.tsv", sep='\t')
scalar = StandardScaler()


class dataPoint:
    def __init__(self, data):
        self.DkDist = sys.maxint
        self.data = data


def dist(firstData, secondData):
    return (((firstData - secondData) ** 2).sum()) ** 0.5


def getKthNeighborDist(data, size, p, k, minDkDist):
    heap = []
    for i in range(size):
        d = dist(p.data, data[i])

        if d == 0:
            continue

        if d < p.DkDist:
            heapq.heappush(heap, (-d, i, data[i])) #set i to avoid same (d, data[i])
            if len(heap) > k:
                top = heapq.heappop(heap)
            if len(heap) == k:
                p.DkDist = -1 * heap[0][0]
            if p.DkDist < minDkDist:
                return


def computeOutliersIndex(N, k):
    data = dataset.dropna(axis=0, how='any')[['peer_index_', 'overall_score']].values
    data = scalar.fit_transform(data)
    size = data.shape[0]
    minDkDist = 0
    heap = []

    for i in range(size):
        p = dataPoint(data[i])
        getKthNeighborDist(data, size, p, k, minDkDist)

        if p.DkDist > minDkDist:
            heapq.heappush(heap, (p.DkDist, i, p))
            if len(heap) > N:
                top = heapq.heappop(heap)
            if len(heap) == N:
                minDkDist = heap[0][0]
    return heap


if __name__ == '__main__':
    N = 10
    k = 10
    heap = computeOutliersIndex(N, k)
    for i in range(len(heap)):
        print scalar.inverse_transform(heap[i][2].data)

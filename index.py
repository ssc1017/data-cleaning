import heapq
import sys


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
            heapq.heappush(heap, (-d, i, data[i]))  # set i to avoid same (d, data[i])
            if len(heap) > k:
                top = heapq.heappop(heap)
            if len(heap) == k:
                p.DkDist = -1 * heap[0][0]
            if p.DkDist < minDkDist:
                return


def computeOutliersIndex(data, N, k):
    print('doing index-based algorithm ...')
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
    print('finished')
    return heap

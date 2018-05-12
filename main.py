import pandas as pd
from sklearn.preprocessing import StandardScaler
from partition import getPartitionsBirch, computeCandidatePartitions
from index import  computeOutliersIndex

dataset = pd.read_csv("dj4e-3xrn.tsv", sep='\t')
scalar = StandardScaler()

def preprocess():
    print('preprocessing...')
    data = dataset.dropna(axis=0, how='any')[['peer_index_', 'overall_score', 'progress_category_score']].values
    data = scalar.fit_transform(data)
    print('finished')
    return data

if __name__ == '__main__':
    N = 20
    k = 30

    data = preprocess()
    candidates = computeCandidatePartitions(getPartitionsBirch(data), k, N)
    outliers = computeOutliersIndex(candidates, N, k)
    for i in range(len(outliers)):
        print scalar.inverse_transform(outliers[i][2].data)
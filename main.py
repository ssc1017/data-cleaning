import pandas as pd
from sklearn.preprocessing import StandardScaler
from partition import getPartitionsKmeans, computeCandidatePartitions
from index import computeOutliersIndex

from pyspark.mllib.clustering import KMeans, KMeansModel
from pyspark import SparkContext
from pyspark.sql import SparkSession
import numpy as np
#from pyspark.ml.clustering import KMeans

dataset = pd.read_csv("dj4e-3xrn.tsv", sep='\t')
scalar = StandardScaler()


def preprocessSpark():
    spark = SparkSession.builder.appName("datacleaning").config("spark.some.config.option", "some-value").getOrCreate()
    df = spark.read.format('csv').options(header='true', inferschema='true', delimiter='\t').load("file:///Users/sichaoshu/PycharmProjects/dataCleaning/dj4e-3xrn.tsv")
    df.createOrReplaceTempView("df")

    data = spark.sql("SELECT peer_index_, overall_score FROM df WHERE peer_index_ IS NOT NULL AND overall_score IS NOT NULL")
    data.show()

    # sd = result.rdd
    # sd = sd.map(lambda x: np.array(x))
    #
    # clusters = KMeans.train(sd, 100, initializationMode='random', maxIterations=10)
    # for i in range(0, len(clusters.centers)):
    #     print("cluster " + str(i) + ": " + str(clusters.centers[i]))

    return scalar.fit_transform(data.toPandas().values)


# def test1():
#     sc = SparkContext()
#     tsvfile = sc.textFile('file:///Users/sichaoshu/PycharmProjects/dataCleaning/dj4e-3xrn.tsv')
#     sensordata = tsvfile.map(lambda line: line.split('\t'))
#     sdfilt = sensordata.filter(lambda x: np.count_nonzero(np.array([float(x[4]), float(x[5]), float(x[7])])) < 3)
#     sdfilt.count()


def preprocessLocal():
    #data = dataset.dropna(axis=0, how='any')[['peer_index_', 'overall_score', 'progress_category_score']].values
    data = dataset.dropna(axis=0, how='any')[['peer_index_', 'overall_score']].values  # delete all rows with any null values
    data = scalar.fit_transform(data)

    print data.shape[0]
    return data


if __name__ == '__main__':
    N = 10
    k = 30

    is_spark = 0

    if is_spark == 1:
        print('preprocessing...')
        data = preprocessSpark()
        print('finished')
    else:
        print('preprocessing...')
        data = preprocessLocal()
        print('finished')

    print('doing partition-based algorithm...')
    cluster_num = 200
    candidates = computeCandidatePartitions(getPartitionsKmeans(data, cluster_num), k, N)
    print('finished')

    print('doing index-based algorithm...')
    outliers = computeOutliersIndex(candidates, N, k)
    print('finished')

    for i in range(len(outliers)):
        print scalar.inverse_transform(outliers[i][2].data)
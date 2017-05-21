import pyspark
from pyspark.sql.session import SparkSession
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.mllib.tree import GradientBoostedTrees, GradientBoostedTreesModel
conf = pyspark.SparkConf().setAll([('spark.driver.memory', '30g'), ('spark.driver.host', '172.17.0.21'), ('spark.app.id', 'local-1492693477461'), ('spark.rdd.compress', 'True'), ('spark.serializer.objectStreamReset', '100'), ('spark.master', 'local[*]'), ('spark.executor.id', 'driver'), ('spark.submit.deployMode', 'client'), ('spark.driver.port', '39274'), ('spark.app.name', 'PySparkShell')])
sc = pyspark.SparkContext(conf=conf)
spark = SparkSession(sc)

from gensim import models
import numpy as np
from utils import CutAndrmStopWords
kemmodel = models.KeyedVectors.load_word2vec_format('med400.model.bin', binary=True)

class SwingerSpark(object):
    """docstring for SwingerSpark"""
    def __init__(self):
        swingerModel = self.load(path)

    def predict(self, sentence):
        for i in CutAndrmStopWords(sentence):
            vec = np.zeros(400)
            for j in i:
                try:
                    vec = np.add(vec, model[j])
                except Exception as e:
                    pass
        return swingerModel.predict(vec.tolist())

    @staticmethod
    def score(PredictionAndLabel):
        # Instantiate metrics object
        metrics = BinaryClassificationMetrics(PredictionAndLabel)
        # Area under ROC curve
        print("Area under ROC = %s" % metrics.areaUnderROC)
        return metrics.areaUnderROC

    @staticmethod
    def featureExtraction(x):
        vector = []
        if x[0]==u'p':
            vector.append(1)
        else:
            vector.append(0)
        for i in range(1,401):
            vector.append(float(x[i]))
        return vector

    def train(self):
        neg_df = spark.read  .format('org.apache.spark.sql.execution.datasources.csv.CSVFileFormat')  .option('header', 'true')  .load('neg.csv')
        pos_df = spark.read  .format('org.apache.spark.sql.execution.datasources.csv.CSVFileFormat')  .option('header', 'true')  .load('pos.csv')
        test_pos_df = spark.read  .format('org.apache.spark.sql.execution.datasources.csv.CSVFileFormat')  .option('header', 'true')  .load('ptest.csv')
        test_neg_df = spark.read  .format('org.apache.spark.sql.execution.datasources.csv.CSVFileFormat')  .option('header', 'true')  .load('ntest.csv')
        training_df = neg_df.union(pos_df)
        test_df = test_neg_df.union(test_pos_df)

        labelpointRdd = training_df.rdd.map(featureExtraction).map(lambda x: LabeledPoint(x[0],x[1:])).cache()
        TestlabelpointRdd = test_df.rdd.map(featureExtraction).map(lambda x: LabeledPoint(x[0],x[1:])).cache()

        GBTmodel = GradientBoostedTrees.trainClassifier(labelpointRdd,categoricalFeaturesInfo={}, numIterations=75)
        predictions = GBTmodel.predict(TestlabelpointRdd.map(lambda x: x.features))
        labelsAndPredictions = TestlabelpointRdd.map(lambda lp: lp.label).zip(predictions)

        # save model
        GBTmodel.save(sc, '.')
        return score(labelsAndPredictions)

    def load(self, path):
        return GradientBoostedTreesModel.load(sc, path)

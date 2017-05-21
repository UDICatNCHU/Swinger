import pyspark
conf = pyspark.SparkConf().setAll([('spark.driver.memory', '30g'), ('spark.driver.host', '172.17.0.21'), ('spark.app.id', 'local-1492693477461'), ('spark.rdd.compress', 'True'), ('spark.serializer.objectStreamReset', '100'), ('spark.master', 'local[*]'), ('spark.executor.id', 'driver'), ('spark.submit.deployMode', 'client'), ('spark.driver.port', '39274'), ('spark.app.name', 'PySparkShell')])
spark = pyspark.SparkContext(conf=conf)
neg_df = spark.read  .format('org.apache.spark.sql.execution.datasources.csv.CSVFileFormat')  .option('header', 'true')  .load('neg.csv')
pos_df = spark.read  .format('org.apache.spark.sql.execution.datasources.csv.CSVFileFormat')  .option('header', 'true')  .load('pos.csv')
test_pos_df = spark.read  .format('org.apache.spark.sql.execution.datasources.csv.CSVFileFormat')  .option('header', 'true')  .load('ptest.csv')
test_neg_df = spark.read  .format('org.apache.spark.sql.execution.datasources.csv.CSVFileFormat')  .option('header', 'true')  .load('ntest.csv')
training_df = neg_df.union(pos_df)
test_df = test_neg_df.union(test_pos_df)
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.evaluation import BinaryClassificationMetrics

def featureExtraction(x):
    vector = []
    if x[0]==u'p':
        vector.append(1)
    else:
        vector.append(0)
    for i in range(1,401):
        vector.append(float(x[i]))
    return vector

def score(PredictionAndLabel):
    # Instantiate metrics object
    metrics = BinaryClassificationMetrics(PredictionAndLabel)
    # Area under ROC curve
    print("Area under ROC = %s" % metrics.areaUnderROC)
    return metrics.areaUnderROC


labelpointRdd = training_df.rdd.map(featureExtraction).map(lambda x: LabeledPoint(x[0],x[1:])).cache()

TestlabelpointRdd = test_df.rdd.map(featureExtraction).map(lambda x: LabeledPoint(x[0],x[1:])).cache()

## enable this line if data set self validation 
# labelpointRdd, TestlabelpointRdd = labelpointRdd.randomSplit([0.8,0.2])  
from pyspark.mllib.classification import LogisticRegressionWithLBFGS
model = LogisticRegressionWithLBFGS.train(labelpointRdd)
PredictionAndLabel = TestlabelpointRdd.map(lambda x: (float(model.predict(x.features)), x.label))
score(PredictionAndLabel)

# ## Use Decision Tree Model

# In[158]:

from pyspark.mllib.tree import DecisionTree
DTModel = DecisionTree.trainClassifier(labelpointRdd,
        numClasses=2,
        categoricalFeaturesInfo={},
        impurity="entropy",
        maxDepth=20,
        maxBins=20)
prediction2 = DTModel.predict(TestlabelpointRdd.map(lambda x: x.features))
predictionAndLabels2 = prediction2.zip(TestlabelpointRdd.map(lambda x: x.label))
predictionAndLabelsPair = predictionAndLabels2.map(lambda x: (float(x[0]), x[1]))
score(predictionAndLabelsPair)

# ## Use Random Forest Tree Model

# In[176]:

from pyspark.mllib.tree import RandomForest, RandomForestModel
ForestModel = RandomForest.trainClassifier(labelpointRdd, numClasses=2, categoricalFeaturesInfo={},
                                     numTrees=40, featureSubsetStrategy="auto",
                                     impurity='entropy', maxDepth=20, maxBins=20)
predictions = ForestModel.predict(TestlabelpointRdd.map(lambda x:x.features))
PredictionAndLabel = predictions.zip(TestlabelpointRdd.map(lambda x:x.label))
predictionAndLabelsPair = PredictionAndLabel.map(lambda x:(float(x[0]), x[1]))
score(predictionAndLabelsPair)


# ## Use Grident Boost Tree Model

# In[173]:

from pyspark.mllib.tree import GradientBoostedTrees, GradientBoostedTreesModel

adaboost = []
for i in range(1,500,5):
    GBTmodel = GradientBoostedTrees.trainClassifier(labelpointRdd,categoricalFeaturesInfo={}, numIterations=i)
    predictions = GBTmodel.predict(TestlabelpointRdd.map(lambda x: x.features))
    labelsAndPredictions = TestlabelpointRdd.map(lambda lp: lp.label).zip(predictions)
    adaboost.append(score(labelsAndPredictions))


print(adaboost)
print(max(adaboost))
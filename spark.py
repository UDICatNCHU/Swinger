# coding: utf-8
from pyspark.sql import SparkSession


spark = SparkSession.builder.getOrCreate()

neg_df = spark.read  .format('org.apache.spark.sql.execution.datasources.csv.CSVFileFormat')  .option('header', 'true')  .load('swift://DefaultProjectyfannchuedutw.' + name + '/ntrain.csv')


# In[15]:

pos_df = spark.read  .format('org.apache.spark.sql.execution.datasources.csv.CSVFileFormat')  .option('header', 'true')  .load('swift://DefaultProjectyfannchuedutw.' + name + '/ptrain.csv')


# In[14]:

test_pos_df = spark.read  .format('org.apache.spark.sql.execution.datasources.csv.CSVFileFormat')  .option('header', 'true')  .load('swift://DefaultProjectyfannchuedutw.' + name + '/ptest.csv')


# In[16]:

test_neg_df = spark.read  .format('org.apache.spark.sql.execution.datasources.csv.CSVFileFormat')  .option('header', 'true')  .load('swift://DefaultProjectyfannchuedutw.' + name + '/ntest.csv')


# In[11]:

neg_df.count()


# In[19]:

from pyspark.sql import functions as F
neg_df = neg_df.withColumn('p', F.when(neg_df[0]==u'p', u'n'))


# In[23]:

from pyspark.sql import functions as F
test_neg_df = test_neg_df.withColumn('p', F.when(test_neg_df[0]==u'p', u'n'))


# In[25]:

training_df = neg_df.union(pos_df)
training_df.count()


# In[26]:

test_df = test_neg_df.union(test_pos_df)


# In[27]:

training_rdd = training_df.rdd


# In[143]:

from pyspark.mllib.regression import LabeledPoint

def featureExtraction(x):
    vector = []
    if x[0]==u'p':
        vector.append(1)
    else:
        vector.append(0)
    for i in range(1,401):
        vector.append(float(x[i]))
    return vector


# ### Training Data and Testing Data Setting Here

# In[147]:

labelpointRdd = training_rdd.map(featureExtraction).map(lambda x: LabeledPoint(x[0],x[1:])).cache()


# In[148]:

TestlabelpointRdd = test_df.rdd.map(featureExtraction).map(lambda x: LabeledPoint(x[0],x[1:]))


# In[157]:

## enable this line if data set self validation 
labelpointRdd, TestlabelpointRdd = labelpointRdd.randomSplit([0.8,0.2])  


# ## Use Logistic Regression Model

# In[175]:

from pyspark.mllib.classification import LogisticRegressionWithLBFGS
model = LogisticRegressionWithLBFGS.train(labelpointRdd)
PredictionAndLabel = TestlabelpointRdd.map(lambda x: (model.predict(x.features), x.label))
PredictionAndLabelMaped = PredictionAndLabel.map(lambda (x,y): (float(x), y))
print "錯誤率", PredictionAndLabelMaped.filter(lambda (v, p): v != p).count()/float(TestlabelpointRdd.count())


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


# In[174]:

predictionAndLabels2 = prediction2.zip(TestlabelpointRdd.map(lambda x: x.label))
predictionAndLabelsPair = predictionAndLabels2.map(lambda (x,y): (float(x), y))
print "錯誤率", predictionAndLabelsPair.filter(lambda (v, p): v != p).count()/float(TestlabelpointRdd.count())


# ## Use Random Forest Tree Model

# In[176]:

from pyspark.mllib.tree import RandomForest, RandomForestModel
ForestModel = RandomForest.trainClassifier(labelpointRdd, numClasses=2, categoricalFeaturesInfo={},
                                     numTrees=40, featureSubsetStrategy="auto",
                                     impurity='entropy', maxDepth=20, maxBins=20)

predictions = ForestModel.predict(TestlabelpointRdd.map(lambda x: x.features))
predictionAndLabels = predictions.zip(TestlabelpointRdd.map(lambda x: x.label))
print "錯誤率", predictionAndLabels.filter(lambda (v, p): v != p).count()/float(TestlabelpointRdd.count())


# ## Use Grident Boost Tree Model

# In[173]:

from pyspark.mllib.tree import GradientBoostedTrees, GradientBoostedTreesModel

for i in range(1,30,5):
    GBTmodel = GradientBoostedTrees.trainClassifier(labelpointRdd,categoricalFeaturesInfo={}, numIterations=i)
    predictions = GBTmodel.predict(TestlabelpointRdd.map(lambda x: x.features))
    labelsAndPredictions = TestlabelpointRdd.map(lambda lp: lp.label).zip(predictions)
    print i, "錯誤率", labelsAndPredictions.filter(lambda (v, p): v != p).count() /float(TestlabelpointRdd.count())


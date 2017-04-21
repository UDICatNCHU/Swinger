import pyspark
import jieba, json, sys
conf = pyspark.SparkConf().setAll([('spark.driver.memory', '30g'), ('spark.driver.host', '172.17.0.21'), ('spark.app.id', 'local-1492693477461'), ('spark.rdd.compress', 'True'), ('spark.serializer.objectStreamReset', '100'), ('spark.master', 'local[*]'), ('spark.executor.id', 'driver'), ('spark.submit.deployMode', 'client'), ('spark.driver.port', '39274'), ('spark.app.name', 'PySparkShell')])
sc = pyspark.SparkContext(conf=conf)
stopwords = json.load(open('stopwords.json', 'r'))
def removeStopWords(sentence):
	return list(filter(lambda x: x not in stopwords, jieba.cut(sentence)))

t = sc.textFile(sys.argv[1])
result = t.map(removeStopWords).collect()
ff = open(sys.argv[2], 'w')
json.dump(result, ff)
ff.close()
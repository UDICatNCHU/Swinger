import jieba, json, sys
stopwords = json.load(open('stopwords.json', 'r'))
def removeStopWords(jiebaGen):
	return list(filter(lambda x: x not in stopwords, jieba.cut(jiebaGen)))
with open(sys.argv[1], 'r') as f:
	result = list(map(removeStopWords, f))
	ff = open('p.json', 'w')
	json.dump(result, ff)
	ff.close()
with open(sys.argv[2], 'r') as f:
	result = list(map(removeStopWords, f))
	ff = open('n.json', 'w')
	json.dump(result, ff)
	ff.close()
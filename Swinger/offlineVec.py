import json, requests, sys
from functools import reduce
from gensim import models
import numpy as np
model = models.KeyedVectors.load_word2vec_format('med400.model.bin', binary=True)
ptrain, ntrain, ptest, ntest = json.load(open(sys.argv[1], 'r')), json.load(open(sys.argv[2], 'r')), json.load(open(sys.argv[3], 'r')), json.load(open(sys.argv[4], 'r'))

ptvec = []
for i in ptrain:
	sum = np.zeros(400)
	for j in i:
		try:
			sum = np.add(sum, model[j])
		except Exception as e:
			pass
	ptvec.append(['p', sum.tolist()])
json.dump(ptvec, open('Ptvec.json', 'w'))

ntvec = []
for i in ntrain:
	sum = np.zeros(400)
	for j in i:
		try:
			sum = np.add(sum, model[j])
		except Exception as e:
			pass
	ntvec.append(['p', sum.tolist()])
json.dump(ntvec, open('Ntvec.json', 'w'))

ptestvec = []
for i in ptest:
	sum = np.zeros(400)
	for j in i:
		try:
			sum = np.add(sum, model[j])
		except Exception as e:
			pass
	ptestvec.append(['p', sum.tolist()])
json.dump(ptestvec, open('ptesttvec.json', 'w'))

ntestvec = []
for i in ntest:
	sum = np.zeros(400)
	for j in i:
		try:
			sum = np.add(sum, model[j])
		except Exception as e:
			pass
	ntestvec.append(['p', sum.tolist()])
json.dump(ntestvec, open('ntesttvec.json', 'w'))
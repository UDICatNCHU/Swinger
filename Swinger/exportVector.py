import json, requests, sys
ptrain, ntrain, ptest, ntest = json.load(open(sys.argv[1], 'r')), json.load(open(sys.argv[2], 'r')), json.load(open(sys.argv[3], 'r')), json.load(open(sys.argv[4], 'r'))

ptvec = list(map(lambda i:['p']+list(map(lambda x:requests.get('http://140.120.13.244:10000/kem/vector?keyword={}'.format(x)).json(), i)), ptrain))
json.dump(ptvec, open('ptvec.json', 'w'))

ntvec = list(map(lambda i:['n']+list(map(lambda x:requests.get('http://140.120.13.244:10000/kem/vector?keyword={}'.format(x)).json(), i)), ntrain))
json.dump(ntvec, open('ntvec.json', 'w'))

ptestvec = list(map(lambda i:['p']+list(map(lambda x:requests.get('http://140.120.13.244:10000/kem/vector?keyword={}'.format(x)).json(), i)), ptest))
json.dump(ptestvec, open('ptestvec.json', 'w'))

ntestvec = list(map(lambda i:['n']+list(map(lambda x:requests.get('http://140.120.13.244:10000/kem/vector?keyword={}'.format(x)).json(), i)), ntest))
json.dump(ntestvec, open('ntestvec.json', 'w'))

# print(['p']+list(map(lambda x:requests.get('http://140.120.13.244:10000/kem/vector?keyword={}'.format(x)).json(), ptest[0])))

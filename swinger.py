# -*- coding: utf-8 -*-
import nltk, json, pickle
import itertools
from random import shuffle
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.probability import FreqDist, ConditionalFreqDist
import sklearn

from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def bag_of_words(words):
    return dict([(word, True) for word in words])

def bigram(words, score_fn=BigramAssocMeasures.chi_sq, n=1000):
    bigram_finder = BigramCollocationFinder.from_words(words)  #把文本变成双词搭配的形式
    bigrams = bigram_finder.nbest(score_fn, n) #使用了卡方统计的方法，选择排名前1000的双词
    return bag_of_words(bigrams)

def bigram_words(words, score_fn=BigramAssocMeasures.chi_sq, n=1000):
    bigram_finder = BigramCollocationFinder.from_words(words)
    bigrams = bigram_finder.nbest(score_fn, n)
    return bag_of_words(words + bigrams)  #所有词和（信息量大的）双词搭配一起作为特征

def create_word_scores():
    posWords = json.load(open('p.json','r'))
    negWords = json.load(open('n.json','r'))
    
    posWords = list(itertools.chain(*posWords)) #把多维数组解链成一维数组
    negWords = list(itertools.chain(*negWords)) #同理

    word_fd = FreqDist() #可统计所有词的词频
    cond_word_fd = ConditionalFreqDist() #可统计积极文本中的词频和消极文本中的词频
    for word in posWords:
        word_fd[word] += 1
        cond_word_fd['pos'][word] += 1
    for word in negWords:
        word_fd[word] += 1
        cond_word_fd['neg'][word] += 1

    pos_word_count = cond_word_fd['pos'].N() #积极词的数量
    neg_word_count = cond_word_fd['neg'].N() #消极词的数量
    total_word_count = pos_word_count + neg_word_count

    word_scores = {}
    for word, freq in word_fd.items():
        pos_score = BigramAssocMeasures.chi_sq(cond_word_fd['pos'][word], (freq, pos_word_count), total_word_count) #计算积极词的卡方统计量，这里也可以计算互信息等其它统计量
        neg_score = BigramAssocMeasures.chi_sq(cond_word_fd['neg'][word], (freq, neg_word_count), total_word_count) #同理
        word_scores[word] = pos_score + neg_score #一个词的信息量等于积极卡方统计量加上消极卡方统计量

    return word_scores #包括了每个词和这个词的信息量

def create_word_bigram_scores():
    posdata = json.load(open('p.json','r'))
    negdata = json.load(open('n.json','r'))
    
    posWords = list(itertools.chain(*posdata))
    negWords = list(itertools.chain(*negdata))

    bigram_finder = BigramCollocationFinder.from_words(posWords)
    bigram_finder = BigramCollocationFinder.from_words(negWords)
    posBigrams = bigram_finder.nbest(BigramAssocMeasures.chi_sq, 5000)
    negBigrams = bigram_finder.nbest(BigramAssocMeasures.chi_sq, 5000)

    pos = posWords + posBigrams #词和双词搭配
    neg = negWords + negBigrams

    word_fd = FreqDist()
    cond_word_fd = ConditionalFreqDist()
    for word in pos:
        word_fd[word] += 1
        cond_word_fd['pos'][word] += 1
    for word in neg:
        word_fd[word] += 1
        cond_word_fd['neg'][word] += 1

    pos_word_count = cond_word_fd['pos'].N()
    neg_word_count = cond_word_fd['neg'].N()
    total_word_count = pos_word_count + neg_word_count

    word_scores = {}
    for word, freq in word_fd.items():
        pos_score = BigramAssocMeasures.chi_sq(cond_word_fd['pos'][word], (freq, pos_word_count), total_word_count)
        neg_score = BigramAssocMeasures.chi_sq(cond_word_fd['neg'][word], (freq, neg_word_count), total_word_count)
        word_scores[word] = pos_score + neg_score

    return word_scores

def find_best_words(word_scores, number):
    best_vals = sorted(word_scores.items(), key=lambda x: -x[1])[:number] #把词按信息量倒序排序。number是特征的维度，是可以不断调整直至最优的
    best_words = set([w for w, s in best_vals])
    return best_words

def score(classifier, name):
    classifier = SklearnClassifier(classifier) #在nltk 中使用scikit-learn 的接口
    classifier.train(train) #训练分类器
    pickle.dump(classifier, open(name + '.pickle','wb'))
    pred = classifier.classify_many(test) #对开发测试集的数据进行分类，给出预测的标签
    return accuracy_score(tag_test, pred) #对比分类预测结果和人工标注的正确结果，给出分类器准确度

def best_word_features(words):
    return dict([(word, True) for word in words if word in best_words])

def pos_features(feature_extraction_method):
    posFeatures = []
    for i in pos:
        posWords = [feature_extraction_method(i),'pos'] #为积极文本赋予"pos"
        posFeatures.append(posWords)
    return posFeatures
def neg_features(feature_extraction_method):
    negFeatures = []
    for j in neg:
        negWords = [feature_extraction_method(j),'neg'] #为消极文本赋予"neg"
        negFeatures.append(negWords)
    return negFeatures

pos_review = json.load(open('p.json','r'))
neg_review = json.load(open('n.json','r'))
word_scores_1 = create_word_scores()

word_scores_2 = create_word_bigram_scores()




shuffle(pos_review) #把积极文本的排列随机化
pos = pos_review
neg = neg_review


posFeatures = pos_features(bag_of_words) #使用所有词作为特征
negFeatures = neg_features(bag_of_words)

train = posFeatures+negFeatures
# train = posFeatures[174:]+negFeatures[174:]
# devtest = posFeatures[124:174]+negFeatures[124:174]
test = posFeatures+negFeatures
test, tag_test = zip(*test)

# dev, tag_dev = zip(*devtest) #把开发测试集（已经经过特征化和赋予标签了）分为数据和标签

print('BernoulliNB`s accuracy is %f' %score(BernoulliNB(), 'BernoulliNB'))
print('MultinomiaNB`s accuracy is %f' %score(MultinomialNB(), 'MultinomialNB'))
print('LogisticRegression`s accuracy is %f' %score(LogisticRegression(), 'LogisticRegression'))
print('SVC`s accuracy is %f' %score(SVC(), 'SVC'))
print('LinearSVC`s accuracy is %f' %score(LinearSVC(), 'LinearSVC'))
print('NuSVC`s accuracy is %f' %score(NuSVC(), 'NuSVC'))
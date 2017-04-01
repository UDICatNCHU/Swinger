# -*- coding: utf-8 -*-
import nltk, json, pickle, sys, itertools,collections, jieba, os
from random import shuffle
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.probability import FreqDist, ConditionalFreqDist
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from nltk.metrics.scores import (accuracy, precision, recall, f_measure, log_likelihood, approxrand)


class Swinger(object):
    """docstring for Swinger"""
    BASEDIR = os.path.dirname(__file__)
    stopwords = json.load(open(os.path.join(BASEDIR, 'stopwords.json'), 'r'))
    classifier_table = {
        'SVC':SVC(probability=False),
        'LinearSVC':LinearSVC(),
        'NuSVC':NuSVC(probability=False),
        'MultinomialNB':MultinomialNB(),
        'BernoulliNB':BernoulliNB()
    }
    
    def __init__(self):
        self.train = []
        self.test = ''
        self.classifier = ''

    def load(self, model, useDefault=True, pos=None, neg=None, BestFeatureVec=2000):
        self.BestFeatureVec = int(BestFeatureVec)

        if useDefault:
            print('load default mainFeatures')
            self.mainFeatures = pickle.load(open(os.path.join(self.BASEDIR, 'mainFeatures.pickle'), 'rb'))
            print('load default mainFeatures success!!')

            self.classifier = pickle.load(open(os.path.join(self.BASEDIR, '{}-{}.pickle'.format(model, self.BestFeatureVec)), 'rb'))
            print("load model from {}".format(model))
        else:
            try:
                print('load local mainFeatures')
                self.mainFeatures = pickle.load(open('mainFeatures.pickle'), 'rb')
                print('load local mainFeatures success!!')

                self.classifier = pickle.load(open('{}-{}.pickle'.format(model, self.BestFeatureVec)), 'rb')
                print("load model from {}".format(model))
            except Exception as e:
                print('load mainFeatures failed!!')
                print('start creating mainFeatures...')
                self.pos_review = json.load(open(pos, 'r'))
                self.neg_review = json.load(open(neg, 'r'))
                self.mainFeatures = self.create_word_features(bigram=True, pos_data=self.pos_review, neg_data=self.neg_review) #使用词和双词搭配作为特征
                pickle.dump(self.mainFeatures, open('mainFeatures.pickle', 'wb'))

                print('start building {} model!!!'.format(model))
                posFeatures = self.emotion_features(self.best_word_features, self.pos_review, 'pos')
                negFeatures = self.emotion_features(self.best_word_features, self.neg_review, 'neg')
                self.train = posFeatures + negFeatures
                self.classifier = SklearnClassifier(self.classifier_table[model]) #在nltk 中使用scikit-learn 的接口
                self.classifier.train(self.train) #训练分类器
                pickle.dump(self.classifier, open('{}-{}.pickle'.format(model, self.BestFeatureVec),'wb'))

    def buildTestData(self, pos_test, neg_test):
        pos_test = json.load(open(pos_test, 'r'))
        neg_test = json.load(open(neg_test, 'r'))
        posFeatures = self.emotion_features(self.best_word_features, pos_test, 'pos')
        negFeatures = self.emotion_features(self.best_word_features, neg_test, 'neg')
        return posFeatures + negFeatures

    def best_word_features(self, word_list):
        def find_best_words(number):
            best_vals = sorted(self.mainFeatures.items(), key=lambda x: -x[1])[:number] #把词按信息量倒序排序。number是特征的维度，是可以不断调整直至最优的
            return set(w for w, s in best_vals) # set comprehension

        best_words = find_best_words(self.BestFeatureVec) #特征维度1500
        return {word:True for word in word_list if word in best_words}

    @staticmethod
    def create_word_features(bigram, pos_data, neg_data):
        posWords = list(itertools.chain(*pos_data)) #把多维数组解链成一维数组
        negWords = list(itertools.chain(*neg_data)) #同理

        if bigram == True:
                bigram_finder = BigramCollocationFinder.from_words(posWords)
                posBigrams = bigram_finder.nbest(BigramAssocMeasures.chi_sq, 5000)
                bigram_finder = BigramCollocationFinder.from_words(negWords)
                negBigrams = bigram_finder.nbest(BigramAssocMeasures.chi_sq, 5000)
                posWords += posBigrams #词和双词搭配
                negWords += negBigrams

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

        word_features = {}
        for word, freq in word_fd.items():
            pos_score = BigramAssocMeasures.chi_sq(cond_word_fd['pos'][word], (freq, pos_word_count), total_word_count) #计算积极词的卡方统计量，这里也可以计算互信息等其它统计量
            neg_score = BigramAssocMeasures.chi_sq(cond_word_fd['neg'][word], (freq, neg_word_count), total_word_count) #同理
            word_features[word] = pos_score + neg_score #一个词的信息量等于积极卡方统计量加上消极卡方统计量

        return word_features #包括了每个词和这个词的信息量

    def score(self, pos_test, neg_test):
        # build test data set
        self.test = self.buildTestData(pos_test, neg_test)

        refsets = collections.defaultdict(set)
        testsets = collections.defaultdict(set)
        for i, (feats, label) in enumerate(self.test):
            refsets[label].add(i)
            observed = self.classifier.classify(feats)
            testsets[observed].add(i)

        # pred = classifier.classify_many(self.test) #对开发测试集的数据进行分类，给出预测的标签
        # print(accuracy_score(self.tag_test, pred)) #对比分类预测结果和人工标注的正确结果，给出分类器准确度
        print('pos precision:', precision(refsets['pos'], testsets['pos']))
        print('pos recall:', recall(refsets['pos'], testsets['pos']))
        print('pos F-measure:', f_measure(refsets['pos'], testsets['pos']))
        print('neg precision:',precision(refsets['neg'], testsets['neg']))
        print('neg recall:',recall(refsets['neg'], testsets['neg']))
        print('neg F-measure:',f_measure(refsets['neg'], testsets['neg']))

    def emotion_features(self, feature_extraction_method, data, emo):
        return list(map(lambda x:[feature_extraction_method(x), emo], data)) #为积极文本赋予"pos"

    def swing(self, word_list):
        word_list = filter(lambda x: x not in self.stopwords, jieba.cut(word_list))
        sentence = self.best_word_features(word_list)
        return self.classifier.classify(sentence)

if __name__ == '__main__':
    s = Swinger()
    s.load('NuSVC')
    s.score(pos_test=sys.argv[1], neg_test=sys.argv[2])
    s.load('SVC')
    s.score(pos_test=sys.argv[1], neg_test=sys.argv[2])
    s.load('LinearSVC')
    s.score(pos_test=sys.argv[1], neg_test=sys.argv[2])
    s.load('MultinomialNB')
    s.score(pos_test=sys.argv[1], neg_test=sys.argv[2])
    s.load('BernoulliNB')
    s.score(pos_test=sys.argv[1], neg_test=sys.argv[2])

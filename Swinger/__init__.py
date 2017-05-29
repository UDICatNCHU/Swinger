# -*- coding: utf-8 -*-
import nltk, json, pickle, sys, collections, jieba, os
from random import shuffle
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from nltk.metrics.scores import (accuracy, precision, recall, f_measure, log_likelihood, approxrand)
from Swinger.utils import create_Mainfeatures, CutAndrmStopWords


class Swinger(object):
    """docstring for Swinger"""
    BASEDIR = os.path.dirname(__file__)
    classifier_table = {
        'SVC':SVC(probability=False),
        'LinearSVC':LinearSVC(),
        'NuSVC':NuSVC(probability=False),
        'MultinomialNB':MultinomialNB(),
        'BernoulliNB':BernoulliNB(),
        'LogisticRegression':LogisticRegression()
    }
    
    def __init__(self):
        self.train = []
        self.test = []
        self.classifier = ''

    def load(self, model, useDefault=True, pos=None, neg=None, BestFeatureVec=700):
        BestFeatureVec = int(BestFeatureVec)

        if useDefault:
            print('load default bestMainFeatures')
            self.bestMainFeatures = pickle.load(open(os.path.join(self.BASEDIR, 'bestMainFeatures.pickle.{}'.format(BestFeatureVec)), 'rb'))
            print('load default bestMainFeatures success!!')

            self.classifier = pickle.load(open(os.path.join(self.BASEDIR, '{}.pickle.{}'.format(model, BestFeatureVec)), 'rb'))
            print("load model from {}".format(model))
        else:
            try:
                print('load local bestMainFeatures')
                self.bestMainFeatures = pickle.load(open('bestMainFeatures.pickle.{}'.format(BestFeatureVec), 'rb'))
                print('load local bestMainFeatures success!!')

                self.classifier = pickle.load(open('{}.pickle.{}'.format(model, BestFeatureVec), 'rb'))
                print("load model from {}".format(model))
            except Exception as e:
                # build best features.
                print('load bestMainFeatures failed!!\nstart creating bestMainFeatures ...')

                self.pos_origin = json.load(open(pos, 'r'))
                self.neg_origin = json.load(open(neg, 'r'))
                shuffle(self.pos_origin)
                shuffle(self.neg_origin)
                poslen = len(self.pos_origin)
                neglen = len(self.neg_origin)

                # build train and test data.
                self.pos_review = self.pos_origin[:int(poslen*0.9)]
                self.pos_test = self.pos_origin[int(poslen*0.9):]
                self.neg_review = self.neg_origin[:int(neglen*0.9)]
                self.neg_test = self.neg_origin[int(neglen*0.9):]

                self.bestMainFeatures = create_Mainfeatures(pos_data=self.pos_review, neg_data=self.neg_review, BestFeatureVec=BestFeatureVec) # 使用詞和雙詞搭配作為特徵

                # build model
                print('start building {} model!!!'.format(model))

                self.classifier = SklearnClassifier(self.classifier_table[model]) #nltk在sklearn的接口
                if len(self.train) == 0:
                    print('build training data')
                    posFeatures = self.emotion_features(self.best_Mainfeatures, self.pos_review, 'pos')
                    negFeatures = self.emotion_features(self.best_Mainfeatures, self.neg_review, 'neg')
                    self.train = posFeatures + negFeatures
                self.classifier.train(self.train) #訓練分類器
                pickle.dump(self.classifier, open('{}.pickle.{}'.format(model, BestFeatureVec),'wb'))

    def buildTestData(self, pos_test, neg_test):
        pos_test = json.load(open(pos_test, 'r'))
        neg_test = json.load(open(neg_test, 'r'))
        posFeatures = self.emotion_features(self.best_Mainfeatures, pos_test, 'pos')
        negFeatures = self.emotion_features(self.best_Mainfeatures, neg_test, 'neg')
        return posFeatures + negFeatures

    def best_Mainfeatures(self, word_list):
        return {word:True for word in word_list if word in self.bestMainFeatures}

    def score(self, pos_test, neg_test):
        from sklearn.metrics import precision_recall_curve
        from sklearn.metrics import roc_curve
        from sklearn.metrics import auc
        # build test data set
        if len(self.test) == 0:
            # self.test = self.buildTestData(self.pos_test, self.neg_test)
            self.test = self.buildTestData(pos_test, neg_test)

        test, test_tag = zip(*self.test)
        pred = list(map(lambda x:1 if x=='pos' else 0, self.classifier.classify_many(test))) #對開發測試集的數據進行分類，給出預測的標籤
        tag = list(map(lambda x:1 if x=='pos' else 0, test_tag))
        # ROC AUC
        fpr, tpr, _ = roc_curve(tag, pred, pos_label=1)
        print("ROC AUC: %.2f" % auc(fpr, tpr))
        return auc(fpr, tpr)

    def emotion_features(self, feature_extraction_method, data, emo):
        return list(map(lambda x:[feature_extraction_method(x), emo], data)) #爲積極文本賦予"pos"

    def swing(self, sentence):
        sentence = self.best_Mainfeatures(CutAndrmStopWords(sentence))
        return self.classifier.classify(sentence)

    def swingList(self, sentenceList):
        sentence = self.best_Mainfeatures(sentenceList)
        return self.classifier.classify(sentence)

if __name__ == '__main__':
    # import matplotlib.pyplot as plt #可視化模塊
    NuSVC_arr=[]
    SVC_arr=[]
    LinearSVC_arr=[]
    MultinomialNB_arr=[]
    BernoulliNB_arr=[]
    LogisticRegression_arr=[]
    for i in range(100, 700, 50):
        s = Swinger()
        # s.load('NuSVC', useDefault=False, pos=sys.argv[1], neg=sys.argv[2], BestFeatureVec=i)
        # NuSVC_arr.append(s.score(pos_test=sys.argv[3], neg_test=sys.argv[4]))

        # s.load('SVC', useDefault=False, pos=sys.argv[1], neg=sys.argv[2], BestFeatureVec=i)
        # SVC_arr.append(s.score(pos_test=sys.argv[3], neg_test=sys.argv[4]))

        # s.load('LinearSVC', useDefault=False, pos=sys.argv[1], neg=sys.argv[2], BestFeatureVec=i)
        # LinearSVC_arr.append(s.score(pos_test=sys.argv[3], neg_test=sys.argv[4]))

        # s.load('MultinomialNB', useDefault=False, pos=sys.argv[1], neg=sys.argv[2], BestFeatureVec=i)
        # MultinomialNB_arr.append(s.score(pos_test=sys.argv[3], neg_test=sys.argv[4]))

        # s.load('BernoulliNB', useDefault=False, pos=sys.argv[1], neg=sys.argv[2], BestFeatureVec=i)
        # BernoulliNB_arr.append(s.score(pos_test=sys.argv[3], neg_test=sys.argv[4]))

        s.load('LogisticRegression', useDefault=False, pos=sys.argv[1], neg=sys.argv[2], BestFeatureVec=i)
        LogisticRegression_arr.append(s.score(pos_test=sys.argv[3], neg_test=sys.argv[4]))

    print(NuSVC_arr)
    print(SVC_arr)
    print(LinearSVC_arr)
    print(MultinomialNB_arr)
    print(BernoulliNB_arr)
    print(LogisticRegression_arr)
    # plt.plot(range(50,2000, 50), NuSVC_arr, 'o-', color="b",label="NuSVC")
    # plt.plot(range(50,2000, 50), SVC_arr, 'o-', color="g",label="SVC")
    # plt.plot(range(50,2000, 50), LinearSVC_arr, 'o-', color="r",label="LinearSVC")
    # plt.plot(range(50,2000, 50), MultinomialNB_arr, 'o-', color="c",label="MultinomialNB")
    # plt.plot(range(50,2000, 50), BernoulliNB_arr, 'o-', color="m",label="BernoulliNB")
    # plt.plot(range(50,2000, 50), LogisticRegression_arr, 'o-', color="y",label="LogisticRegression")
    # plt.legend(loc='best')
    # plt.xlabel("features vectors")
    # plt.ylabel("AUC")
    # plt.savefig(sys.argv[5]+'.png')
    # plt.show()

    # s = Swinger()
    # s.load('NuSVC', useDefault=False, pos=sys.argv[1], neg=sys.argv[2], BestFeatureVec=sys.argv[5])
    # NuSVC_arr.append(s.score(pos_test=sys.argv[3], neg_test=sys.argv[4]))

    # s.load('SVC', useDefault=False, pos=sys.argv[1], neg=sys.argv[2], BestFeatureVec=sys.argv[5])
    # SVC_arr.append(s.score(pos_test=sys.argv[3], neg_test=sys.argv[4]))

    # s.load('LinearSVC', useDefault=False, pos=sys.argv[1], neg=sys.argv[2], BestFeatureVec=sys.argv[5])
    # LinearSVC_arr.append(s.score(pos_test=sys.argv[3], neg_test=sys.argv[4]))

    # s.load('MultinomialNB', useDefault=False, pos=sys.argv[1], neg=sys.argv[2], BestFeatureVec=sys.argv[5])
    # MultinomialNB_arr.append(s.score(pos_test=sys.argv[3], neg_test=sys.argv[4]))

    # s.load('BernoulliNB', useDefault=False, pos=sys.argv[1], neg=sys.argv[2], BestFeatureVec=sys.argv[5])
    # BernoulliNB_arr.append(s.score(pos_test=sys.argv[3], neg_test=sys.argv[4])) 

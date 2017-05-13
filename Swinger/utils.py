import itertools, pickle, json, sys
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.probability import FreqDist, ConditionalFreqDist

def create_Mainfeatures(pos_data, neg_data, BestFeatureVec):
    posWords = list(itertools.chain(*pos_data)) #把多為數組解煉成一維數組
    negWords = list(itertools.chain(*neg_data)) #同理

    # bigram
    bigram_finder = BigramCollocationFinder.from_words(posWords)
    posBigrams = bigram_finder.nbest(BigramAssocMeasures.chi_sq, 5000)
    bigram_finder = BigramCollocationFinder.from_words(negWords)
    negBigrams = bigram_finder.nbest(BigramAssocMeasures.chi_sq, 5000)
    posWords += posBigrams #詞和雙詞搭配
    negWords += negBigrams

    word_fd = FreqDist() #可統計所有詞的詞頻
    cond_word_fd = ConditionalFreqDist() #可統計積極文本中的詞頻和消極文本中的詞頻
    for word in posWords:
        word_fd[word] += 1
        cond_word_fd['pos'][word] += 1
    for word in negWords:
        word_fd[word] += 1
        cond_word_fd['neg'][word] += 1

    pos_word_count = cond_word_fd['pos'].N() #積極詞的數量
    neg_word_count = cond_word_fd['neg'].N() #消極詞的數量
    total_word_count = pos_word_count + neg_word_count

    word_features = {}
    for word, freq in word_fd.items():
        pos_score = BigramAssocMeasures.chi_sq(cond_word_fd['pos'][word], (freq, pos_word_count), total_word_count) #計算積極詞的卡方統計量，這裏也可以計算互信息等其它統計量
        neg_score = BigramAssocMeasures.chi_sq(cond_word_fd['neg'][word], (freq, neg_word_count), total_word_count) #同理
        word_features[word] = pos_score + neg_score

    def find_best_words(number):
        best = sorted(word_features.items(), key=lambda x: -x[1])[:number] # 把詞按信息量倒序排序。number 是特徵的微度，式可以不斷調整至最優的
        return set(w for w, s in best)

    best = find_best_words(BestFeatureVec)
    pickle.dump(best, open('bestMainFeatures.pickle.{}'.format(BestFeatureVec), 'wb'))
    return best

if __name__ == '__main__':
    create_Mainfeatures(sys.argv[1], sys.argv[2], int(sys.argv[3]))
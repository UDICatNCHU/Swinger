# Swinger

一個自製的判斷中文情緒的函式庫，因為算出來的機率會在0~1之間搖擺，故命名搖擺者

## Getting Started

#### Prerequisities

1. OS：Ubuntu / OSX would be nice
2. environment：need python3 `sudo apt-get update; sudo apt-get install; python3 python3-dev`

#### Installing

1. 使用虛擬環境 Use virtualenv is recommended：
  1. `virtualenv venv`
2. 啟動方法 How to activate virtualenv
  1. for Linux：`. venv/bin/activate`
  2. for Windows：`venv\Scripts\activate`
3. 安裝 Install：`pip install Swinger`

## Running & Testing

#### Method

* `load`：讀取或是建立情緒分析的模型。有就會直接讀取，不會重新建立。
* `score`：計算此模型的precision, recal等數據。
* `swing`：判斷一句話的情緒是正面或反面。

#### Model

目前提供的分類器有：  
* SVC
* NuSVC
* LinearSVC
* MultinomialNB
* BernoulliNB

#### Run

1. 訓練資料來源：使用我們自製的 `公開訓練資料` -> [Open-Sentiment-Training-Data](https://github.com/UDICatNCHU/Open-Sentiment-Training-Data)
  * 按照 `Open-Sentiment-Training-Data` 的README執行  
  `text2json.py` 會產生出斷好詞的json檔  
  此格式即為`Swinger`的input data。
2. 訓練模型：`python __init__.py posTrain negTrain posTest negTest`
  1. 先準備好訓練資料及測試資料
  2. 訓練出NuSVC的模型：
    ```
    from Swinger import Swinger
    s = Swinger(pos=正面情緒訓練資料, neg=負面情緒訓練資料, BestFeatureVec=選取的特徵數)
    s.load('NuSVC') # 以NuSVC建立model
    s.score(pos_test=正面測試資料, neg_test=負面測試資料) #
    ```

3. 測試效果：  
  1. 先準備好要測試的文集
  2. 執行下列程式碼
  ```
  from Swinger import Swinger
  s = Swinger(pos=POS的訓練資料, neg=NEG的訓練資料)
  s.load('NuSVC') # 可以是任意的分類器名稱
  s.swing(要測試的字串)
  ```

## Unit tests

not yet.

## Built With

* nltk
* sklearn
* jieba
* numpy
* scipy

## Contributors

* **張泰瑋** [david](https://github.com/david30907d)
* **黃翔宇**

## License

This package use `GPL3.0` License.

## Acknowledgments

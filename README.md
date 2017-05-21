# Swinger

一個自製的判斷中文情緒的函式庫，因為算出來的機率會在0~1之間搖擺，故命名搖擺者  
可透過`pip`安裝 內含已經訓練好的模型

![swinger](img/swing.svg)

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
* `score`：計算此模型的area under ROC本專案因為是二元分類，所以使用這個當作benchmark。
* `swing`：判斷一句話的情緒是正面或反面。

#### Model

目前提供的分類器有：  
* LogisticRegression
* MultinomialNB
不同情況下準確度會有些差異，但實驗出來的auc都有0.76左右

#### Run

1. 訓練資料來源：使用我們自製的 `公開訓練資料` -> [Open-Sentiment-Training-Data](https://github.com/UDICatNCHU/Open-Sentiment-Training-Data)
    * 按照 `Open-Sentiment-Training-Data` 的README執行  
  `text2json.py` 會產生出斷好詞的json檔  
  此格式即為`Swinger`的input data。
    * 訓練資料:PTT黑特版+好人版等等
    * 測試資料:蔡英文粉專
2. 訓練模型：
    * 先準備好訓練資料及測試資料
    * 訓練出指定分類器的模型：
    ```
    from Swinger import Swinger
    s = Swinger()
    s.load('LogisticRegression', useDefault=False, pos=正面情緒訓練資料, neg=負面情緒訓練資料, BestFeatureVec=選取的特徵數) # 以LogisticRegression建立model
    s.score(pos_test=正面測試資料, neg_test=負面測試資料)
    ```
3. 測試效果：  
    * 先準備好要測試的文集
    * 執行下列程式碼：
    ```
    from Swinger import Swinger
    s = Swinger()
    s.load('LogisticRegression') # 或是其他模型例如MultinomialNB
    s.swing('齊家治國平天下，小家給治了！國家更需要妳，加油!擇善固執莫在意全家滿意，至於她家謾駡攻許隨她去(正常情緒紓緩)，革命尚未成功期盼繼續努力') # 結果為pos，正面情緒
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

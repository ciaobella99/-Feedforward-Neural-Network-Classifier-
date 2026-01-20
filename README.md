# -Feedforward-Neural-Network-Classifier-
本專案實作一個 **前饋式神經網路（Feedforward Neural Network, FNN）分類系統**，  
針對給定的結構化資料進行訓練，並完成模型推論、結果輸出與完整技術文件整理。

---

## 專案目標

- 建立一個前饋式神經網路進行多類別分類
- 使用已標註資料進行模型訓練
- 對未知資料進行推論並輸出分類結果
- 將模型與實驗流程完整保存，確保可重現性

---

## 專案結構

```
.
├── clsn1_trn.csv            # 訓練資料集（最後一欄為類別標籤）
├── clsn1_tst.csv            # 測試資料集
├── clsn1_ans.csv            # 測試資料和預測分類結果
│
├── report1.h5               # 訓練完成的神經網路模型
│
├── train_classifier.py      # source code
└── README.md                # 專案說明文件
```

---

## 方法概述（Methodology）

### 資料處理
- 輸入資料為數值型特徵
- 類別標籤位於每筆資料的最後一個欄位
- 訓練與測試資料分開處理，避免資料洩漏

### 模型架構
- 模型類型：Fully Connected Neural Network
- 隱藏層：多層 Dense Layers
- 啟動函數：ReLU（隱藏層）
- 輸出層：Softmax（多類別分類）

### 訓練設定
- 損失函數：Categorical Cross-Entropy
- 最佳化方法：Adam Optimizer

---

## 使用方式

```bash
python train_classifier.py
```

執行後將完成：
1. 載入並前處理訓練資料
2. 建立並訓練分類模型
3. 使用測試資料進行推論
4. 輸出分類結果至 `clsn1_ans.csv`
5. 儲存模型至 `report1.h5`

---

## 輸出結果說明

### `clsn1_ans.csv`
- 包含原始測試資料
- 以及模型對應的分類預測結果
- 每一列對應一筆測試樣本
### 1) 模型訓練準確率（Figure 1）
![訓練結果可視化](images/accuracy.png)

**說明：**
- 本圖展示訓練集（Training Set）與驗證集（Validation Set）在不同訓練 epoch 下的性能變化情況。

### 2) 模型訓練Loss（Figure 2）
![訓練結果可視化](images/loss.png)

**說明：**
- 訓練與驗證資料集在訓練過程中的損失函數收斂情形

---

## 環境需求

- Python 3.x
- TensorFlow / Keras
- NumPy
- Pandas

---

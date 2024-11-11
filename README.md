# Deep_learning_models_for_Dogecoin_price_prediction

# Main_doge.ipynb - 전체 코드 설명

### 마크다운 셀 설명

# 캔들차트 데이터 생성


### 마크다운 셀 설명

참고 : https://dataplay.tistory.com/37?category=845492

### 마크다운 셀 설명

참고코드 : https://colab.research.google.com/drive/1WXG3cohwO6_0mbmB9CdT37cc1jfE2Zon

### 코드 셀 설명

`pip install mpl_finance` - 여기에 이 줄에 대한 설명을 추가합니다.

### 코드 셀 설명

`import os` - 여기에 이 줄에 대한 설명을 추가합니다.

`import pandas as pd` - 여기에 이 줄에 대한 설명을 추가합니다.

`import numpy as np` - 여기에 이 줄에 대한 설명을 추가합니다.

`import matplotlib.pyplot as plt` - 여기에 이 줄에 대한 설명을 추가합니다.

`from pathlib import Path` - 여기에 이 줄에 대한 설명을 추가합니다.

`from shutil import copyfile, move` - 여기에 이 줄에 대한 설명을 추가합니다.

`from mpl_finance import candlestick2_ochl` - 여기에 이 줄에 대한 설명을 추가합니다.

### 마크다운 셀 설명

도지 코인 파일 불러오기


### 코드 셀 설명

`from google.colab import drive` - 여기에 이 줄에 대한 설명을 추가합니다.

`drive.mount('/content/drive/')` - 여기에 이 줄에 대한 설명을 추가합니다.

### 코드 셀 설명

`import os` - 여기에 이 줄에 대한 설명을 추가합니다.

`currentPath = os.getcwd()` - 여기에 이 줄에 대한 설명을 추가합니다.

`#change path` - 여기에 이 줄에 대한 설명을 추가합니다.

`os.chdir('/content/drive/MyDrive')` - 여기에 이 줄에 대한 설명을 추가합니다.

### 코드 셀 설명

`coinbase = pd.read_csv("out_KRW-DOGE_min_1_2021-05-21.csv")` - 여기에 이 줄에 대한 설명을 추가합니다.

`cb_index = coinbase.index.ravel() # 메모리에서 발생하는 순서대로 인덱싱하여 평평하게 배열` - 여기에 이 줄에 대한 설명을 추가합니다.

`coinbase` - 여기에 이 줄에 대한 설명을 추가합니다.

### 코드 셀 설명

`coinbase.isna().sum()` - 여기에 이 줄에 대한 설명을 추가합니다.

### 코드 셀 설명

`data = coinbase.fillna(method="backfill") # 결측치를 뒷방향으로 채워나가는 것` - 여기에 이 줄에 대한 설명을 추가합니다.

### 코드 셀 설명

`data = data.drop(columns=['code', 'candleDateTime', 'candleDateTimeKst', 'candleAccTradeVolume','candleAccTradePrice','timestamp','unit'], axis=1)` - 여기에 이 줄에 대한 설명을 추가합니다.

### 코드 셀 설명

`data = data.reset_index(drop=True)` - 여기에 이 줄에 대한 설명을 추가합니다.

### 코드 셀 설명

`data` - 여기에 이 줄에 대한 설명을 추가합니다.

### 코드 셀 설명

`plt.plot(data['tradePrice'])` - 여기에 이 줄에 대한 설명을 추가합니다.

### 코드 셀 설명

`def ohlc2cs2(data, seq_len, dimension):` - 여기에 이 줄에 대한 설명을 추가합니다.

`    # python preprocess.py -m ohlc2cs -l 20 -i stockdatas/EWT_testing.csv -t testing` - 여기에 이 줄에 대한 설명을 추가합니다.

`    print("Converting olhc to candlestick")` - 여기에 이 줄에 대한 설명을 추가합니다.

`    df = data` - 여기에 이 줄에 대한 설명을 추가합니다.

`    plt.style.use('dark_background')` - 여기에 이 줄에 대한 설명을 추가합니다.

`    figs = np.zeros((len(df)-1, dimension, dimension, 3))` - 여기에 이 줄에 대한 설명을 추가합니다.

`    labels = []` - 여기에 이 줄에 대한 설명을 추가합니다.

`    for i in range(0, len(df)-1):` - 여기에 이 줄에 대한 설명을 추가합니다.

`        # ohlc+volume` - 여기에 이 줄에 대한 설명을 추가합니다.

`        c = df.loc[i:i + int(seq_len) - 1, :]` - 여기에 이 줄에 대한 설명을 추가합니다.

`        c_ = df.loc[i:i + int(seq_len), :]` - 여기에 이 줄에 대한 설명을 추가합니다.

`        if len(c) == int(seq_len):` - 여기에 이 줄에 대한 설명을 추가합니다.

`            my_dpi = 96` - 여기에 이 줄에 대한 설명을 추가합니다.

`            fig = plt.figure(figsize=(dimension / my_dpi,` - 여기에 이 줄에 대한 설명을 추가합니다.

`                                      dimension / my_dpi), dpi=my_dpi)` - 여기에 이 줄에 대한 설명을 추가합니다.

`            ax1 = fig.add_subplot(1, 1, 1)` - 여기에 이 줄에 대한 설명을 추가합니다.

`            candlestick2_ochl(ax1, c['openingPrice'], c['tradePrice'], c['highPrice'],` - 여기에 이 줄에 대한 설명을 추가합니다.

`                              c['lowPrice'], width=1,` - 여기에 이 줄에 대한 설명을 추가합니다.

`                              colorup='#77d879', colordown='#db3f3f')` - 여기에 이 줄에 대한 설명을 추가합니다.

`            ax1.grid(False)` - 여기에 이 줄에 대한 설명을 추가합니다.

`            ax1.set_xticklabels([])` - 여기에 이 줄에 대한 설명을 추가합니다.

`            ax1.set_yticklabels([])` - 여기에 이 줄에 대한 설명을 추가합니다.

`            ax1.xaxis.set_visible(False)` - 여기에 이 줄에 대한 설명을 추가합니다.

`            ax1.yaxis.set_visible(False)` - 여기에 이 줄에 대한 설명을 추가합니다.

`            ax1.axis('off')` - 여기에 이 줄에 대한 설명을 추가합니다.

`            # create the second axis for the volume bar-plot` - 여기에 이 줄에 대한 설명을 추가합니다.

`            # Add a seconds axis for the volume overlay` - 여기에 이 줄에 대한 설명을 추가합니다.

`        starting = c_["tradePrice"].iloc[-2]` - 여기에 이 줄에 대한 설명을 추가합니다.

`        endvalue = c_["tradePrice"].iloc[-1]` - 여기에 이 줄에 대한 설명을 추가합니다.

`        if endvalue > starting :` - 여기에 이 줄에 대한 설명을 추가합니다.

`            label = 1` - 여기에 이 줄에 대한 설명을 추가합니다.

`        else :` - 여기에 이 줄에 대한 설명을 추가합니다.

`            label = 0` - 여기에 이 줄에 대한 설명을 추가합니다.

`        labels.append(label)` - 여기에 이 줄에 대한 설명을 추가합니다.

`        fig.canvas.draw()` - 여기에 이 줄에 대한 설명을 추가합니다.

`        fig_np = np.array(fig.canvas.renderer._renderer)` - 여기에 이 줄에 대한 설명을 추가합니다.

`        figs[i] = fig_np[:,:,:3]` - 여기에 이 줄에 대한 설명을 추가합니다.

`        plt.close(fig)` - 여기에 이 줄에 대한 설명을 추가합니다.

`        # normal length - end` - 여기에 이 줄에 대한 설명을 추가합니다.

`    print("Converting olhc to candlestik finished.")` - 여기에 이 줄에 대한 설명을 추가합니다.

`    return figs, labels` - 여기에 이 줄에 대한 설명을 추가합니다.

### 코드 셀 설명

`inputs = data` - 여기에 이 줄에 대한 설명을 추가합니다.

`seq_len = 30` - 여기에 이 줄에 대한 설명을 추가합니다.

`dimension = 48` - 여기에 이 줄에 대한 설명을 추가합니다.

`figures, labels = ohlc2cs2(inputs, seq_len, dimension)` - 여기에 이 줄에 대한 설명을 추가합니다.

### 코드 셀 설명

`#위 함수로 생성된 figures는 값의 범위가 0~255 이기 때문에 0~1로 맞춰주기 위해 255로 나눕니다.` - 여기에 이 줄에 대한 설명을 추가합니다.

`figures = figures/255.0` - 여기에 이 줄에 대한 설명을 추가합니다.

`print(np.shape(labels), np.shape(figures))` - 여기에 이 줄에 대한 설명을 추가합니다.

### 마크다운 셀 설명

데이터 제너레이팅

### 코드 셀 설명

`def single_stock_generator(chart, labels, batch_size) :` - 여기에 이 줄에 대한 설명을 추가합니다.

`    #output [chart, labels]` - 여기에 이 줄에 대한 설명을 추가합니다.

`    while True :` - 여기에 이 줄에 대한 설명을 추가합니다.

`        stock_batch = np.zeros(shape=(batch_size, dimension, dimension, 3))` - 여기에 이 줄에 대한 설명을 추가합니다.

`        label_batch = np.zeros(shape=(batch_size, ))` - 여기에 이 줄에 대한 설명을 추가합니다.

`        for i in range(batch_size) :` - 여기에 이 줄에 대한 설명을 추가합니다.

`            idx = np.random.randint(len(labels))` - 여기에 이 줄에 대한 설명을 추가합니다.

`            stock_batch[i] = chart[idx]` - 여기에 이 줄에 대한 설명을 추가합니다.

`            label_batch[i] = labels[idx]` - 여기에 이 줄에 대한 설명을 추가합니다.

`        yield stock_batch, label_batch` - 여기에 이 줄에 대한 설명을 추가합니다.

### 코드 셀 설명

`train_len = 6753` - 여기에 이 줄에 대한 설명을 추가합니다.

`batch_size = 16` - 여기에 이 줄에 대한 설명을 추가합니다.

`train_gen = single_stock_generator(figures[:train_len], labels[:train_len], batch_size)` - 여기에 이 줄에 대한 설명을 추가합니다.

`test_gen = single_stock_generator(figures[train_len:], labels[train_len:], batch_size)` - 여기에 이 줄에 대한 설명을 추가합니다.

### 코드 셀 설명

`tmp_data = next(train_gen)` - 여기에 이 줄에 대한 설명을 추가합니다.

`print("Chart image shape : ",np.shape(tmp_data[0]))` - 여기에 이 줄에 대한 설명을 추가합니다.

`print("Label shape :",np.shape(tmp_data[1]))` - 여기에 이 줄에 대한 설명을 추가합니다.

### 코드 셀 설명

`# 만들어진 차트 이미지 중 하나를 예시로 그려보겠습니다.` - 여기에 이 줄에 대한 설명을 추가합니다.

`import matplotlib as mpl` - 여기에 이 줄에 대한 설명을 추가합니다.

`import matplotlib.pylab as plt` - 여기에 이 줄에 대한 설명을 추가합니다.

`%matplotlib inline` - 여기에 이 줄에 대한 설명을 추가합니다.

### 코드 셀 설명

`len(tmp_data[0][0][:,:,:])` - 여기에 이 줄에 대한 설명을 추가합니다.

### 코드 셀 설명

`plt.figure()` - 여기에 이 줄에 대한 설명을 추가합니다.

`plt.imshow(tmp_data[0][15][:,:,:])` - 여기에 이 줄에 대한 설명을 추가합니다.

`plt.show()` - 여기에 이 줄에 대한 설명을 추가합니다.

### 마크다운 셀 설명

# 모듈, 라이브러리 설치

### 코드 셀 설명

`# Keras의 Functional APi를 이용할 거라서 불러와줍니다.` - 여기에 이 줄에 대한 설명을 추가합니다.

`import tensorflow as tf` - 여기에 이 줄에 대한 설명을 추가합니다.

`from tensorflow import keras` - 여기에 이 줄에 대한 설명을 추가합니다.

`from tensorflow.keras import layers` - 여기에 이 줄에 대한 설명을 추가합니다.

### 코드 셀 설명

`from keras.models import Sequential` - 여기에 이 줄에 대한 설명을 추가합니다.

`from keras.layers import Bidirectional` - 여기에 이 줄에 대한 설명을 추가합니다.

`from keras.layers import LSTM` - 여기에 이 줄에 대한 설명을 추가합니다.

`from keras.layers import Dense` - 여기에 이 줄에 대한 설명을 추가합니다.

`from keras.layers import TimeDistributed` - 여기에 이 줄에 대한 설명을 추가합니다.

`import numpy as np` - 여기에 이 줄에 대한 설명을 추가합니다.

`import pandas as pd` - 여기에 이 줄에 대한 설명을 추가합니다.

`from sklearn import preprocessing` - 여기에 이 줄에 대한 설명을 추가합니다.

`from sklearn.model_selection import KFold, cross_val_score, cross_val_predict, GridSearchCV` - 여기에 이 줄에 대한 설명을 추가합니다.

`from sklearn.tree import DecisionTreeClassifier` - 여기에 이 줄에 대한 설명을 추가합니다.

`from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier` - 여기에 이 줄에 대한 설명을 추가합니다.

`from sklearn.neighbors import KNeighborsClassifier` - 여기에 이 줄에 대한 설명을 추가합니다.

`from sklearn.naive_bayes import GaussianNB` - 여기에 이 줄에 대한 설명을 추가합니다.

`from sklearn.svm import SVC, LinearSVC` - 여기에 이 줄에 대한 설명을 추가합니다.

`from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier` - 여기에 이 줄에 대한 설명을 추가합니다.

`from sklearn.model_selection import StratifiedKFold` - 여기에 이 줄에 대한 설명을 추가합니다.

`from sklearn.model_selection import train_test_split` - 여기에 이 줄에 대한 설명을 추가합니다.

`from sklearn.feature_selection import RFECV` - 여기에 이 줄에 대한 설명을 추가합니다.

`from sklearn.metrics import make_scorer, accuracy_score, classification_report, confusion_matrix, mean_squared_error` - 여기에 이 줄에 대한 설명을 추가합니다.

`import matplotlib as mpl` - 여기에 이 줄에 대한 설명을 추가합니다.

`import matplotlib.pyplot as plt` - 여기에 이 줄에 대한 설명을 추가합니다.

`import matplotlib.pylab as pylab` - 여기에 이 줄에 대한 설명을 추가합니다.

`import seaborn as sns` - 여기에 이 줄에 대한 설명을 추가합니다.

`from pandas import get_dummies` - 여기에 이 줄에 대한 설명을 추가합니다.

`import xgboost as xgb` - 여기에 이 줄에 대한 설명을 추가합니다.

`import scipy` - 여기에 이 줄에 대한 설명을 추가합니다.

`import math` - 여기에 이 줄에 대한 설명을 추가합니다.

`import json` - 여기에 이 줄에 대한 설명을 추가합니다.

`import sys` - 여기에 이 줄에 대한 설명을 추가합니다.

`import csv` - 여기에 이 줄에 대한 설명을 추가합니다.

`import os` - 여기에 이 줄에 대한 설명을 추가합니다.

`import tqdm` - 여기에 이 줄에 대한 설명을 추가합니다.

`import keras` - 여기에 이 줄에 대한 설명을 추가합니다.

`from keras.models import Sequential` - 여기에 이 줄에 대한 설명을 추가합니다.

`from keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional` - 여기에 이 줄에 대한 설명을 추가합니다.

`from keras.optimizers import SGD` - 여기에 이 줄에 대한 설명을 추가합니다.

`from tqdm import tqdm_notebook` - 여기에 이 줄에 대한 설명을 추가합니다.

### 코드 셀 설명

`from tensorflow.keras.models import Sequential` - 여기에 이 줄에 대한 설명을 추가합니다.

`from tensorflow.keras.layers import Dense, LSTM, Conv1D, Lambda, ConvLSTM2D, Flatten` - 여기에 이 줄에 대한 설명을 추가합니다.

`from tensorflow.keras.losses import Huber` - 여기에 이 줄에 대한 설명을 추가합니다.

`from tensorflow.keras.optimizers import Adam` - 여기에 이 줄에 대한 설명을 추가합니다.

`from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint` - 여기에 이 줄에 대한 설명을 추가합니다.

### 마크다운 셀 설명

# ANN


### 마크다운 셀 설명

참고 코드 : https://colab.research.google.com/drive/1rIylR9RWEckndbyFNx1Wl_yUQZUF-wyI#scrollTo=70zNPAmbZcGh

### 코드 셀 설명

`inputs = keras.Input(shape=(48, 48, 3))` - 여기에 이 줄에 대한 설명을 추가합니다.

`x = inputs` - 여기에 이 줄에 대한 설명을 추가합니다.

`x = layers.Flatten()(x)` - 여기에 이 줄에 대한 설명을 추가합니다.

`x = layers.Dense(32, activation='relu')(x)` - 여기에 이 줄에 대한 설명을 추가합니다.

`x = layers.Dense(1, activation='sigmoid')(x)` - 여기에 이 줄에 대한 설명을 추가합니다.

`outputs = x` - 여기에 이 줄에 대한 설명을 추가합니다.

`ANN = keras.Model(inputs, outputs)` - 여기에 이 줄에 대한 설명을 추가합니다.

`ANN.summary()` - 여기에 이 줄에 대한 설명을 추가합니다.

### 코드 셀 설명

`#정확도` - 여기에 이 줄에 대한 설명을 추가합니다.

`num_iters = train_len // batch_size` - 여기에 이 줄에 대한 설명을 추가합니다.

`num_epochs = 11` - 여기에 이 줄에 대한 설명을 추가합니다.

`optimizer = tf.keras.optimizers.Adam(0.0001)` - 여기에 이 줄에 대한 설명을 추가합니다.

`loss_fn = tf.keras.losses.BinaryCrossentropy()` - 여기에 이 줄에 대한 설명을 추가합니다.

`acc_fn = tf.keras.metrics.BinaryAccuracy()` - 여기에 이 줄에 대한 설명을 추가합니다.

`num_test_iters = num_iters // 4` - 여기에 이 줄에 대한 설명을 추가합니다.

`for epoch in range(num_epochs) :` - 여기에 이 줄에 대한 설명을 추가합니다.

`    epoch_loss_avg = tf.keras.metrics.Mean()` - 여기에 이 줄에 대한 설명을 추가합니다.

`    val_loss_avg = tf.keras.metrics.Mean()` - 여기에 이 줄에 대한 설명을 추가합니다.

`    epoch_acc_avg = tf.keras.metrics.Mean()` - 여기에 이 줄에 대한 설명을 추가합니다.

`    val_acc_avg = tf.keras.metrics.Mean()` - 여기에 이 줄에 대한 설명을 추가합니다.

`    for iter in range(num_iters) :` - 여기에 이 줄에 대한 설명을 추가합니다.

`        x_batch, y_batch = next(train_gen)` - 여기에 이 줄에 대한 설명을 추가합니다.

`        y_batch = y_batch.reshape(-1,1)` - 여기에 이 줄에 대한 설명을 추가합니다.

`        y_ = ANN(x_batch)` - 여기에 이 줄에 대한 설명을 추가합니다.

`        loss_value = loss_fn(y_batch, y_)` - 여기에 이 줄에 대한 설명을 추가합니다.

`        acc_value = acc_fn(y_batch, y_)` - 여기에 이 줄에 대한 설명을 추가합니다.

`        epoch_loss_avg(loss_value)` - 여기에 이 줄에 대한 설명을 추가합니다.

`        epoch_acc_avg(acc_value)` - 여기에 이 줄에 대한 설명을 추가합니다.

`    for iter in range(num_test_iters) :` - 여기에 이 줄에 대한 설명을 추가합니다.

`        x_batch, y_batch = next(test_gen)` - 여기에 이 줄에 대한 설명을 추가합니다.

`        y_batch = y_batch.reshape(-1,1)` - 여기에 이 줄에 대한 설명을 추가합니다.

`        y_= ANN(x_batch)` - 여기에 이 줄에 대한 설명을 추가합니다.

`        loss_value2 = loss_fn(y_batch, y_)` - 여기에 이 줄에 대한 설명을 추가합니다.

`        acc_value2 = acc_fn(y_batch, y_)` - 여기에 이 줄에 대한 설명을 추가합니다.

`        val_loss_avg(loss_value2)` - 여기에 이 줄에 대한 설명을 추가합니다.

`        val_acc_avg(acc_value2)` - 여기에 이 줄에 대한 설명을 추가합니다.

`    print("Epoch {:03d}: , Train Loss: {:.5f} , Train acc: {:.5f}".format(epoch, epoch_loss_avg.result(), epoch_acc_avg.result()))` - 여기에 이 줄에 대한 설명을 추가합니다.

`    print("Val_Loss: {:.3f}, Val_acc: {:.3f}".format(val_loss_avg.result(), val_acc_avg.result()))` - 여기에 이 줄에 대한 설명을 추가합니다.

`ANN_loss = float(format(val_loss_avg.result()))` - 여기에 이 줄에 대한 설명을 추가합니다.

`ANN_ACC = float(format(val_acc_avg.result()))` - 여기에 이 줄에 대한 설명을 추가합니다.

### 코드 셀 설명

`ANN_ACC` - 여기에 이 줄에 대한 설명을 추가합니다.

### 마크다운 셀 설명

# 1D CNN

### 코드 셀 설명

`inputs = keras.Input(shape=(48, 48, 3))` - 여기에 이 줄에 대한 설명을 추가합니다.

`x = inputs` - 여기에 이 줄에 대한 설명을 추가합니다.

`x = layers.Conv1D(filters=32, kernel_size=5, padding="causal",activation="relu")(x)` - 여기에 이 줄에 대한 설명을 추가합니다.

`x = layers.Flatten()(x)` - 여기에 이 줄에 대한 설명을 추가합니다.

`x = layers.Dense(16, activation="relu")(x)` - 여기에 이 줄에 대한 설명을 추가합니다.

`x = layers.Dense(1, activation='sigmoid')(x)` - 여기에 이 줄에 대한 설명을 추가합니다.

`outputs = x` - 여기에 이 줄에 대한 설명을 추가합니다.

`CNN1D = keras.Model(inputs, outputs)` - 여기에 이 줄에 대한 설명을 추가합니다.

`CNN1D.summary()` - 여기에 이 줄에 대한 설명을 추가합니다.

### 코드 셀 설명

`#정확도` - 여기에 이 줄에 대한 설명을 추가합니다.

`num_iters = train_len // batch_size` - 여기에 이 줄에 대한 설명을 추가합니다.

`num_epochs = 11` - 여기에 이 줄에 대한 설명을 추가합니다.

`optimizer = tf.keras.optimizers.Adam(0.0001)` - 여기에 이 줄에 대한 설명을 추가합니다.

`loss_fn = tf.keras.losses.BinaryCrossentropy()` - 여기에 이 줄에 대한 설명을 추가합니다.

`acc_fn = tf.keras.metrics.BinaryAccuracy()` - 여기에 이 줄에 대한 설명을 추가합니다.

`num_test_iters = num_iters // 4` - 여기에 이 줄에 대한 설명을 추가합니다.

`for epoch in range(num_epochs) :` - 여기에 이 줄에 대한 설명을 추가합니다.

`    epoch_loss_avg = tf.keras.metrics.Mean()` - 여기에 이 줄에 대한 설명을 추가합니다.

`    val_loss_avg = tf.keras.metrics.Mean()` - 여기에 이 줄에 대한 설명을 추가합니다.

`    epoch_acc_avg = tf.keras.metrics.Mean()` - 여기에 이 줄에 대한 설명을 추가합니다.

`    val_acc_avg = tf.keras.metrics.Mean()` - 여기에 이 줄에 대한 설명을 추가합니다.

`    for iter in range(num_iters) :` - 여기에 이 줄에 대한 설명을 추가합니다.

`        x_batch, y_batch = next(train_gen)` - 여기에 이 줄에 대한 설명을 추가합니다.

`        y_batch = y_batch.reshape(-1,1)` - 여기에 이 줄에 대한 설명을 추가합니다.

`        y_ = CNN1D(x_batch)` - 여기에 이 줄에 대한 설명을 추가합니다.

`        loss_value = loss_fn(y_batch, y_)` - 여기에 이 줄에 대한 설명을 추가합니다.

`        acc_value = acc_fn(y_batch, y_)` - 여기에 이 줄에 대한 설명을 추가합니다.

`        epoch_loss_avg(loss_value)` - 여기에 이 줄에 대한 설명을 추가합니다.

`        epoch_acc_avg(acc_value)` - 여기에 이 줄에 대한 설명을 추가합니다.

`    for iter in range(num_test_iters) :` - 여기에 이 줄에 대한 설명을 추가합니다.

`        x_batch, y_batch = next(test_gen)` - 여기에 이 줄에 대한 설명을 추가합니다.

`        y_batch = y_batch.reshape(-1,1)` - 여기에 이 줄에 대한 설명을 추가합니다.

`        y_= CNN1D(x_batch)` - 여기에 이 줄에 대한 설명을 추가합니다.

`        loss_value2 = loss_fn(y_batch, y_)` - 여기에 이 줄에 대한 설명을 추가합니다.

`        acc_value2 = acc_fn(y_batch, y_)` - 여기에 이 줄에 대한 설명을 추가합니다.

`        val_loss_avg(loss_value2)` - 여기에 이 줄에 대한 설명을 추가합니다.

`        val_acc_avg(acc_value2)` - 여기에 이 줄에 대한 설명을 추가합니다.

`    print("Epoch {:03d}: , Train Loss: {:.5f} , Train acc: {:.5f}".format(epoch, epoch_loss_avg.result(), epoch_acc_avg.result()))` - 여기에 이 줄에 대한 설명을 추가합니다.

`    print("Val_Loss: {:.3f}, Val_acc: {:.3f}".format(val_loss_avg.result(), val_acc_avg.result()))` - 여기에 이 줄에 대한 설명을 추가합니다.

`CNN1D_loss = float(format(val_loss_avg.result()))` - 여기에 이 줄에 대한 설명을 추가합니다.

`CNN1D_ACC = float(format(val_acc_avg.result()))` - 여기에 이 줄에 대한 설명을 추가합니다.

### 코드 셀 설명

`CNN1D_ACC` - 여기에 이 줄에 대한 설명을 추가합니다.

### 마크다운 셀 설명

# CNN

### 마크다운 셀 설명

참고 논문 : Using Deep Learning Neural Networks and Candlestick chart Representation to Predict Stock Market
https://arxiv.org/pdf/1903.12258.pdf

### 마크다운 셀 설명

다음날 종가가 상승이냐 아니냐를 맞추는 binary classification 문제

### 코드 셀 설명

`inputs = keras.Input(shape=(48, 48, 3))` - 여기에 이 줄에 대한 설명을 추가합니다.

`x = inputs` - 여기에 이 줄에 대한 설명을 추가합니다.

`x = layers.Conv2D(48, 3, activation='relu', padding="same")(x)` - 여기에 이 줄에 대한 설명을 추가합니다.

`x = layers.MaxPooling2D(2)(x)` - 여기에 이 줄에 대한 설명을 추가합니다.

`x = layers.Dropout(rate=0.5)(x)` - 여기에 이 줄에 대한 설명을 추가합니다.

`x = layers.Conv2D(96, 3, activation='relu', padding="same")(x)` - 여기에 이 줄에 대한 설명을 추가합니다.

`x = layers.MaxPooling2D(2)(x)` - 여기에 이 줄에 대한 설명을 추가합니다.

`x = layers.Dropout(rate=0.5)(x)` - 여기에 이 줄에 대한 설명을 추가합니다.

`x = layers.Flatten()(x)` - 여기에 이 줄에 대한 설명을 추가합니다.

`x = layers.Dense(1, activation='sigmoid')(x)` - 여기에 이 줄에 대한 설명을 추가합니다.

`outputs = x` - 여기에 이 줄에 대한 설명을 추가합니다.

`CNN = keras.Model(inputs, outputs)` - 여기에 이 줄에 대한 설명을 추가합니다.

`CNN.summary()` - 여기에 이 줄에 대한 설명을 추가합니다.

### 코드 셀 설명

`#정확도` - 여기에 이 줄에 대한 설명을 추가합니다.

`num_iters = train_len // batch_size` - 여기에 이 줄에 대한 설명을 추가합니다.

`num_epochs = 11` - 여기에 이 줄에 대한 설명을 추가합니다.

`optimizer = tf.keras.optimizers.Adam(0.0001)` - 여기에 이 줄에 대한 설명을 추가합니다.

`loss_fn = tf.keras.losses.BinaryCrossentropy()` - 여기에 이 줄에 대한 설명을 추가합니다.

`acc_fn = tf.keras.metrics.BinaryAccuracy()` - 여기에 이 줄에 대한 설명을 추가합니다.

`num_test_iters = num_iters // 4` - 여기에 이 줄에 대한 설명을 추가합니다.

`for epoch in range(num_epochs) :` - 여기에 이 줄에 대한 설명을 추가합니다.

`    epoch_loss_avg = tf.keras.metrics.Mean()` - 여기에 이 줄에 대한 설명을 추가합니다.

`    val_loss_avg = tf.keras.metrics.Mean()` - 여기에 이 줄에 대한 설명을 추가합니다.

`    epoch_acc_avg = tf.keras.metrics.Mean()` - 여기에 이 줄에 대한 설명을 추가합니다.

`    val_acc_avg = tf.keras.metrics.Mean()` - 여기에 이 줄에 대한 설명을 추가합니다.

`    for iter in range(num_iters) :` - 여기에 이 줄에 대한 설명을 추가합니다.

`        x_batch, y_batch = next(train_gen)` - 여기에 이 줄에 대한 설명을 추가합니다.

`        y_batch = y_batch.reshape(-1,1)` - 여기에 이 줄에 대한 설명을 추가합니다.

`        y_ = CNN(x_batch)` - 여기에 이 줄에 대한 설명을 추가합니다.

`        loss_value = loss_fn(y_batch, y_)` - 여기에 이 줄에 대한 설명을 추가합니다.

`        acc_value = acc_fn(y_batch, y_)` - 여기에 이 줄에 대한 설명을 추가합니다.

`        epoch_loss_avg(loss_value)` - 여기에 이 줄에 대한 설명을 추가합니다.

`        epoch_acc_avg(acc_value)` - 여기에 이 줄에 대한 설명을 추가합니다.

`    for iter in range(num_test_iters) :` - 여기에 이 줄에 대한 설명을 추가합니다.

`        x_batch, y_batch = next(test_gen)` - 여기에 이 줄에 대한 설명을 추가합니다.

`        y_batch = y_batch.reshape(-1,1)` - 여기에 이 줄에 대한 설명을 추가합니다.

`        y_= CNN(x_batch)` - 여기에 이 줄에 대한 설명을 추가합니다.

`        loss_value2 = loss_fn(y_batch, y_)` - 여기에 이 줄에 대한 설명을 추가합니다.

`        acc_value2 = acc_fn(y_batch, y_)` - 여기에 이 줄에 대한 설명을 추가합니다.

`        val_loss_avg(loss_value2)` - 여기에 이 줄에 대한 설명을 추가합니다.

`        val_acc_avg(acc_value2)` - 여기에 이 줄에 대한 설명을 추가합니다.

`    print("Epoch {:03d}: , Train Loss: {:.5f} , Train acc: {:.5f}".format(epoch, epoch_loss_avg.result(), epoch_acc_avg.result()))` - 여기에 이 줄에 대한 설명을 추가합니다.

`    print("Val_Loss: {:.3f}, Val_acc: {:.3f}".format(val_loss_avg.result(), val_acc_avg.result()))` - 여기에 이 줄에 대한 설명을 추가합니다.

`CNN_loss = float(format(val_loss_avg.result()))` - 여기에 이 줄에 대한 설명을 추가합니다.

`CNN_ACC = float(format(val_acc_avg.result()))` - 여기에 이 줄에 대한 설명을 추가합니다.

### 마크다운 셀 설명

# GRU


### 마크다운 셀 설명

소스코드 : https://github.com/zutshianand/Stock-Price-Prediction/blob/master/main.ipynb

### 코드 셀 설명

`regressorGRU = Sequential()` - 여기에 이 줄에 대한 설명을 추가합니다.

`# First GRU layer with Dropout regularisation` - 여기에 이 줄에 대한 설명을 추가합니다.

`regressorGRU.add(GRU(units=50, return_sequences=True, input_shape=(2304, 3), activation='tanh'))` - 여기에 이 줄에 대한 설명을 추가합니다.

`# The output layer` - 여기에 이 줄에 대한 설명을 추가합니다.

`regressorGRU.add(Dense(units=1))` - 여기에 이 줄에 대한 설명을 추가합니다.

`regressorGRU.summary()` - 여기에 이 줄에 대한 설명을 추가합니다.

### 코드 셀 설명

`#정확도` - 여기에 이 줄에 대한 설명을 추가합니다.

`num_iters = train_len // batch_size` - 여기에 이 줄에 대한 설명을 추가합니다.

`num_epochs = 11` - 여기에 이 줄에 대한 설명을 추가합니다.

`optimizer = tf.keras.optimizers.Adam(0.0001)` - 여기에 이 줄에 대한 설명을 추가합니다.

`loss_fn = tf.keras.losses.BinaryCrossentropy()` - 여기에 이 줄에 대한 설명을 추가합니다.

`acc_fn = tf.keras.metrics.BinaryAccuracy()` - 여기에 이 줄에 대한 설명을 추가합니다.

`num_test_iters = num_iters // 4` - 여기에 이 줄에 대한 설명을 추가합니다.

`for epoch in range(num_epochs) :` - 여기에 이 줄에 대한 설명을 추가합니다.

`    epoch_loss_avg = tf.keras.metrics.Mean()` - 여기에 이 줄에 대한 설명을 추가합니다.

`    val_loss_avg = tf.keras.metrics.Mean()` - 여기에 이 줄에 대한 설명을 추가합니다.

`    epoch_acc_avg = tf.keras.metrics.Mean()` - 여기에 이 줄에 대한 설명을 추가합니다.

`    val_acc_avg = tf.keras.metrics.Mean()` - 여기에 이 줄에 대한 설명을 추가합니다.

`    for iter in range(num_iters) :` - 여기에 이 줄에 대한 설명을 추가합니다.

`        x_batch, y_batch = next(train_gen)` - 여기에 이 줄에 대한 설명을 추가합니다.

`        x_batch = x_batch.reshape(16, 2304, 3)` - 여기에 이 줄에 대한 설명을 추가합니다.

`        epoch_loss_avg(loss_value)` - 여기에 이 줄에 대한 설명을 추가합니다.

`        epoch_acc_avg(acc_value)` - 여기에 이 줄에 대한 설명을 추가합니다.

`    for iter in range(num_test_iters) :` - 여기에 이 줄에 대한 설명을 추가합니다.

`        x_batch, y_batch = next(test_gen)` - 여기에 이 줄에 대한 설명을 추가합니다.

`        x_batch = x_batch.reshape(16, 2304, 3)` - 여기에 이 줄에 대한 설명을 추가합니다.

`        y_= regressorGRU(x_batch)` - 여기에 이 줄에 대한 설명을 추가합니다.

`        loss_value = loss_fn(y_batch, y_)` - 여기에 이 줄에 대한 설명을 추가합니다.

`        acc_value = acc_fn(y_batch, y_)` - 여기에 이 줄에 대한 설명을 추가합니다.

`        val_loss_avg(loss_value)` - 여기에 이 줄에 대한 설명을 추가합니다.

`        val_acc_avg(acc_value)` - 여기에 이 줄에 대한 설명을 추가합니다.

`    print("Epoch {:03d}: , Train Loss: {:.5f} , Train acc: {:.5f}".format(epoch, epoch_loss_avg.result(), epoch_acc_avg.result()))` - 여기에 이 줄에 대한 설명을 추가합니다.

`    print("Val_Loss: {:.3f}, Val_acc: {:.3f}".format(val_loss_avg.result(), val_acc_avg.result()))` - 여기에 이 줄에 대한 설명을 추가합니다.

`GRU_loss = float(format(val_loss_avg.result()))` - 여기에 이 줄에 대한 설명을 추가합니다.

`GRU_ACC = float(format(val_acc_avg.result()))` - 여기에 이 줄에 대한 설명을 추가합니다.

### 코드 셀 설명

`GRU_ACC` - 여기에 이 줄에 대한 설명을 추가합니다.

### 마크다운 셀 설명

# LSTM

### 마크다운 셀 설명

shape 확인 기존 48,48,3 -> 48,3으로 변경

### 코드 셀 설명

`inputs = keras.Input(shape=(2304, 3))` - 여기에 이 줄에 대한 설명을 추가합니다.

`x = inputs` - 여기에 이 줄에 대한 설명을 추가합니다.

`# First LSTM layer with Dropout regularisation` - 여기에 이 줄에 대한 설명을 추가합니다.

`x = layers.LSTM(units=32, return_sequences=True)(x)` - 여기에 이 줄에 대한 설명을 추가합니다.

`x = layers.Dropout(rate=0.5)(x)` - 여기에 이 줄에 대한 설명을 추가합니다.

`# Second LSTM layer` - 여기에 이 줄에 대한 설명을 추가합니다.

`x = layers.LSTM(units=32, return_sequences=True)(x)` - 여기에 이 줄에 대한 설명을 추가합니다.

`x = layers.Dropout(rate=0.5)(x)` - 여기에 이 줄에 대한 설명을 추가합니다.

`# Third LSTM layer` - 여기에 이 줄에 대한 설명을 추가합니다.

`x = layers.LSTM(units=32, return_sequences=True)(x)` - 여기에 이 줄에 대한 설명을 추가합니다.

`x = layers.Dropout(rate=0.5)(x)` - 여기에 이 줄에 대한 설명을 추가합니다.

`# Fourth LSTM layer` - 여기에 이 줄에 대한 설명을 추가합니다.

`x = layers.LSTM(units=32, return_sequences=True)(x)` - 여기에 이 줄에 대한 설명을 추가합니다.

`x = layers.Dropout(rate=0.5)(x)` - 여기에 이 줄에 대한 설명을 추가합니다.

`# The output layer` - 여기에 이 줄에 대한 설명을 추가합니다.

`x = layers.Dense(1)(x)` - 여기에 이 줄에 대한 설명을 추가합니다.

`outputs = x` - 여기에 이 줄에 대한 설명을 추가합니다.

`LSTM = keras.Model(inputs, outputs)` - 여기에 이 줄에 대한 설명을 추가합니다.

`LSTM.summary()` - 여기에 이 줄에 대한 설명을 추가합니다.

### 코드 셀 설명

`#정확도` - 여기에 이 줄에 대한 설명을 추가합니다.

`num_iters = train_len // batch_size` - 여기에 이 줄에 대한 설명을 추가합니다.

`num_epochs = 11` - 여기에 이 줄에 대한 설명을 추가합니다.

`optimizer = tf.keras.optimizers.Adam(0.0001)` - 여기에 이 줄에 대한 설명을 추가합니다.

`loss_fn = tf.keras.losses.BinaryCrossentropy()` - 여기에 이 줄에 대한 설명을 추가합니다.

`acc_fn = tf.keras.metrics.BinaryAccuracy()` - 여기에 이 줄에 대한 설명을 추가합니다.

`num_test_iters = num_iters // 4` - 여기에 이 줄에 대한 설명을 추가합니다.

`for epoch in range(num_epochs) :` - 여기에 이 줄에 대한 설명을 추가합니다.

`    epoch_loss_avg = tf.keras.metrics.Mean()` - 여기에 이 줄에 대한 설명을 추가합니다.

`    val_loss_avg = tf.keras.metrics.Mean()` - 여기에 이 줄에 대한 설명을 추가합니다.

`    epoch_acc_avg = tf.keras.metrics.Mean()` - 여기에 이 줄에 대한 설명을 추가합니다.

`    val_acc_avg = tf.keras.metrics.Mean()` - 여기에 이 줄에 대한 설명을 추가합니다.

`    for iter in range(num_iters) :` - 여기에 이 줄에 대한 설명을 추가합니다.

`        x_batch, y_batch = next(train_gen)` - 여기에 이 줄에 대한 설명을 추가합니다.

`        x_batch = x_batch.reshape(16, 2304, 3)` - 여기에 이 줄에 대한 설명을 추가합니다.

`        y_ = LSTM(x_batch)` - 여기에 이 줄에 대한 설명을 추가합니다.

`        loss_value = loss_fn(y_batch, y_)` - 여기에 이 줄에 대한 설명을 추가합니다.

`        acc_value = acc_fn(y_batch, y_)` - 여기에 이 줄에 대한 설명을 추가합니다.

`        epoch_loss_avg.update_state(loss_value)` - 여기에 이 줄에 대한 설명을 추가합니다.

`        epoch_acc_avg(acc_value)` - 여기에 이 줄에 대한 설명을 추가합니다.

`    for iter in range(num_test_iters) :` - 여기에 이 줄에 대한 설명을 추가합니다.

`        x_batch, y_batch = next(test_gen)` - 여기에 이 줄에 대한 설명을 추가합니다.

`        x_batch = x_batch.reshape(16, 2304, 3)` - 여기에 이 줄에 대한 설명을 추가합니다.

`        y_= LSTM(x_batch)` - 여기에 이 줄에 대한 설명을 추가합니다.

`        loss_value2 = loss_fn(y_batch, y_)` - 여기에 이 줄에 대한 설명을 추가합니다.

`        acc_value2 = acc_fn(y_batch, y_)` - 여기에 이 줄에 대한 설명을 추가합니다.

`        val_loss_avg.update_state(loss_value2)` - 여기에 이 줄에 대한 설명을 추가합니다.

`        val_acc_avg(acc_value2)` - 여기에 이 줄에 대한 설명을 추가합니다.

`    print("Epoch {:03d}: , Train Loss: {:.5f} , Train acc: {:.5f}".format(epoch, epoch_loss_avg.result(), epoch_acc_avg.result()))` - 여기에 이 줄에 대한 설명을 추가합니다.

`    print("Val_Loss: {:.3f}, Val_acc: {:.3f}".format(val_loss_avg.result(), val_acc_avg.result()))` - 여기에 이 줄에 대한 설명을 추가합니다.

`LSTM_loss = float(format(val_loss_avg.result()))` - 여기에 이 줄에 대한 설명을 추가합니다.

`LSTM_ACC = float(format(val_acc_avg.result()))` - 여기에 이 줄에 대한 설명을 추가합니다.

### 코드 셀 설명

`LSTM_ACC` - 여기에 이 줄에 대한 설명을 추가합니다.

### 마크다운 셀 설명

# BLSTM

### 마크다운 셀 설명

소스코드 : https://machinelearningmastery.com/develop-bidirectional-lstm-sequence-classification-python-keras/

### 코드 셀 설명

`from keras.models import Sequential` - 여기에 이 줄에 대한 설명을 추가합니다.

`from keras.layers import Bidirectional` - 여기에 이 줄에 대한 설명을 추가합니다.

`from keras.layers import LSTM` - 여기에 이 줄에 대한 설명을 추가합니다.

`from keras.layers import Dense` - 여기에 이 줄에 대한 설명을 추가합니다.

`from keras.layers import TimeDistributed` - 여기에 이 줄에 대한 설명을 추가합니다.

### 코드 셀 설명

`BL = Sequential()` - 여기에 이 줄에 대한 설명을 추가합니다.

`BL.add(Bidirectional(LSTM(20, return_sequences=True), input_shape=(2304, 3)))` - 여기에 이 줄에 대한 설명을 추가합니다.

`BL.add(TimeDistributed(Dense(1, activation='sigmoid')))` - 여기에 이 줄에 대한 설명을 추가합니다.

`BL.summary()` - 여기에 이 줄에 대한 설명을 추가합니다.

### 코드 셀 설명

`#정확도` - 여기에 이 줄에 대한 설명을 추가합니다.

`num_iters = train_len // batch_size` - 여기에 이 줄에 대한 설명을 추가합니다.

`num_epochs = 11` - 여기에 이 줄에 대한 설명을 추가합니다.

`optimizer = tf.keras.optimizers.Adam(0.0001)` - 여기에 이 줄에 대한 설명을 추가합니다.

`loss_fn = tf.keras.losses.BinaryCrossentropy()` - 여기에 이 줄에 대한 설명을 추가합니다.

`acc_fn = tf.keras.metrics.BinaryAccuracy()` - 여기에 이 줄에 대한 설명을 추가합니다.

`num_test_iters = num_iters // 4` - 여기에 이 줄에 대한 설명을 추가합니다.

`for epoch in range(num_epochs) :` - 여기에 이 줄에 대한 설명을 추가합니다.

`    epoch_loss_avg = tf.keras.metrics.Mean()` - 여기에 이 줄에 대한 설명을 추가합니다.

`    val_loss_avg = tf.keras.metrics.Mean()` - 여기에 이 줄에 대한 설명을 추가합니다.

`    epoch_acc_avg = tf.keras.metrics.Mean()` - 여기에 이 줄에 대한 설명을 추가합니다.

`    val_acc_avg = tf.keras.metrics.Mean()` - 여기에 이 줄에 대한 설명을 추가합니다.

`    for iter in range(num_iters) :` - 여기에 이 줄에 대한 설명을 추가합니다.

`        x_batch, y_batch = next(train_gen)` - 여기에 이 줄에 대한 설명을 추가합니다.

`        x_batch = x_batch.reshape(16, 2304, 3)` - 여기에 이 줄에 대한 설명을 추가합니다.

`        y_ = BL(x_batch)` - 여기에 이 줄에 대한 설명을 추가합니다.

`        loss_value = loss_fn(y_batch, y_)` - 여기에 이 줄에 대한 설명을 추가합니다.

`        acc_value = acc_fn(y_batch, y_)` - 여기에 이 줄에 대한 설명을 추가합니다.

`        epoch_loss_avg(loss_value)` - 여기에 이 줄에 대한 설명을 추가합니다.

`        epoch_acc_avg(acc_value)` - 여기에 이 줄에 대한 설명을 추가합니다.

`    for iter in range(num_test_iters) :` - 여기에 이 줄에 대한 설명을 추가합니다.

`        x_batch, y_batch = next(test_gen)` - 여기에 이 줄에 대한 설명을 추가합니다.

`        x_batch = x_batch.reshape(16, 2304, 3)` - 여기에 이 줄에 대한 설명을 추가합니다.

`        y_= BL(x_batch)` - 여기에 이 줄에 대한 설명을 추가합니다.

`        loss_value2 = loss_fn(y_batch, y_)` - 여기에 이 줄에 대한 설명을 추가합니다.

`        acc_value = acc_fn(y_batch, y_)` - 여기에 이 줄에 대한 설명을 추가합니다.

`        val_loss_avg(loss_value2)` - 여기에 이 줄에 대한 설명을 추가합니다.

`        val_acc_avg(acc_value)` - 여기에 이 줄에 대한 설명을 추가합니다.

`    print("Epoch {:03d}: , Train Loss: {:.5f} , Train acc: {:.5f}".format(epoch, epoch_loss_avg.result(), epoch_acc_avg.result()))` - 여기에 이 줄에 대한 설명을 추가합니다.

`    print("Val_Loss: {:.3f}, Val_acc: {:.3f}".format(val_loss_avg.result(), val_acc_avg.result()))` - 여기에 이 줄에 대한 설명을 추가합니다.

`BLSTM_loss = float(format(val_loss_avg.result()))` - 여기에 이 줄에 대한 설명을 추가합니다.

`BLSTM_ACC = float(format(val_acc_avg.result()))` - 여기에 이 줄에 대한 설명을 추가합니다.

### 코드 셀 설명

`BLSTM_ACC` - 여기에 이 줄에 대한 설명을 추가합니다.

### 마크다운 셀 설명

# Convlstm2D

### 마크다운 셀 설명

참고 : https://deep-deep-deep.tistory.com/32 [딥딥딥]

### 마크다운 셀 설명

소스 코드 : https://keras.io/examples/vision/conv_lstm/

### 마크다운 셀 설명

CNN 관련 : http://taewan.kim/post/cnn/

### 코드 셀 설명

### 코드 셀 설명

`seq = keras.Sequential(` - 여기에 이 줄에 대한 설명을 추가합니다.

`    [` - 여기에 이 줄에 대한 설명을 추가합니다.

`        keras.Input(` - 여기에 이 줄에 대한 설명을 추가합니다.

`            shape=(48, 48, 3, 1)` - 여기에 이 줄에 대한 설명을 추가합니다.

`        ),  # Variable-length sequence of 40x40x1 frames` - 여기에 이 줄에 대한 설명을 추가합니다.

`        layers.ConvLSTM2D(` - 여기에 이 줄에 대한 설명을 추가합니다.

`            filters=40, kernel_size=(3, 3), padding="same", return_sequences=True` - 여기에 이 줄에 대한 설명을 추가합니다.

`        ),` - 여기에 이 줄에 대한 설명을 추가합니다.

`        layers.BatchNormalization(),` - 여기에 이 줄에 대한 설명을 추가합니다.

`        layers.Conv3D(` - 여기에 이 줄에 대한 설명을 추가합니다.

`            filters=1, kernel_size=(3, 3, 3), activation="sigmoid", padding="same"` - 여기에 이 줄에 대한 설명을 추가합니다.

`        ),` - 여기에 이 줄에 대한 설명을 추가합니다.

`    ]` - 여기에 이 줄에 대한 설명을 추가합니다.

`)` - 여기에 이 줄에 대한 설명을 추가합니다.

`seq.compile(loss="binary_crossentropy", optimizer="adadelta")` - 여기에 이 줄에 대한 설명을 추가합니다.

### 코드 셀 설명

`seq.summary()` - 여기에 이 줄에 대한 설명을 추가합니다.

### 코드 셀 설명

`#정확도` - 여기에 이 줄에 대한 설명을 추가합니다.

`num_iters = train_len // batch_size` - 여기에 이 줄에 대한 설명을 추가합니다.

`num_epochs = 11` - 여기에 이 줄에 대한 설명을 추가합니다.

`optimizer = tf.keras.optimizers.Adam(0.0001)` - 여기에 이 줄에 대한 설명을 추가합니다.

`loss_fn = tf.keras.losses.BinaryCrossentropy()` - 여기에 이 줄에 대한 설명을 추가합니다.

`acc_fn = tf.keras.metrics.BinaryAccuracy()` - 여기에 이 줄에 대한 설명을 추가합니다.

`num_test_iters = num_iters // 4` - 여기에 이 줄에 대한 설명을 추가합니다.

`for epoch in range(num_epochs) :` - 여기에 이 줄에 대한 설명을 추가합니다.

`    epoch_loss_avg = tf.keras.metrics.Mean()` - 여기에 이 줄에 대한 설명을 추가합니다.

`    val_loss_avg = tf.keras.metrics.Mean()` - 여기에 이 줄에 대한 설명을 추가합니다.

`    epoch_acc_avg = tf.keras.metrics.Mean()` - 여기에 이 줄에 대한 설명을 추가합니다.

`    val_acc_avg = tf.keras.metrics.Mean()` - 여기에 이 줄에 대한 설명을 추가합니다.

`    for iter in range(num_iters) :` - 여기에 이 줄에 대한 설명을 추가합니다.

`        x_batch, y_batch = next(train_gen)` - 여기에 이 줄에 대한 설명을 추가합니다.

`        x_batch = x_batch.reshape(16, 48, 48, 3, 1)` - 여기에 이 줄에 대한 설명을 추가합니다.

`        y_ = seq(x_batch)[0]` - 여기에 이 줄에 대한 설명을 추가합니다.

`        loss_value = loss_fn(y_batch, y_)` - 여기에 이 줄에 대한 설명을 추가합니다.

`        acc_value = acc_fn(y_batch, y_)` - 여기에 이 줄에 대한 설명을 추가합니다.

`        epoch_loss_avg(loss_value)` - 여기에 이 줄에 대한 설명을 추가합니다.

`        epoch_acc_avg(acc_value)` - 여기에 이 줄에 대한 설명을 추가합니다.

`    for iter in range(num_test_iters) :` - 여기에 이 줄에 대한 설명을 추가합니다.

`        x_batch, y_batch = next(test_gen)` - 여기에 이 줄에 대한 설명을 추가합니다.

`        x_batch = x_batch.reshape(16, 48, 48, 3, 1)` - 여기에 이 줄에 대한 설명을 추가합니다.

`        y_= seq(x_batch)[0]` - 여기에 이 줄에 대한 설명을 추가합니다.

`        loss_value = loss_fn(y_batch, y_)` - 여기에 이 줄에 대한 설명을 추가합니다.

`        acc_value = acc_fn(y_batch, y_)` - 여기에 이 줄에 대한 설명을 추가합니다.

`        val_loss_avg(loss_value)` - 여기에 이 줄에 대한 설명을 추가합니다.

`        val_acc_avg(acc_value)` - 여기에 이 줄에 대한 설명을 추가합니다.

`    print("Epoch {:03d}: , Train Loss: {:.5f} , Train acc: {:.5f}".format(epoch, epoch_loss_avg.result(), epoch_acc_avg.result()))` - 여기에 이 줄에 대한 설명을 추가합니다.

`    print("Val_Loss: {:.3f}, Val_acc: {:.3f}".format(val_loss_avg.result(), val_acc_avg.result()))` - 여기에 이 줄에 대한 설명을 추가합니다.

`Convlstm2D_loss = float(format(val_loss_avg.result()))` - 여기에 이 줄에 대한 설명을 추가합니다.

`Convlstm2D_acc = float(format(val_acc_avg.result()))` - 여기에 이 줄에 대한 설명을 추가합니다.

### 코드 셀 설명

`Convlstm2D_acc` - 여기에 이 줄에 대한 설명을 추가합니다.

### 마크다운 셀 설명

 # CNN-LSTM

### 마크다운 셀 설명

소스 코드 : https://colab.research.google.com/drive/1rIylR9RWEckndbyFNx1Wl_yUQZUF-wyI#scrollTo=M1QKFdJOYTjx

### 코드 셀 설명

`from tensorflow.keras.models import Sequential` - 여기에 이 줄에 대한 설명을 추가합니다.

`from tensorflow.keras.layers import Dense, LSTM, Conv1D, Lambda, ConvLSTM2D, Flatten` - 여기에 이 줄에 대한 설명을 추가합니다.

`from tensorflow.keras.losses import Huber` - 여기에 이 줄에 대한 설명을 추가합니다.

`from tensorflow.keras.optimizers import Adam` - 여기에 이 줄에 대한 설명을 추가합니다.

`from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint` - 여기에 이 줄에 대한 설명을 추가합니다.

### 코드 셀 설명

`inputs = keras.Input(shape=(2304, 3))` - 여기에 이 줄에 대한 설명을 추가합니다.

`x = inputs` - 여기에 이 줄에 대한 설명을 추가합니다.

`# 1차원 feature map 생성` - 여기에 이 줄에 대한 설명을 추가합니다.

`x = layers.Conv1D(filters=32, kernel_size=5, padding="causal", activation="relu")(x)` - 여기에 이 줄에 대한 설명을 추가합니다.

`# LSTM` - 여기에 이 줄에 대한 설명을 추가합니다.

`x = layers.LSTM(16, activation='tanh')(x)` - 여기에 이 줄에 대한 설명을 추가합니다.

`x = layers.Dense(16, activation="relu")(x)` - 여기에 이 줄에 대한 설명을 추가합니다.

`x = layers.Dense(1, activation='sigmoid')(x)` - 여기에 이 줄에 대한 설명을 추가합니다.

`outputs = x` - 여기에 이 줄에 대한 설명을 추가합니다.

`M10 = keras.Model(inputs, outputs)` - 여기에 이 줄에 대한 설명을 추가합니다.

`M10.summary()` - 여기에 이 줄에 대한 설명을 추가합니다.

### 코드 셀 설명

`#정확도` - 여기에 이 줄에 대한 설명을 추가합니다.

`num_iters = train_len // batch_size` - 여기에 이 줄에 대한 설명을 추가합니다.

`num_epochs = 11` - 여기에 이 줄에 대한 설명을 추가합니다.

`optimizer = tf.keras.optimizers.Adam(0.0001)` - 여기에 이 줄에 대한 설명을 추가합니다.

`loss_fn = tf.keras.losses.BinaryCrossentropy()` - 여기에 이 줄에 대한 설명을 추가합니다.

`acc_fn = tf.keras.metrics.BinaryAccuracy()` - 여기에 이 줄에 대한 설명을 추가합니다.

`num_test_iters = num_iters // 4` - 여기에 이 줄에 대한 설명을 추가합니다.

`for epoch in range(num_epochs) :` - 여기에 이 줄에 대한 설명을 추가합니다.

`    epoch_loss_avg = tf.keras.metrics.Mean()` - 여기에 이 줄에 대한 설명을 추가합니다.

`    val_loss_avg = tf.keras.metrics.Mean()` - 여기에 이 줄에 대한 설명을 추가합니다.

`    epoch_acc_avg = tf.keras.metrics.Mean()` - 여기에 이 줄에 대한 설명을 추가합니다.

`    val_acc_avg = tf.keras.metrics.Mean()` - 여기에 이 줄에 대한 설명을 추가합니다.

`    for iter in range(num_iters) :` - 여기에 이 줄에 대한 설명을 추가합니다.

`        x_batch, y_batch = next(train_gen)` - 여기에 이 줄에 대한 설명을 추가합니다.

`        x_batch = x_batch.reshape(16, 2304, 3)` - 여기에 이 줄에 대한 설명을 추가합니다.

`        y_ = M10(x_batch)` - 여기에 이 줄에 대한 설명을 추가합니다.

`        y_batch = y_batch.reshape(-1,1)` - 여기에 이 줄에 대한 설명을 추가합니다.

`        loss_value = loss_fn(y_batch, y_)` - 여기에 이 줄에 대한 설명을 추가합니다.

`        acc_value = acc_fn(y_batch, y_)` - 여기에 이 줄에 대한 설명을 추가합니다.

`        epoch_loss_avg(loss_value)` - 여기에 이 줄에 대한 설명을 추가합니다.

`        epoch_acc_avg(acc_value)` - 여기에 이 줄에 대한 설명을 추가합니다.

`    for iter in range(num_test_iters) :` - 여기에 이 줄에 대한 설명을 추가합니다.

`        x_batch, y_batch = next(test_gen)` - 여기에 이 줄에 대한 설명을 추가합니다.

`        x_batch = x_batch.reshape(16, 2304, 3)` - 여기에 이 줄에 대한 설명을 추가합니다.

`        y_= M10(x_batch)` - 여기에 이 줄에 대한 설명을 추가합니다.

`        y_batch = y_batch.reshape(-1,1)` - 여기에 이 줄에 대한 설명을 추가합니다.

`        loss_value = loss_fn(y_batch, y_)` - 여기에 이 줄에 대한 설명을 추가합니다.

`        acc_value = acc_fn(y_batch, y_)` - 여기에 이 줄에 대한 설명을 추가합니다.

`        val_loss_avg(loss_value)` - 여기에 이 줄에 대한 설명을 추가합니다.

`        val_acc_avg(acc_value)` - 여기에 이 줄에 대한 설명을 추가합니다.

`    print("Epoch {:03d}: , Train Loss: {:.5f} , Train acc: {:.5f}".format(epoch, epoch_loss_avg.result(), epoch_acc_avg.result()))` - 여기에 이 줄에 대한 설명을 추가합니다.

`    print("Val_Loss: {:.3f}, Val_acc: {:.3f}".format(val_loss_avg.result(), val_acc_avg.result()))` - 여기에 이 줄에 대한 설명을 추가합니다.

`CNN_LSTM_loss = float(format(val_loss_avg.result()))` - 여기에 이 줄에 대한 설명을 추가합니다.

`CNN_LSTM_ACC = float(format(val_acc_avg.result()))` - 여기에 이 줄에 대한 설명을 추가합니다.

### 코드 셀 설명

`CNN_LSTM_ACC` - 여기에 이 줄에 대한 설명을 추가합니다.

### 마크다운 셀 설명

# CNN-BLSTM

### 코드 셀 설명

`from random import random` - 여기에 이 줄에 대한 설명을 추가합니다.

`from numpy import array` - 여기에 이 줄에 대한 설명을 추가합니다.

`from numpy import cumsum` - 여기에 이 줄에 대한 설명을 추가합니다.

`from keras.models import Sequential` - 여기에 이 줄에 대한 설명을 추가합니다.

`from keras.layers import LSTM` - 여기에 이 줄에 대한 설명을 추가합니다.

`from keras.layers import Dense` - 여기에 이 줄에 대한 설명을 추가합니다.

`from keras.layers import TimeDistributed` - 여기에 이 줄에 대한 설명을 추가합니다.

`from keras.layers import Bidirectional` - 여기에 이 줄에 대한 설명을 추가합니다.

`from random import random` - 여기에 이 줄에 대한 설명을 추가합니다.

`from numpy import array` - 여기에 이 줄에 대한 설명을 추가합니다.

`from numpy import cumsum` - 여기에 이 줄에 대한 설명을 추가합니다.

`from keras.models import Sequential` - 여기에 이 줄에 대한 설명을 추가합니다.

`from keras.layers import LSTM` - 여기에 이 줄에 대한 설명을 추가합니다.

`from keras.layers import Dense` - 여기에 이 줄에 대한 설명을 추가합니다.

`from keras.layers import TimeDistributed` - 여기에 이 줄에 대한 설명을 추가합니다.

`from keras.layers import Bidirectional` - 여기에 이 줄에 대한 설명을 추가합니다.

### 코드 셀 설명

`inputs = keras.Input(shape=(2304, 3))` - 여기에 이 줄에 대한 설명을 추가합니다.

`x = inputs` - 여기에 이 줄에 대한 설명을 추가합니다.

`# 1차원 feature map 생성` - 여기에 이 줄에 대한 설명을 추가합니다.

`x = layers.Conv1D(filters=32, kernel_size=5, padding="causal", activation="relu")(x)` - 여기에 이 줄에 대한 설명을 추가합니다.

`# LSTM` - 여기에 이 줄에 대한 설명을 추가합니다.

`x = layers.Bidirectional(LSTM(20, return_sequences=True))(x)` - 여기에 이 줄에 대한 설명을 추가합니다.

`x = layers.TimeDistributed(Dense(1, activation='sigmoid'))(x)` - 여기에 이 줄에 대한 설명을 추가합니다.

`outputs = x` - 여기에 이 줄에 대한 설명을 추가합니다.

`M11 = keras.Model(inputs, outputs)` - 여기에 이 줄에 대한 설명을 추가합니다.

`M11.summary()` - 여기에 이 줄에 대한 설명을 추가합니다.

### 코드 셀 설명

`#정확도` - 여기에 이 줄에 대한 설명을 추가합니다.

`num_iters = train_len // batch_size` - 여기에 이 줄에 대한 설명을 추가합니다.

`num_epochs = 11` - 여기에 이 줄에 대한 설명을 추가합니다.

`optimizer = tf.keras.optimizers.Adam(0.0001)` - 여기에 이 줄에 대한 설명을 추가합니다.

`loss_fn = tf.keras.losses.BinaryCrossentropy()` - 여기에 이 줄에 대한 설명을 추가합니다.

`acc_fn = tf.keras.metrics.BinaryAccuracy()` - 여기에 이 줄에 대한 설명을 추가합니다.

`num_test_iters = num_iters // 4` - 여기에 이 줄에 대한 설명을 추가합니다.

`for epoch in range(num_epochs) :` - 여기에 이 줄에 대한 설명을 추가합니다.

`    epoch_loss_avg = tf.keras.metrics.Mean()` - 여기에 이 줄에 대한 설명을 추가합니다.

`    val_loss_avg = tf.keras.metrics.Mean()` - 여기에 이 줄에 대한 설명을 추가합니다.

`    epoch_acc_avg = tf.keras.metrics.Mean()` - 여기에 이 줄에 대한 설명을 추가합니다.

`    val_acc_avg = tf.keras.metrics.Mean()` - 여기에 이 줄에 대한 설명을 추가합니다.

`    for iter in range(num_iters) :` - 여기에 이 줄에 대한 설명을 추가합니다.

`        x_batch, y_batch = next(train_gen)` - 여기에 이 줄에 대한 설명을 추가합니다.

`        x_batch = x_batch.reshape(16, 2304, 3)` - 여기에 이 줄에 대한 설명을 추가합니다.

`        y_ = M11(x_batch)` - 여기에 이 줄에 대한 설명을 추가합니다.

`        loss_value = loss_fn(y_batch, y_)` - 여기에 이 줄에 대한 설명을 추가합니다.

`        acc_value = acc_fn(y_batch, y_)` - 여기에 이 줄에 대한 설명을 추가합니다.

`        epoch_loss_avg(loss_value)` - 여기에 이 줄에 대한 설명을 추가합니다.

`        epoch_acc_avg(acc_value)` - 여기에 이 줄에 대한 설명을 추가합니다.

`    for iter in range(num_test_iters) :` - 여기에 이 줄에 대한 설명을 추가합니다.

`        x_batch, y_batch = next(test_gen)` - 여기에 이 줄에 대한 설명을 추가합니다.

`        x_batch = x_batch.reshape(16, 2304, 3)` - 여기에 이 줄에 대한 설명을 추가합니다.

`        y_= M11(x_batch)` - 여기에 이 줄에 대한 설명을 추가합니다.

`        loss_value = loss_fn(y_batch, y_)` - 여기에 이 줄에 대한 설명을 추가합니다.

`        acc_value = acc_fn(y_batch, y_)` - 여기에 이 줄에 대한 설명을 추가합니다.

`        val_loss_avg(loss_value)` - 여기에 이 줄에 대한 설명을 추가합니다.

`        val_acc_avg(acc_value)` - 여기에 이 줄에 대한 설명을 추가합니다.

`    print("Epoch {:03d}: , Train Loss: {:.5f} , Train acc: {:.5f}".format(epoch, epoch_loss_avg.result(), epoch_acc_avg.result()))` - 여기에 이 줄에 대한 설명을 추가합니다.

`    print("Val_Loss: {:.3f}, Val_acc: {:.3f}".format(val_loss_avg.result(), val_acc_avg.result()))` - 여기에 이 줄에 대한 설명을 추가합니다.

`CNN_BLSTM_loss = float(format(val_loss_avg.result()))` - 여기에 이 줄에 대한 설명을 추가합니다.

`CNN_BLSTM_ACC = float(format(val_acc_avg.result()))` - 여기에 이 줄에 대한 설명을 추가합니다.

### 코드 셀 설명

`CNN_BLSTM_ACC` - 여기에 이 줄에 대한 설명을 추가합니다.

### 마크다운 셀 설명

# Model Comparison

### 코드 셀 설명

`models = pd.DataFrame({` - 여기에 이 줄에 대한 설명을 추가합니다.

`    'Model': ['CNN','ANN', 'GRU', '1D CNN', 'LSTM', 'BLSTM', 'Convlstm2D', 'CNN-LSTM ','CNN-BLSTM'],` - 여기에 이 줄에 대한 설명을 추가합니다.

`    'Score': [CNN_ACC, ANN_ACC, GRU_ACC, CNN1D_ACC, LSTM_ACC, BLSTM_ACC, Convlstm2D_acc, CNN_LSTM_ACC, CNN_BLSTM_ACC]})` - 여기에 이 줄에 대한 설명을 추가합니다.

`models.sort_values (by='Score', ascending=False)` - 여기에 이 줄에 대한 설명을 추가합니다.

### 코드 셀 설명


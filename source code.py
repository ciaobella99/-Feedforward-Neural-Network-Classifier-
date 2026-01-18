import pandas as pd
import numpy as np
from tensorflow import keras
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
import tensorflow as tf
import random
import os
from keras.callbacks import EarlyStopping



def set_seed(seed_value):
    np.random.seed(seed_value)
    random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    tf.random.set_seed(seed_value)

seed = 666
set_seed(seed)


train_data = pd.read_csv("D:/NTUT/deep_learning_venv/DLcourse/firstexam/Data/firstReport/clsn1_trn.csv", header=None)


X_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values


norm = StandardScaler()
X_train_normed = norm.fit_transform(X_train)


label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
onehot_encoder = OneHotEncoder()
y_train_onehot = onehot_encoder.fit_transform(y_train_encoded.reshape(-1, 1)).toarray()


test_data = pd.read_csv("D:/NTUT/deep_learning_venv/DLcourse/firstexam/Data/firstReport/clsn1_tst.csv", header=None)
X_test = test_data.values
X_test_scaled = norm.transform(X_test)

model = Sequential()

model.add(Dense(16, activation='relu', input_shape=(8,)))

model.add(Dense(32, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))

model.add(Dense(7, activation='softmax'))


learning_rate = 0.001

optimizer = Adam(learning_rate=learning_rate)

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(
    monitor='val_loss',  
    patience=10, 
    restore_best_weights=True 
)

trainmodel = model.fit(X_train_normed, y_train_onehot, epochs=100, batch_size=36
                       , validation_split=0.2, verbose=1, callbacks=[early_stopping])


best_val_accuracy = max(trainmodel.history['val_accuracy'])
best_val_loss = min(trainmodel.history['val_loss'])

print(f'Best Validation Accuracy: {best_val_accuracy}')
print(f'Best Validation Loss: {best_val_loss}')



model.save("report1.h5")


y_test_pred_onehot = model.predict(X_test_scaled)
y_test_pred_encoded = np.argmax(y_test_pred_onehot, axis=1)


y_test_pred = label_encoder.inverse_transform(y_test_pred_encoded)

test_data[''] = y_test_pred
test_data.to_csv("regr_ans.csv", index=False, header=None)


import matplotlib.pyplot as plt

def plot_trainmodel(trainmodel):
    
    plt.figure()
    plt.plot(trainmodel.history['accuracy'], label='Training Accuracy')
    plt.plot(trainmodel.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('accuracy.png') 

    
    plt.figure()
    plt.plot(trainmodel.history['loss'], label='Training Loss')
    plt.plot(trainmodel.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss.png') 

    plt.show()

plot_trainmodel(trainmodel)
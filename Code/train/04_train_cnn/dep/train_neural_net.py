import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import Activation
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import Flatten
from keras.layers import Dense
from keras.activations import relu
from keras.optimizers import Adam
from keras.callbacks.callbacks import ModelCheckpoint

with open('/home/ubuntu/hists/input.pkl','rb') as infile:
    inputData = pickle.load(infile)

with open('/home/ubuntu/hists/outcome.pkl','rb') as infile:
    outcomeData = pickle.load(infile)

inputData = np.array(inputData)
outcomeData = np.array(outcomeData)

print(inputData.shape)
print(outcomeData.shape)

inputData_train, inputData_test, outcomeData_train, outcomeData_test = \
train_test_split(inputData, outcomeData,test_size=1000)

with open('/home/ubuntu/input_test.pkl','wb') as outfile1:
    pickle.dump(inputData_test,outfile1)
with open('/home/ubuntu/oucome_test.pkl','wb') as outfile2:
    pickle.dump(outcomeData_test,outfile2)
with open('/home/ubuntu/input_train.pkl','wb') as outfile3:
    pickle.dump(inputData_train,outfile3)
with open('/home/ubuntu/oucome_train.pkl','wb') as outfile4:
    pickle.dump(outcomeData_train,outfile4)

# Make the model from You et al.
cnn = Sequential()
cnn.add(Conv2D(filters=128,kernel_size=3,input_shape=(32,32,8),data_format="channels_last"))
cnn.add(BatchNormalization())
cnn.add(Activation('relu'))
cnn.add(Dropout(.5))

cnn.add(Conv2D(filters=128,kernel_size=3,input_shape=(30,30,128),strides=2,data_format="channels_last",padding="same"))
cnn.add(BatchNormalization())
cnn.add(Activation('relu'))
cnn.add(Dropout(.5))

cnn.add(Conv2D(filters=256,kernel_size=3,input_shape=(15,15,128),data_format="channels_last"))
cnn.add(BatchNormalization())
cnn.add(Activation('relu'))
cnn.add(Dropout(.5))

cnn.add(Conv2D(filters=256,kernel_size=3,input_shape=(13,13,256),strides=2,data_format="channels_last",padding="same"))
cnn.add(BatchNormalization())
cnn.add(Activation('relu'))
cnn.add(Dropout(.5))

cnn.add(Conv2D(filters=512,kernel_size=3,input_shape=(6,6,256),data_format="channels_last"))
cnn.add(BatchNormalization())
cnn.add(Activation('relu'))
cnn.add(Dropout(.5))

cnn.add(Conv2D(filters=512,kernel_size=3,input_shape=(4,4,512),data_format="channels_last"))
cnn.add(BatchNormalization())
cnn.add(Activation('relu'))
cnn.add(Dropout(.5))

cnn.add(Flatten())
cnn.add(Dense(1,activation='relu'))
# Added relu activation at end

opt = Adam(learning_rate=.001)

cnn.compile(loss='mean_squared_error',\
optimizer = opt,\
metrics = ['mse'])

cnn.fit(inputData_train,outcomeData_train,batch_size=32,\
epochs=100,validation_split=.1,shuffle=True,\
callbacks=[ModelCheckpoint('/home/ubuntu/cnn_training.h5',period=2)])

cnn.save('/home/ubuntu/cnn_trained.h5')

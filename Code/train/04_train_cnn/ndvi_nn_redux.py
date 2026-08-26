import os
import re
import sys
import pickle
from ast import literal_eval

import pandas as pd

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from siap_yields import load_muni_yields
import numpy as np
from tensorflow import keras
from keras.layers import Conv2D, Conv1D, AveragePooling2D
from keras.layers import Activation
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import Flatten
from keras.layers import Concatenate
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Sequence
from tensorflow.keras.utils import set_random_seed
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from keras import backend as K

holdout_year = int(sys.argv[1])
bins = 32


def return_adc_sample_mhist(munhist, smallhistN):
    largehistN = np.sum(munhist)
    sizearray = len(munhist.ravel())

    random_choices = np.random.choice(list(range(sizearray)), # Choose indices
                                      size=smallhistN, # Choose as many as in small histogram
                                      p=(munhist.ravel()/largehistN)) # Probs given by frequency
    d1,  d2,_            =  munhist.shape # N col, row
    d1g, d2g           =  np.meshgrid(range(d1),range(d2)) # Get indices for cols and rows in matrix form
    d1g, d2g           =  d1g.ravel(), d2g.ravel() # Flatten those index matrices
    h_tl               =  np.zeros((d1,d2))

    for chz in random_choices:
        h_pos = d1g[chz],d2g[chz] # Find position in array of choice
        h_tl[h_pos] += 1 # and add 1

    return(np.expand_dims(h_tl.T,2)/smallhistN)

def ls_nn(bins,name="model"):
  ls_in = keras.Input(shape=(bins,bins,3))

  l1_conv2d = Conv2D(
      kernel_size=(bins//5,bins//5),
      strides=(2,2),
      filters = bins*2,
      data_format="channels_last"
  )

  l1 = l1_conv2d(ls_in)
  l1 = BatchNormalization()(l1)
  l1 = Activation("relu")(l1)
  l1 = Dropout(0.5)(l1)

  l2_conv2d = Conv2D(
      kernel_size=(bins//5,bins//5),
      filters = bins*4,
      data_format="channels_last"
  )

  l2 = l2_conv2d(l1)
  l2 = BatchNormalization()(l2)
  l2 = Activation("relu")(l2)
  l2 = Dropout(0.5)(l2)

  l3_conv2d = Conv2D(
      kernel_size=((bins//5)-2,(bins//5)-2),
      filters = bins*8,
      data_format="channels_last"
  )

  l3 = l3_conv2d(l2)
  l3 = BatchNormalization()(l3)
  l3 = Activation("relu")(l3)
  l3 = Dropout(0.5)(l3)

  l4_conv2d = Conv2D(
      kernel_size=((bins//5)-2,(bins//5)-2),
      #kernel_size=(5,5),
      filters = bins*16,
      data_format="channels_last"
  )

  l4 = l4_conv2d(l3)
  l4 = BatchNormalization()(l4)
  l4 = Activation("relu")(l4)
  l4 = Dropout(0.5)(l4)

  l4 = Flatten()(l4)

  out = Dense(1)(l4)

  model = keras.Model(inputs = ls_in,outputs=out,name=name)
  return(model)


def add_zeros(x,n):
    x = str(x)
    while len(x)<n:
        x = "0"+x
    return(x)


mun_maize_yields = load_muni_yields()   # canonical SIAP Maize/Spring-Summer
#mun_maize_yields = mun_maize_yields.loc[mun_maize_yields['yield']>0]
#mun_maize_yields['yield'] = mun_maize_yields['yield'].apply(np.log)

hist_fs = os.listdir("../processed_data/muni_vi_hists_0.2_1.0_0.0_12.0_0.0_0.6_32_max/")
hist_fs = [f for f in hist_fs if ".pkl" in f]

hist_df = pd.concat([pd.read_pickle(f"../processed_data/muni_vi_hists_0.2_1.0_0.0_12.0_0.0_0.6_32_max/{f}") for f in hist_fs])

hist_df['hist'] = hist_df.apply(lambda x: np.concatenate(
    [np.reshape(x['ndvi_hist'],(bins,bins,1)),
     np.reshape(x['gcvi_hist'],(bins,bins,1)),
     np.reshape(x['ndti_hist'],(bins,bins,1))],
     axis=-1
    ),
    axis=1)

#hist_df['hist'] = hist_df['gcvi_hist'].apply(lambda x: np.reshape(x,(bins,bins,1)))

hist_df = pd.merge(hist_df,mun_maize_yields,
                  on=['muncode','year'])

adc_pixel_counts = pd.read_csv("../processed_data/mun_adc_pixel_counts.csv")
adc_pixel_counts = adc_pixel_counts.loc[adc_pixel_counts['year'] == 2007]
adc_pixel_counts['muncode'] = adc_pixel_counts['muncode'].apply(int)
adc_pixel_counts['n_pixel'] = adc_pixel_counts['n_pixel'].apply(literal_eval)

hist_df = pd.merge(hist_df,adc_pixel_counts[['muncode','n_pixel']],
                   on='muncode')

set_random_seed(1364) # random.org Min: 1, Max: 10000 2023-01-30 03:58:35 UTC

muncodes = hist_df[['muncode']].drop_duplicates()
muncodes['fold'] = np.random.choice(range(0,5),len(muncodes.index),
                                   replace=True)
hist_df = pd.merge(hist_df,muncodes,
                    on='muncode')
hist_df.to_csv("../processed_data/ndvi_nn_sample.csv",index=False)

#train_df = hist_df.loc[hist_df['fold']!=0]
#val_df = hist_df.loc[hist_df['fold']==0]
train_df = hist_df.loc[hist_df['year']!=holdout_year]
val_df = hist_df.loc[hist_df['year']==holdout_year]

print(f"Train Mean: {train_df['yield'].mean()}")
print(f"Train SD: {train_df['yield'].std()}")

scale_df = pd.DataFrame({
    'mean':[train_df['yield'].mean()],
    'sd':[train_df['yield'].std()]
})

scale_df.to_csv("../processed_data/nn_rescale.csv",index=False)

val_df['yield'] = (val_df['yield']-train_df['yield'].mean())/train_df['yield'].std()
train_df['yield'] = (train_df['yield']-train_df['yield'].mean())/train_df['yield'].std()

class hist_batching(Sequence):

        def __init__(self,sample,batch_size,bins,seed=42):

            np.random.seed(seed)
            # Assign batches by file
            sample['batch'] = np.random.choice(range(0,len(sample.index)),
            len(sample.index),
            replace=False)
            sample.loc[:,'batch'] = sample['batch']//batch_size
            
            
            sample.loc[:,'small_n'] = sample['n_pixel'].apply(lambda x: np.random.choice(x,1)[0])
            sample.loc[:,'small_hist'] = sample.apply(lambda x: np.concatenate([return_adc_sample_mhist(np.reshape(x['hist'][:,:,c],(bins, bins,1)),int(x['small_n'])) for c in range(0,3)],axis=-1),axis=1)
            #sample.loc[:,'small_hist'] = sample.apply(lambda x: return_adc_sample_mhist(x['hist'],int(x['small_n'])) ,axis=1)

            
            self.sample = sample.copy()
            self.bins = bins
            self.bs = batch_size

        def __len__(self):
            return(self.sample['batch'].max()+1)

        def __getitem__(self,idx):

            batch_df = self.sample.loc[self.sample['batch']==idx]
            
            
            X = np.array(batch_df['hist'].tolist())
            X = 10000*X
            Y = batch_df['yield'].to_numpy()
            #print(f"\n {np.mean(Y)}")
            Y = np.expand_dims(Y,1)
            return X,Y

        def on_epoch_end(self):
            sample = self.sample
            sample.loc[:,'batch'] = np.random.choice(range(0,len(sample.index)),
            len(sample.index),
            replace=False)
            sample.loc[:,'batch'] = sample['batch']//self.bs

            sample.loc[:,'small_n'] = sample['n_pixel'].apply(lambda x: np.random.choice(x,1)[0])
            sample.loc[:,'small_hist'] = sample.apply(lambda x: np.concatenate([return_adc_sample_mhist(np.reshape(x['hist'][:,:,c],(bins, bins,1)),int(x['small_n'])) for c in range(0,3)],axis=-1),axis=1)
            #sample.loc[:,'small_hist'] = sample.apply(lambda x: return_adc_sample_mhist(x['hist'],int(x['small_n'])) ,axis=1)
            
            self.sample = sample


lr = 0.01
def lr_schedule(epoch,lr):
    return(lr*0.96)

reduce_lr = LearningRateScheduler(lr_schedule)

opt = keras.optimizers.Adam(learning_rate=lr)


es = EarlyStopping(patience=10)

ls25nn = ls_nn(bins,f"hist_nn_year{holdout_year}")
print(ls25nn.summary())
ls25nn.compile(loss = keras.losses.MeanSquaredError(),optimizer = opt)
mcp = ModelCheckpoint(f"../processed_data/ndvi_b{bins}_year{holdout_year}_lr{lr}.h5",save_best_only=True)


print(f"Val MSS: {((val_df['yield']-val_df['yield'].mean())**2).mean()}")


ls25nn.fit(hist_batching(train_df,30,bins),
validation_data=hist_batching(val_df,30,bins),
             verbose=1,
             epochs=200, shuffle=False,
             callbacks=[mcp,reduce_lr])


p install -q tensorflow tensorflow-datasets
import matplotlib.pyplot as plt
import numpy as np

#IMPORTING NECESSARY LIBRARIES
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow import keras
tfds.disable_progress_bar()
import pandas as pd
from scipy.fft import fft, ifft
from scipy.fftpack import irfft, rfft

#IMPORTING THE DATASET pure.xlsx
from google.colab import files
uploaded = files.upload()
df = pd.read_excel('pure.xlsx',header=None)
pure_acc = df.to_numpy()

#IMPORTING THE DATASET noisy.xlsx
from google.colab import files
uploaded = files.upload()
df = pd.read_excel('noisy.xlsx',header=None)
noisy_acc = df.to_numpy()

#IMPORTING THE DATASET test_pure.xlsx
from google.colab import files
uploaded = files.upload()
df = pd.read_excel('test_pure.xlsx',header=None)
test_pure_acc = df.to_numpy()

#IMPORTING THE DATASET test_noisy.xlsx
from google.colab import files
uploaded = files.upload()
df = pd.read_excel('test_noisy.xlsx',header=None)
test_noisy_acc = df.to_numpy()

#RANDOMIZING ORDER OF DATASET TO AVOID BIAS
n = np.linspace(0,noisy_acc.shape[0]-1,noisy_acc.shape[0],dtype=int)
#print(n)
np.random.shuffle(n)
#print(n)
temp_noisy = noisy_acc
temp_pure = pure_acc

for i in n:
  noisy_acc[i] = temp_noisy[n[i]]
  pure_acc[i] = temp_pure[n[i]]

#DATA PREPROCESSING, CONVERTING TRAINING DATA TO FREQUENCY DOMAIN
noisy_acc_freq = np.zeros((noisy_acc.shape[0],noisy_acc.shape[1]),np.float64)
for i in range(noisy_acc.shape[0]):
  noisy_acc_freq[i] = rfft(noisy_acc[i])

pure_acc_freq = np.zeros((pure_acc.shape[0],pure_acc.shape[1]),np.float64)
for i in range(pure_acc.shape[0]):
  pure_acc_freq[i] = rfft(pure_acc[i])

#RESHAPING THE NOISY ACCELEROMETER ARRAY
noisy_acc_freq = noisy_acc_freq.reshape(noisy_acc_freq.shape[0],noisy_acc_freq.shape[1],1)
print(noisy_acc_freq.shape)

#MODELING THE CNN FOR DENOISING
# linear activations are used as unlike regression it is a pattern to pattern matching
model = keras.Sequential([
   keras.layers.ZeroPadding1D(padding=3),
   keras.layers.Conv1D(16, 7, strides=1, activation='linear'),
   keras.layers.ZeroPadding1D(padding=8),
   keras.layers.Conv1D(32, 3, strides=1, activation='linear'),
   keras.layers.Conv1D(32, 3, strides=1, activation='linear'),
   keras.layers.Conv1D(32, 3, strides=1, activation='linear'),
   keras.layers.Conv1D(16, 3, strides=1, activation='linear'),
   keras.layers.Conv1D(16, 3, strides=1, activation='linear'),
   keras.layers.Conv1D(16, 3, strides=1, activation='linear'),
   keras.layers.Flatten(),
   keras.layers.Dense(16, activation='linear'),
   keras.layers.Dense(pure_acc_freq.shape[1], activation=None)
])

optim = tf.keras.optimizers.Adam(3e-4)

model.compile(optimizer=optim,
              loss = 'mse',
              metrics=[tf.keras.metrics.RootMeanSquaredError('rmse')])

model.fit(noisy_acc_freq, pure_acc_freq, epochs=100, batch_size=16)
model.summary()

#loss: 0.2855 - rmse: 0.5343
#MODELING THE ANN FOR DENOISING
# linear activations are used as unlike regression it is a pattern to pattern matching
model = keras.Sequential([
   keras.layers.Flatten(),
   keras.layers.Dense(4096, activation='linear'),
   keras.layers.Dense(8192, activation='linear'),
   keras.layers.Dense(4096, activation='linear'),
   keras.layers.Dense(2048, activation='linear'),
   keras.layers.Dense(pure_acc_freq.shape[1], activation=None)
])

optim = tf.keras.optimizers.SGD(1e-3)
#the momentum aspect of Adam caused it to spiral out of control

model.compile(optimizer=optim,
              loss = 'mse',
              metrics=[tf.keras.metrics.RootMeanSquaredError('rmse')])

model.fit(noisy_acc_freq, pure_acc_freq, epochs=100, batch_size=12)
model.summary()

#MODELLING THE RNN FOR DENOISING
#2 layer deep RNN/LSTM for many-to-many sequence matching
model = tf.keras.models.Sequential([
  tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)),
  tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)),
  tf.keras.layers.Dense(1, activation=None)
])

optimizer = tf.keras.optimizers.SGD(lr=1e-3, momentum=1)
model.compile(loss=tf.keras.losses.Huber(),
              optimizer=optimizer,
              metrics=[tf.keras.metrics.RootMeanSquaredError('rmse')])
model.fit(noisy_acc_freq, pure_acc_freq.reshape(pure_acc_freq.shape[0],pure_acc_freq.shape[1],1), epochs=30, batch_size=16)

#SAVING THE MODEL
model.save('cnn.h5') 
new_model = tf.keras.models.load_model('cnn.h5')
new_model.summary()

#COMPARING THE PURE, NOISY AND DENOISED SIGNALS USING MATPLOTLIB
z = test_noisy_acc[1038]
z= rfft(z)
z = z.reshape(1,z.shape[0],1)
y_denoised = new_model.predict(z)
y_denoised = irfft(y_denoised)
from matplotlib.pyplot import figure
figure(num=None, figsize=(10, 6), dpi=80, facecolor='w', edgecolor='k')
x = np.linspace(start=0,stop=7,num=701)
y_noisy = test_noisy_acc[1038]
plt.plot(x,y_noisy)				#NOISY SIGNAL
figure(num=None, figsize=(10, 6), dpi=80, facecolor='w', edgecolor='k')
plt.plot(x,pure_acc[1038])		#PURE SIGNAL
figure(num=None, figsize=(10, 6), dpi=80, facecolor='w', edgecolor='k')
plt.plot(x,y_denoised.reshape(701))		#DENOISED SIGNAL

#NOW THAT OUR ML ALGORITHM IS WORKING FINE, WE WILL MOVE ON TO EVALUATING IT USING [IO,LS,CP]
#CALCULATING IDR BY CALCULATING RELATIVE DISPLACEMENTS BY DOUBLE INTEGRATION. AND FINALLY CREATING [IO,LS,CP] ARRAY FOR FULL PURE #DATASET
time = 1/360 #time taken between 2 readings. Sampling rate = 360Hz

pure_classification = np.zeros((int(test_pure_acc.shape[0]*0.5),3),np.float64)

v = np.zeros(test_pure_acc.shape,np.float64) #velocity
disp = np.zeros(test_pure_acc.shape,np.float64) #displacement
floor_height = 2.00


for i in range(0,test_pure_acc.shape[0]):
  for j in range(1,test_pure_acc.shape[1]):   #double integration.
      v[i][j] = v[i][j-1] + (((test_pure_acc[i][j-1]+test_pure_acc[i][j])/2) * (time))

for i in range(0,test_pure_acc.shape[0]):
  for j in range(1,test_pure_acc.shape[1]):   #double integration.
      disp[i][j] = disp[i][j-1] + (((v[i][j-1]+v[i][j])/2) * (time))

for i in range(0,disp.shape[0],2): 
  idr = np.zeros(disp.shape[1],np.float64)

  for j in range(disp.shape[1]):
    idr[j] = ( np.abs(disp[i][j]-disp[i+1][j]) )/(floor_height)

  '''
  if idr < 0.007       => Immediate Occupancy
  if idr 0.007 to 0.05 => Life Safety
  if idr >0.05         => Collapse prevention
  '''
  scores=np.array([0,0,0])
  '''
  io_score=0th index
  ls_score=1st index
  cp_score=2nd index
  '''
  for k in range (idr.shape[0]):
    if idr[k]<0.007:
      scores[0]+=1
    elif idr[k]>0.05:
      scores[2]+=1
    else:
      scores[1]+=1

  #most sever score is considered for labeling the dataset
  if scores[2]>0:
    scores = [0,0,1]
  elif scores[1]>0:
    scores = [0,1,0]
  else:
    scores = [1,0,0]

  # scores = np.floor(scores/(np.amax(scores)))
  pure_classification[int(i/2)]=scores
  scores=np.array([0,0,0])

#FOR NOISY CLASSIFICATION

time = 1/360 #time taken between 2 readings. Sampling rate = 360Hz

noisy_classification = np.zeros((int(test_noisy_acc.shape[0]*0.5),3),np.float64)

v = np.zeros(test_noisy_acc.shape,np.float64) #velocity
disp = np.zeros(test_noisy_acc.shape,np.float64) #displacement
floor_height = 2.00


for i in range(0,test_noisy_acc.shape[0]):
  for j in range(1,test_noisy_acc.shape[1]):   #double integration.
      v[i][j] = v[i][j-1] + (((test_noisy_acc[i][j-1]+test_noisy_acc[i][j])/2) * (time))

for i in range(0,test_noisy_acc.shape[0]):
  for j in range(1,test_noisy_acc.shape[1]):  
      disp[i][j] = disp[i][j-1] + (((v[i][j-1]+v[i][j])/2) * (time))

for i in range(0,disp.shape[0],2): 
  idr = np.zeros(disp.shape[1],np.float64)

  for j in range(disp.shape[1]):
    idr[j] = ( np.abs(disp[i][j]-disp[i+1][j]) )/(floor_height)

  '''
  if idr < 0.007       => Immediate Occupancy
  if idr 0.007 to 0.05 => Life Safety
  if idr >0.05         => Collapse prevention
  '''
  scores=np.array([0,0,0])
  '''
  io_score=0th index
  ls_score=1st index
  cp_score=2nd index
  '''
  for k in range (idr.shape[0]):
    if idr[k]<0.007:
      scores[0]+=1
    elif idr[k]>0.05:
      scores[2]+=1
    else:
      scores[1]+=1
  #most sever score is considered for labeling the dataset
  if scores[2]>0:
    scores = [0,0,1]
  elif scores[1]>0:
    scores = [0,1,0]
  else:
    scores = [1,0,0]

  # scores = np.floor(scores/(np.amax(scores)))
  noisy_classification[int(i/2)]=scores
  scores=np.array([0,0,0])


#calculating accuracy without denoising.

b=0
for i in range(pure_classification.shape[0]):
  if pure_classification[i][0] == noisy_classification[i][0] and pure_classification[i][1] == noisy_classification[i][1] and pure_classification[i][2] == noisy_classification[i][2] :
    b+=1

print("accuracy = ",end="")
print(np.float64(b)/pure_classification.shape[0])

#DENOISING TEST_NOISY_ACC
test_denoised_acc = np.zeros((test_noisy_acc.shape[0],test_noisy_acc.shape[1]),np.float64)
for i in range(test_noisy_acc.shape[0]):
  z = rfft(test_noisy_acc[i])
  z = z.reshape(1,701,1)
  test_denoised_acc[i] = irfft(new_model.predict(z))

#GETTING CLASSIFICATION ARRAY FOR DENOISED SIGNALS
time = 1/360 #time taken between 2 readings. Sampling rate = 360Hz

denoised_classification = np.zeros((int(test_denoised_acc.shape[0]*0.5),3),np.float64)

v = np.zeros(test_denoised_acc.shape,np.float64) #velocity
disp = np.zeros(test_denoised_acc.shape,np.float64) #displacement
floor_height = 2.00


for i in range(0,test_denoised_acc.shape[0]):
  for j in range(1,test_denoised_acc.shape[1]):   #double integration.
      v[i][j] = v[i][j-1] + (((test_denoised_acc[i][j-1]+test_denoised_acc[i][j])/2) * (time))

for i in range(0,test_denoised_acc.shape[0]):
  for j in range(1,test_denoised_acc.shape[1]):   #double integration.
      disp[i][j] = disp[i][j-1] + (((v[i][j-1]+v[i][j])/2) * (time))

for i in range(0,disp.shape[0],2): 
  idr = np.zeros(disp.shape[1],np.float64)

  for j in range(disp.shape[1]):
    idr[j] = ( np.abs(disp[i][j]-disp[i+1][j]) )/(floor_height)

  '''
  if idr < 0.007       => Immediate Occupancy
  if idr 0.007 to 0.05 => Life Safety
  if idr >0.05         => Collapse prevention
  '''
  scores=np.array([0,0,0])
  '''
  io_score=0th index
  ls_score=1st index
  cp_score=2nd index
  '''
  for k in range (idr.shape[0]):
    if idr[k]<0.007:
      scores[0]+=1
    elif idr[k]>0.05:
      scores[2]+=1
    else:
      scores[1]+=1
  
  if scores[2]>0:
    scores = [0,0,1]
  elif scores[1]>0:
    scores = [0,1,0]
  else:
    scores = [1,0,0]

  # scores = np.floor(scores/(np.amax(scores)))
  denoised_classification[int(i/2)]=scores
  scores=np.array([0,0,0])

#calculating accuracy with denoising.

b=0
for i in range(pure_classification.shape[0]):
  if pure_classification[i][0] == denoised_classification[i][0] and pure_classification[i][1] == denoised_classification[i][1] and pure_classification[i][2] == denoised_classification[i][2] :
    b+=1

print("accuracy = ",end="")
print(np.float64(b)/pure_classification.shape[0])



#####DIRECT CLASSIFICATION USING ACCELEROMETER DATA

#RELATIVE ACCELERATION CALCULATION
rel_acc_noisy = np.zeros((int(noisy_acc.shape[0]*0.5),noisy_acc.shape[1]),dtype=np.float64)
rel_acc_pure = np.zeros((int(pure_acc.shape[0]*0.5),pure_acc.shape[1]),dtype=np.float64)

a=0

for i in range(rel_acc_pure.shape[0]):
  rel_acc_noisy[i] = noisy_acc[a+1] - noisy_acc[a]
  rel_acc_pure[i]  = pure_acc[a+1] - pure_acc[a]
  a+=2
rel_acc_noisy = rel_acc_noisy.reshape(rel_acc_noisy.shape[0],rel_acc_noisy.shape[1],1)

### Getting the pure classifications from pure_acc
#For relative acceleration calculation we assume 2 simultaneous accelerometer-time signals

time = 1/360 #time taken between 2 readings. Sampling rate = 360Hz

pure_classification2 = np.zeros((int(pure_acc.shape[0]*0.5),3),np.float64)

v = np.zeros(pure_acc.shape,np.float64) #velocity
disp = np.zeros(pure_acc.shape,np.float64) #displacement
floor_height = 2.00


for i in range(0,pure_acc.shape[0]):
  for j in range(1,pure_acc.shape[1]):   #double integration.
      v[i][j] = v[i][j-1] + (((pure_acc[i][j-1]+pure_acc[i][j])/2) * (time))

for i in range(0,pure_acc.shape[0]):
  for j in range(1,pure_acc.shape[1]):   #double integration.
      disp[i][j] = disp[i][j-1] + (((v[i][j-1]+v[i][j])/2) * (time))

for i in range(0,disp.shape[0],2): 
  idr = np.zeros(disp.shape[1],np.float64)

  for j in range(disp.shape[1]):
    idr[j] = ( np.abs(disp[i][j]-disp[i+1][j]) )/(floor_height)

  '''
  if idr < 0.007       => Immediate Occupancy
  if idr 0.007 to 0.05 => Life Safety
  if idr >0.05         => Collapse prevention
  '''
  scores=np.array([0,0,0])
  '''
  io_score=0th index
  ls_score=1st index
  cp_score=2nd index
  '''
  for k in range (idr.shape[0]):
    if idr[k]<0.007:
      scores[0]+=1
    elif idr[k]>0.05:
      scores[2]+=1
    else:
      scores[1]+=1
  
  if scores[2]>0:
    scores = [0,0,1]
  elif scores[1]>0:
    scores = [0,1,0]
  else:
    scores = [1,0,0]

  # scores = np.floor(scores/(np.amax(scores)))
  pure_classification2[int(i/2)]=scores
  scores=np.array([0,0,0])

print(pure_classification2.shape)

###CNN FOR DIRECT CLASSIFICATION

#loss: 0.2265 - accuracy: 89.11
model = keras.Sequential([
   keras.layers.AveragePooling1D(1,1,input_shape = (701,1)),
   keras.layers.Conv1D(32, 32, strides=1, activation='relu'),
   keras.layers.Conv1D(32, 32, strides=1, activation='relu'),
   keras.layers.MaxPool1D(pool_size=8),
   keras.layers.Flatten(),
   keras.layers.Dense(512, activation='relu'),
   keras.layers.Dense(512, activation='relu'),
   keras.layers.Dense(3, activation='softmax')
])

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(),
              metrics=['accuracy'])

model.fit(rel_acc_noisy, pure_classification2, epochs=5, batch_size=16)

#ANN FOR DIRECT CLASSIFICATION

#loss: 0.4211- accuracy: 0.7067
model = keras.Sequential([
   keras.layers.Flatten(),
   keras.layers.Dense(512, activation='relu'),
   keras.layers.Dense(512, activation='relu'),
   keras.layers.Dense(512, activation='relu'),
   keras.layers.Dense(3, activation='softmax')
])

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(),
              metrics=['accuracy'])

model.fit(rel_acc_noisy, pure_classification2, epochs=5, batch_size=4)

e = np.zeros((int(test_noisy_acc.shape[0]*0.5),test_noisy_acc.shape[1]),dtype=np.float64)
#e is the relative acceleration for the test_noisy_acc
a=0

for i in range(e.shape[0]):
  e[i] = test_noisy_acc[a+1] - test_noisy_acc[a]
  a+=2
e = e.reshape(e.shape[0],e.shape[1],1)

#EVALUATING MODELS
model.evaluate(e,pure_classification)
model.evaluate(rel_acc_noisy,pure_classification2)
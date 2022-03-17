import numpy as np
import sys
sys.path
sys.executable
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from numpy import absolute
from numpy import mean
from numpy import std
from sklearn.datasets import make_regression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from tensorflow import keras
import pandas as pd 
import csv
from csv import reader
import simplejson
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.layers import Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from scipy.stats import zscore
from tensorflow.keras.callbacks import EarlyStopping
from matplotlib.pyplot import figure, show
N1 = 10 #col11 omegaR
N2 = 12 #col13 vel
output_folder = '/home/sci06/chenda_inverseDegisn/result/lr3_init_newloss_adamno0_select1w_5layer_mae/'
myname = 'lr3_init_newloss_adamno0_select1w_5layer_mae'
df = pd.read_csv('/home/sci06/chenda_inverseDegisn/2tails_sub.csv')#2tails_Sim_added.csv includes the data to be tested.
df = df.sample(frac=1).reset_index(drop=True)
dflen = len(df)
fortest = df[int(dflen * 0.95):]
df = df[:int(dflen * 0.95)]
fortest.to_csv(output_folder + "testingset.csv", index = False, sep=',', mode='a',columns=None)
df.to_csv(output_folder + "trainingset.csv", index = False, sep=',', mode='a',columns=None)
#this takes the first row
df.columns = ['col1','col2','col3','col4','col5','col6','col7','col8','col9','col10','col11','col12','col13','col14'] 
x = df[['col1','col2','col3','col4','col5','col6','col7','col8','col9','col10','col11','col12','col13','col14']]
xinput = df[['col1','col2','col3','col4','col5','col6','col7','col8']]
xoutput = df[['col10','col11','col12','col13','col14']]
# print(np.array(xoutput))
xinput = np.array(xinput)


# print(xoutput)
x = np.array(x)
x9 = df['col9']#omegahead
x9 = abs(np.array(x9)) 
x9 = np.reshape(x9, (x9.shape[0], 1))
# print(x9)
# print(x9.shape, np.array(xoutput).shape)
xoutput = np.hstack((x9, np.array(xoutput)))
# print(xoutput)


#print(xinput.shape[0])#checked
xnew = np.hstack((xinput, xoutput))
# print(xnew.shape)
# print(xnew)

# ##################Step 4 Multioutput regression with k-fold cross-validation
# ## get datasets from nondimensionalized variables
# L = xnew[:, 3] # tail length
# R = xnew[:, 4] # head radius
# r0 = xnew[:, 6] # tail radius
# x4 = L/R # element-wise division, checked
# x5 = L/r0
# eta_per = 4*np.pi*1/(np.log(2*x5) + 1/2)
# EI = 1e6*np.pi*r0**4/4
# x6 = xnew[:, 7]*eta_per*L**4/EI

# x4 = np.reshape(x4,(x4.shape[0],1))
# x5 = np.reshape(x5,(x5.shape[0],1))
# x6 = np.reshape(x6,(x6.shape[0],1))
# xdata = np.hstack((addTailNum, xnew[:,:3], x4, x5, x6))
xdata = xnew[:,:8]
# print(xdata.shape)
# print(xnew[:,8])
ydata2 = xnew[:,N2]
# ydata2 = xnew[:,N2]*eta_per*L**3/EI #this is only for N2=12 vel
ydata2 = np.reshape(ydata2,(ydata2.shape[0],1))
ydata = ydata2
# print(ydata.shape)

#Dividing the dataset
X_train, X_test, y_train, y_test = train_test_split(
    xdata, ydata, train_size=0.88, test_size=0.12)

#Preprocess the training data: normalization
mean1 = X_train.mean(axis=0)
X_train -= mean1 #X_train -= np.mean(X_train, axis=0) #cal the average of each column
std1 = X_train.std(axis = 0)
X_train /= std1

###########!!!test data should use the mean of training data too, not test data!!!!!!!
X_test -= mean1
X_test /= std1
spec_file = open(output_folder+"spec_file_"+myname+ ".txt",'w+')
out_mean = [str(float(i)) for i in list(mean1)]
out_std = [str(float(i)) for i in list(std1)]
spec_file.write(','.join(out_mean)+'\n' + ','.join(out_std)+'\n')
spec_file.flush()
spec_file.close()

from tensorflow.keras.backend import int_shape
def my_upper_loss(arraylength):
    def my_loss_fn(y_true, y_pred):
        ylen = arraylength
        print(ylen)
        y_true_0 = y_true[0:int(0.2*ylen)]
        y_pred_0 = y_pred[0:int(0.2*ylen)]
        y_true_1 = y_true[int(0.2*ylen):int(0.8*ylen)]
        y_pred_1 = y_pred[int(0.2*ylen):int(0.8*ylen)]
        y_true_2 = y_true[int(0.8*ylen):]
        y_pred_2 = y_pred[int(0.8*ylen):]
        # mseloss = np.mean(np.abs(y_true_rest - y_pred_rest), axis=-1)
        mae = keras.losses.MeanAbsoluteError()
        maeloss0 = mae(y_true_0, y_pred_0)
        maeloss1 = mae(y_true_1, y_pred_1)
        maeloss2 = mae(y_true_2, y_pred_2)
        return 3 * maeloss0 + maeloss1 + 3 * maeloss2 # Note the `axis=-1`
    return my_loss_fn

def build_model(arraylength):
    model = Sequential()
    initializer = keras.initializers.VarianceScaling(scale=0.1, mode='fan_in', distribution='uniform')
    model.add(Dense(100, activation='relu', input_shape=(X_train.shape[1],),kernel_initializer = initializer))
    model.add(Dense(100, activation='relu',kernel_initializer = initializer))#
    model.add(Dense(100, activation='relu',kernel_initializer = initializer))
    model.add(Dense(100, activation='relu',kernel_initializer = initializer))   
    model.add(Dense(100, activation='relu',kernel_initializer = initializer))  
    model.add(Dense(1, kernel_initializer='normal'))#output number
#     earlyStopping=keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, verbose=0, mode='auto')
#     model.compile(loss='mse', optimizer='adam')#https://keras.io/zh/getting-started/sequential-model-guide/, 
    model.compile(optimizer='adam', loss= my_upper_loss(arraylength = arraylength), metrics = ['mae'])
    return model


#evaluate the model on test dataset, this is only for small dataset

# batch size is related to the threads, so the larger the better, especially for the second-order gradient, normally
#they are set as 16、32、64、128 better than 10, 100, 1000

#  (1) batchsize, SGD is normally used in deep learning，this means to grab batchsize datapts for each training
# （2）iteration：1 iteration means the forward ad back propogation once using batchsize numbers
# （3）epoch：one epoch means using all the numbers in the training set to train

BATCH_SIZE = 32
FOLDS = 30
# STOPPING_PATIENCE = 32
# LR_PATIENCE = 16
# INITIAL_LR = 0.0001

num_val_samples = len(X_train) // FOLDS

num_epochs = 500
# Train the model for 500 epoches first
from tensorflow.keras import backend as K
K.clear_session()
all_mae_histories = []
all_train_mae_histories = []
print("start training!!!!!!!!!!!!!!!!!")
for i in range(FOLDS):
    print('processing fold #', i)
    val_data = X_train[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = y_train[i * num_val_samples: (i + 1) * num_val_samples]
    x_0 = [[0,0,0,0,0,0,0,0]]
    y_0 = [[0]]
    partial_train_data = np.concatenate(
        [X_train[:i * num_val_samples],
        X_train[(i + 1) * num_val_samples:]],
        axis=0)
    partial_train_targets = np.concatenate(
        [y_train[:i * num_val_samples],
        y_train[(i + 1) * num_val_samples:]],
        axis=0)
    partial_train_data = partial_train_data[partial_train_targets[:,0].argsort()]
    partial_train_targets = partial_train_targets[partial_train_targets[:,0].argsort()]
    val_data = val_data[val_targets[:,0].argsort()]
    val_targets = val_targets[val_targets[:,0].argsort()]
    length = partial_train_targets.shape[0]
    model = build_model(length)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.3, patience=5, min_lr=0.000001)
    history = model.fit(partial_train_data, partial_train_targets,
                        validation_data=(val_data, val_targets),
                        epochs=num_epochs, batch_size=BATCH_SIZE, verbose=1,callbacks=[reduce_lr])
    mae_history = history.history['val_mae']
    train_mae_history = history.history['mae']
    all_mae_histories.append(mae_history)
    all_train_mae_histories.append(train_mae_history)
print("finish training!!!!!!!!!!!!!!!!!")
print("starting saving model")
model.save(output_folder + 'model_' + myname) 
print("model saved successfully")
average_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]
average_train_mae_history = [np.mean([x[i] for x in all_train_mae_histories]) for i in range(num_epochs)]
import matplotlib.pyplot as plt
plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAPE')
plt.savefig(output_folder + 'validation_'+myname+'.png')
plt.clf()
plt.plot(range(1, len(average_train_mae_history) + 1), average_train_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Training MAPE')
plt.savefig(output_folder + 'training_'+myname+'.png')
plt.clf()
print(np.min(average_mae_history))
print(np.std(average_mae_history))
print("average and std")
print(mean1)
print(std1)
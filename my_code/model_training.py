#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf
import numpy as np
import pandas as pd
import datetime
from modules.TFR_load import TFR_load
from models.Classification import Classification
from models.DenseNet import DenseNet


# In[3]:


# global_whitening
'''
BATCH_SIZE = 1
NUM_TRAIN_DATA = 891

path = './data/TFRs/train/'
ds = TFR_load(
    path,
    BATCH_SIZE,
    NUM_TRAIN_DATA,
    is_training = False,
    data_shape = [5]
)
_data = []
for x, y in ds.take(1):
    _data.append(x)
_data = sum(_data)

global_whitening_factor = _data / NUM_TRAIN_DATA
'''
#global_whitening_factor
gwf= np.array([[0.003367  , 0.00112233, 0.03928171, 0.00886173, 0.00224467]],
      dtype=np.float32)
#global_max
gmax = np.array([[  3.    ,   1.    ,  80.    , 512.3292,   4.    ]],
      dtype=np.float32)
#global_min
gmin= np.array([[ 1.,  0., -1.,  0.,  1.]],
      dtype=np.float32)


# In[4]:


data = pd.read_csv("./data/original_data/test.csv")
data["Age"] = data["Age"].fillna(-1)
data = data.drop(['Name','SibSp','Parch'], axis = 1)
data = data.drop(['Ticket','Cabin'], axis = 1)

def fn(x):
    if x == 'male':
        return 1
    else:
        return 0

data['Sex'] = data['Sex'].map(fn)

data['Embarked'] = data['Embarked'].fillna('N')

def fn(x):
    if x == 'S':
        return 1
    elif x == 'C':
        return 2
    elif x == 'Q':
        return 3
    elif x == 'N':
        return 4
    
data['Embarked'] = data['Embarked'].map(fn)

data.head()


# In[5]:


BATCH_SIZE = 32
NUM_TRAIN_DATA = 891
LEARNING_RATE = 1e-4
SCHEDULER = None
EPOCH = 50


# In[6]:


path = './data/TFRs/train/'
ds = TFR_load(
    path,
    BATCH_SIZE,
    NUM_TRAIN_DATA,
    is_training = False,
    data_shape = [5]
)


# In[7]:


#model = Classification()
model = DenseNet()


# In[8]:


#model.trace_graph([32,5])


# In[9]:


#optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, epsilon=1e-07)

optimizer = tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE,momentum=0.9)
loss = tf.keras.losses.BinaryCrossentropy()
tr_accuracy = tf.keras.metrics.BinaryAccuracy()


# In[ ]:





# In[10]:


def whitening(x, gwf, gmin, gmax):
    x -= gwf
    gmin -= gwf
    gmax -= gwf
    
    return (x - gmin) / (gmax - gmin) * 2 -1


# In[11]:


hist_log = True

for birth in range(10):
    
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    
    step = 1
    for EP in range(EPOCH):
        EP += 1

        for x, y in ds:
        
            # Train loop
            loss_value, acc = 0, 0
            with tf.GradientTape() as tape:
                logits = model(whitening(x, gwf, gmin, gmax), training = True)
                for i, layer in enumerate(model.layers):
                    loss_value += tf.math.reduce_sum(layer.losses)
                loss_value /= (i+1)
                loss_value += loss(y, logits)
            
                tr_accuracy.update_state(y, logits)
                acc += tr_accuracy.result()
            grads = tape.gradient(loss_value, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            if SCHEDULER == 'custom':
                if step in [32000, 48000]:
                    optimizer.learning_rate = 0.1*optimizer.learning_rate.numpy()

            #Train log
            with train_summary_writer.as_default():
                tf.summary.scalar('loss', loss_value, step=step)
                tf.summary.scalar('acc', acc, step=step)
                tr_accuracy.reset_states()

                if hist_log:
                    # histogram
                    for w in model.weights:
                        if "batch_normalization" in w.name:
                            tf.summary.histogram(
                                "batch_normalization/" + w.name, w, step=step)
                        elif "conv2d" in w.name:
                            tf.summary.histogram("conv2d/" + w.name, w, step=step)
                        elif "dense" in w.name:
                            tf.summary.histogram("dense/" + w.name, w, step=step)
                        else:
                            tf.summary.histogram(w.name, w, step=step)
            
            step += 1
            
        PID, Survived = [], []
        for i in range(len(data)):

            inputs = []
            for k, row in enumerate(data.loc[i]):
                if not k :
                    PID.append(int(row))
                else:
                    inputs.append(float(row))
            inputs = np.array([inputs]).astype(np.float32)

            logits = model(whitening(x, gwf, gmin, gmax), training = True)
            if logits[0][0] > logits[0][1]:
                Survived.append(0)
            elif logits[0][0] <= logits[0][1]:
                Survived.append(1)


        pd.DataFrame({"PassengerId" : PID, "Survived" : Survived}).to_csv(f'./result/{birth}_{EP}_{step}.csv', mode='w',index = False)


# In[ ]:





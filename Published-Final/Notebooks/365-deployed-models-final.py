#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf


# In[3]:


import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, precision_recall_fscore_support
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from sklearn.metrics import precision_score, recall_score, f1_score


# In[40]:


from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.compose import make_column_transformer, make_column_selector


# ## XGBoost All Features

# In[4]:


data=pd.read_csv('df_student_features_entiredata.csv')
data.drop(columns=['Unnamed: 0','subscriptiontype','onboard_flag','daysafterpur'],inplace=True)
data.info()


# In[5]:


data.to_csv('365_Final_Features_deployed.csv')


# In[25]:


import xgboost as xgb
y=data['conversion']
X=data.drop(columns=['conversion'])
ct=ColumnTransformer([('ohecc',OneHotEncoder(sparse=False),['student_country'])],remainder='passthrough')
X=ct.fit_transform(X)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,shuffle=True)
xgfin=xgb.XGBClassifier(use_label_encoder=False,eval_metric='logloss')
xgfin.fit(X_train,y_train)
y_pred=xgfin.predict(X_test)
print("Confusion Matrix: \n",confusion_matrix(y_pred,y_test))
print("F1 Score: \n",f1_score(y_pred,y_test))
print("Balanced Accuracy Score: \n",balanced_accuracy_score(y_pred,y_test))
print("All Scores: \n",precision_recall_fscore_support(y_pred,y_test))


# In[11]:


data['conversion'].value_counts()


# ## Random Undersampler / Oversampler

# ### Random Undersampler

# In[10]:


from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
from imblearn.over_sampling import RandomOverSampler


# In[13]:


rus = RandomUnderSampler(random_state=123, replacement=True) 
X_train1,y_train1 = rus.fit_resample(X_train,y_train)
print('original dataset shape:', Counter(y_train))
print('Resample dataset shape', Counter(y_train1))


# In[26]:


xgrus=xgb.XGBClassifier(use_label_encoder=False,,eval_metric='logloss')
xgrus.fit(X_train1,y_train1)
y_pred=xgrus.predict(X_test)
cm=confusion_matrix(y_pred,y_test)
F1=f1_score(y_pred,y_test)
print("Confusion Matrix: \n",confusion_matrix(y_pred,y_test))
print("F1 Score: \n",f1_score(y_pred,y_test))
print("Balanced Accuracy Score: \n",balanced_accuracy_score(y_pred,y_test))
print("All Scores: \n",precision_recall_fscore_support(y_pred,y_test))


# ### Oversampler

# In[17]:


rus = RandomOverSampler(random_state=123) 
X_train2,y_train2 = rus.fit_resample(X_train,y_train)
print('original dataset shape:', Counter(y_train))
print('Resample dataset shape', Counter(y_train2))


# In[27]:


xgros=xgb.XGBClassifier(use_label_encoder=False,eval_metric='logloss')
xgros.fit(X_train2,y_train2)
y_pred=xgros.predict(X_test)
cm=confusion_matrix(y_pred,y_test)
F1=f1_score(y_pred,y_test)
print("Confusion Matrix: \n",confusion_matrix(y_pred,y_test))
print("F1 Score: \n",f1_score(y_pred,y_test))
print("Balanced Accuracy Score: \n",balanced_accuracy_score(y_pred,y_test))
print("All Scores: \n",precision_recall_fscore_support(y_pred,y_test))


# In[ ]:





# ## XGBoost - short form (4 features for Tableau deployment / dashboards)

# In[20]:


df=pd.read_csv('pd_data3.csv')
df.drop(columns='Unnamed: 0',inplace=True)
df.drop(columns=['onboard_flag'],inplace=True)
df.info()


# In[28]:


import xgboost as xgb
xgccfin1=xgb.XGBClassifier(use_label_encoder=False,eval_metric='logloss')
y=df['conversion']
X=df.drop(columns=['conversion'])
ct1=ColumnTransformer([('ohecc',OneHotEncoder(sparse=False),['student_country'])],remainder='passthrough')
X=ct1.fit_transform(X)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,shuffle=True)
 
xgccfin1.fit(X_train,y_train)
y_pred=xgccfin1.predict(X_test)


print("Confusion Matrix: \n",confusion_matrix(y_pred,y_test))
print("F1 Score: \n",f1_score(y_pred,y_test))
print("Balanced Accuracy Score: \n",balanced_accuracy_score(y_pred,y_test))
print("All Scores: \n",precision_recall_fscore_support(y_pred,y_test))


# In[30]:


def predict_student_xgccfin1(a1,a2,a3):
    
    import pandas as pd
    import numpy as np
    
    x=pd.DataFrame(np.array([[a1,a2,a3]]),columns=['student_country','totaldays_engaged','tendaysafter'])
    X_test=ct1.transform(x) 
    y_pred2=xgccfin1.predict(X_test)[0] 
    return y_pred2.tolist()


# In[31]:


def predict_prob_xgccfin1(a1,a2,a3):
    
    import pandas as pd
    import numpy as np
    
    x=pd.DataFrame(np.array([[a1,a2,a3]]),columns=['student_country','totaldays_engaged','tendaysafter'])
    X_test=ct1.transform(x) 
    y_pred2=xgccfin1.predict_proba(X_test)[0][1]
    
    print("ypred prob is ",y_pred2.tolist())
    return y_pred2.tolist()


# In[32]:


predict_prob_xgccfin1('IN',20,10)


# In[33]:


from tabpy.tabpy_tools.client import Client
client = Client('http://localhost:9004/')


# In[34]:


client.deploy('predict_student_xgccfin1',
predict_student_xgccfin1, 'Returns prediction of student conversion'
, override = True)


# In[35]:


client.deploy('predict_prob_xgccfin1',
predict_prob_xgccfin1, 'Returns prob prediction of student conversion'
, override = True)


# In[36]:


def predict_student_xgchart(a1,a2,a3):
     
    import pandas as pd
    import numpy as np   
    
    
    data={'student_country':a1,'totaldays_engaged':a2,'tendaysafter':a3}
    
    print(data)
    
    x=pd.DataFrame(data)
    X_test=ct1.transform(x)
    
    print(X_test.shape) 
    
    y_pred1=xgccfin1.predict(X_test)
    
    print("ypred is ",y_pred1)
     
    return y_pred1.tolist()


# In[37]:


client.deploy('predict_student_xgchart',
predict_student_xgchart, 'Returns prob prediction of student conversion'
, override = True)


# In[ ]:





# ## Tensorflow - All features

# In[38]:


data=pd.read_csv('df_student_features_entiredata.csv')
data.drop(columns=['Unnamed: 0','subscriptiontype','onboard_flag','daysafterpur'],inplace=True)
data.info()


# In[39]:


y=data['conversion']
X=data.drop(columns=['conversion'])


# In[41]:


ct1=make_column_transformer(
    (OneHotEncoder(sparse=False),make_column_selector(dtype_include=object)),
(StandardScaler(),make_column_selector(dtype_exclude=object)))


# In[42]:


X=ct1.fit_transform(X)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,shuffle=True)


# In[77]:


model_tf=tf.keras.models.Sequential([
    tf.keras.layers.Dense(30,activation='relu',activity_regularizer=tf.keras.regularizers.L2(10e-6)),
#     tf.keras.layers.Dense(30,activation='relu',activity_regularizer=tf.keras.regularizers.L1(10e-8)),
    tf.keras.layers.Dense(20,activation='relu'),
    tf.keras.layers.Dense(20,activation='relu'),
    tf.keras.layers.Dense(1,activation='sigmoid')
    
])

model_tf.compile(loss=tf.keras.losses.BinaryCrossentropy(),optimizer='Adam',metrics=[tf.keras.metrics.Precision()])#(class_id=1)])


# In[78]:


early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", # watch the val loss metric
                                                  patience=5)
 
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",  
                                                 factor=0.1,  
                                                 patience=5,
                                                 verbose=1,  
                                                 min_lr=1e-7)


# In[79]:


num_epochs=100
hist=model_tf.fit(X_train,y_train,epochs=num_epochs,verbose=1,
                  validation_data=(X_test,y_test),callbacks=[early_stopping])#,reduce_lr])


# In[83]:


y_pred=tf.round(model_tf.predict(X_test))
confusion_matrix(y_pred,y_test)


# In[81]:


print("Confusion Matrix: \n",confusion_matrix(y_pred,y_test))
print("F1 Score: \n",f1_score(y_pred,y_test))
print("Balanced Accuracy Score: \n",balanced_accuracy_score(y_pred,y_test))
print("All Scores: \n",precision_recall_fscore_support(y_pred,y_test))


# ### Tensorflow - short form - 3 - features - deployed in tableau

# In[84]:


df=pd.read_csv('pd_data3.csv')
df.drop(columns='Unnamed: 0',inplace=True)
df.drop(columns=['onboard_flag'],inplace=True)
df.info()


# In[85]:


y=df['conversion']
X=df.drop(columns=['conversion'])


# In[86]:


ct1=ColumnTransformer([('ohecc',OneHotEncoder(sparse=False),['student_country'])],remainder='passthrough')
X=ct1.fit_transform(X)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,shuffle=True)


# In[87]:


model_tf=tf.keras.models.Sequential([
    tf.keras.layers.Dense(30,activation='relu',activity_regularizer=tf.keras.regularizers.L2(10e-6)),
#     tf.keras.layers.Dense(30,activation='relu',activity_regularizer=tf.keras.regularizers.L1(10e-8)),
    tf.keras.layers.Dense(20,activation='relu'),
    tf.keras.layers.Dense(20,activation='relu'),
    tf.keras.layers.Dense(1,activation='sigmoid')
    
])

model_tf.compile(loss=tf.keras.losses.BinaryCrossentropy(),optimizer='Adam',metrics=[tf.keras.metrics.Precision()])#(class_id=1)])


# In[88]:


early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", # watch the val loss metric
                                                  patience=8) 
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",  
                                                 factor=0.1,  
                                                 patience=6,
                                                 verbose=1,  
                                                 min_lr=1e-7)


# In[89]:


num_epochs=20
hist=model_tf.fit(X_train,y_train,epochs=num_epochs,verbose=1,validation_data=(X_test,y_test))


# In[90]:


y_pred=tf.round(model_tf.predict(X_test))
confusion_matrix(y_pred,y_test)


# In[92]:


print("Confusion Matrix: \n",confusion_matrix(y_pred,y_test))
print("F1 Score: \n",f1_score(y_pred,y_test))
print("Balanced Accuracy Score: \n",balanced_accuracy_score(y_pred,y_test))
print("All Scores: \n",precision_recall_fscore_support(y_pred,y_test))


# In[37]:


# from keras_pickle_wrapper import KerasPickleWrapper
# twr=KerasPickleWrapper(model_tf)


# In[38]:


# twr().fit(X_train,y_train,epochs=20)


# In[94]:


def predict_student_keras1(a1,a2,a3):
    import pandas as pd
    import numpy as np
#     import tensorflow as tf
    # import pickle5
    
#     da1=pickle.dumps(twr,protocol=5)
    
#     tw=pickle.loads(da1)
    
    
    
    x=pd.DataFrame(np.array([[a1,a2,a3]]),columns=['student_country','totaldays_engaged','tendaysafter'])
    X_test=ct1.transform(x)    
     
    y_pred1=tf.math.round(model_tf.predict(tf.convert_to_tensor(X_test,dtype=tf.float64),verbose=0)[0])
    
    y_pred=tf.convert_to_tensor(y_pred1, dtype=tf.float32)
    
    print("Y pred is ", y_pred)
    
    return float(y_pred1.numpy()[0])


# In[95]:


predict_student_keras1("IN",10,20)


# In[96]:


client.deploy('predict_student_keras1',
predict_student_keras1, 'keras Returns prediction of student conversion'
, override = True)


# In[97]:


def predict_prob_keras1(a1,a2,a3):
    import pandas as pd
    import numpy as np
#     import tensorflow as tf
    # import pickle5
    
#     da1=pickle.dumps(twr,protocol=5)
    
#     tw=pickle.loads(da1)
    
    
    
    x=pd.DataFrame(np.array([[a1,a2,a3]]),columns=['student_country','totaldays_engaged','tendaysafter'])
    X_test=ct1.transform(x)    
     
    y_pred1=model_tf.predict(tf.convert_to_tensor(X_test,dtype=tf.float64),verbose=0)[0]
    
    print("yppred1 is  ",y_pred1)
    
    print("Y pred is ", y_pred1[0])
    
    y_pred=tf.convert_to_tensor(y_pred1, dtype=tf.float32)
    
    return float(y_pred.numpy()[0])


# In[98]:


predict_prob_keras1('DE',20,10)


# In[99]:


client.deploy('predict_prob_keras1',
predict_prob_keras1, 'keras Returns prediction of student conversion'
, override = True)


# ## Autoencoder - All features

# In[100]:


data=pd.read_csv('df_student_features_entiredata.csv')
data.drop(columns=['Unnamed: 0','subscriptiontype','onboard_flag','daysafterpur'],inplace=True)
data.info()


# In[101]:


from sklearn.compose import make_column_transformer, make_column_selector
ct1=make_column_transformer(
    (OneHotEncoder(sparse=False),make_column_selector(dtype_include=object)),
(StandardScaler(),make_column_selector(dtype_exclude=object)))


# In[102]:


df1=data.drop(columns=['conversion'])


# In[103]:


ct1.fit(df1)


# In[104]:


X_train1,X_test1=train_test_split(data,test_size=0.2,shuffle=True)


# In[105]:


X_train1=X_train1[X_train1['conversion']==0]

X_train1.drop(columns='conversion',inplace=True)
y_test=X_test1['conversion']
X_test1.drop(columns='conversion',inplace=True)


# In[106]:


X_train1,X_test1=ct1.transform(X_train1),ct1.transform(X_test1)


# In[107]:


import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dropout
from tensorflow.keras import Sequential
from keras import regularizers


# In[108]:


X_train1.shape


# In[115]:


input_dim=X_train1.shape[1]
autoencoder1=Sequential([
    
       
#     Dense(135,activation='tanh',activity_regularizer=regularizers.l1(10e-8)),
    
    Dense(135,activation='tanh',activity_regularizer=tf.keras.regularizers.L2(10e-6)),
    
    Dense(90,activation='tanh'),#activity_regularizer=regularizers.l1(10e-8)),
     
    Dense(45,activation='tanh'),#,activity_regularizer=regularizers.l1(10e-8)),
    
    
    Dense(45,activation='relu'),
    
    Dense(90,activation='tanh'),
    Dense(135,activation='tanh'),
    Dense(input_dim,activation='sigmoid')     
     
])

autoencoder1.compile(loss='mse',optimizer=tf.keras.optimizers.Adam(0.0007), metrics=['accuracy'])


# In[116]:


early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", # watch the val loss metric
                                                  patience=8) 
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",  
                                                 factor=0.1,  
                                                 patience=6,
                                                 verbose=1,  
                                                 min_lr=1e-7)


# In[117]:


history=autoencoder1.fit(X_train1,X_train1, epochs=70, batch_size=32,
                         validation_data=(X_test1,X_test1),shuffle=True,verbose=1,
                         callbacks=[early_stopping,reduce_lr])


# In[118]:


# predictions = autoencoder1.predict(X_test1)
# mse = np.mean(np.power(X_test1 - predictions, 2), axis=1)

# error_df = pd.DataFrame({'reconstruction_error': mse,
#                         'true_class': y_test})

# error_df.describe()


# In[128]:


predictions = autoencoder1.predict(X_test1)
error = np.mean((X_test1 - predictions), axis=1)


# In[129]:


th1=np.mean(error)+np.std(error)
th2=np.mean(error)-np.std(error)


# In[131]:


y_pred = [1 if (e > th1 or e < th2) else 0 for e in error]


# In[132]:


print("Confusion Matrix: \n",confusion_matrix(y_pred,y_test))
print("F1 Score: \n",f1_score(y_pred,y_test))
print("Balanced Accuracy Score: \n",balanced_accuracy_score(y_pred,y_test))
print("All Scores: \n",precision_recall_fscore_support(y_pred,y_test))


# In[133]:


import matplotlib.pyplot as plt
pr=[]
rc=[]
f1=[]
ba1=[]
for i in np.arange(0,0.02,0.004):    
    y_pred=[1 if e > i else 0 for e in error]
    pr.append(precision_recall_fscore_support(y_test,y_pred)[0][1])
    rc.append(precision_recall_fscore_support(y_test,y_pred)[1][1])
    f1.append(precision_recall_fscore_support(y_test,y_pred)[2][1])
    ba1.append(balanced_accuracy_score(y_test,y_pred))
    
plt.plot(np.arange(0,0.02,0.004),pr)
plt.plot(np.arange(0,0.02,0.004),rc)
plt.plot(np.arange(0,0.02,0.004),f1)
plt.plot(np.arange(0,0.02,0.004),ba1)
plt.legend(['precision', 'recall'], loc='upper right');


# In[ ]:





# ## Auto Encoder - Short form - deployed in Tableau

# In[134]:


df=pd.read_csv('pd_data3.csv')
df.drop(columns='Unnamed: 0',inplace=True)
df.drop(columns=['onboard_flag'],inplace=True)
df.info()


# In[135]:


from sklearn.compose import make_column_transformer, make_column_selector
ct1=make_column_transformer(
    (OneHotEncoder(sparse=False),make_column_selector(dtype_include=object)),
(StandardScaler(),make_column_selector(dtype_exclude=object)))

df1=df.drop(columns=['conversion'])

ct1.fit(df1)

X_train1,X_test1=train_test_split(df,test_size=0.2,shuffle=True)

X_train1=X_train1[X_train1['conversion']==0]

X_train1.drop(columns='conversion',inplace=True)
y_test=X_test1['conversion']
X_test1.drop(columns='conversion',inplace=True)

X_train1,X_test1=ct1.transform(X_train1),ct1.transform(X_test1)

X_train1.shape


# In[136]:


input_dim=X_train1.shape[1]
autoencoder1=Sequential([
    
       
#     Dense(135,activation='tanh',activity_regularizer=regularizers.l1(10e-8)),
    
    Dense(135,activation='tanh',activity_regularizer=tf.keras.regularizers.L2(10e-6)),
    
    Dense(90,activation='tanh'),#activity_regularizer=regularizers.l1(10e-8)),
     
    Dense(45,activation='tanh'),#,activity_regularizer=regularizers.l1(10e-8)),
    
    
    Dense(45,activation='relu'),
    
    Dense(90,activation='tanh'),
    Dense(135,activation='tanh'),
    Dense(input_dim,activation='sigmoid')     
     
])

autoencoder1.compile(loss='mse',optimizer=tf.keras.optimizers.Adam(0.0007), metrics=['accuracy'])


# In[ ]:


early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", # watch the val loss metric
                                                  patience=8) 
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",  
                                                 factor=0.1,  
                                                 patience=6,
                                                 verbose=1,  
                                                 min_lr=1e-7)


# In[137]:


history=autoencoder1.fit(X_train1,X_train1, epochs=70, batch_size=32,
                         validation_data=(X_test1,X_test1),shuffle=True,verbose=1,
                         callbacks=[early_stopping,reduce_lr])


# In[142]:


predictions = autoencoder1.predict(X_test1)
error = np.mean((X_test1 - predictions), axis=1)

th1=np.mean(error)+np.std(error)*2
th2=np.mean(error)-np.std(error)*2

y_pred = [1 if (e > th1 or e < th2) else 0 for e in error]

print("Confusion Matrix: \n",confusion_matrix(y_pred,y_test))
print("F1 Score: \n",f1_score(y_pred,y_test))
print("Balanced Accuracy Score: \n",balanced_accuracy_score(y_pred,y_test))
print("All Scores: \n",precision_recall_fscore_support(y_pred,y_test))


# In[139]:


def predict_student_autoencoder(a1,a2,a3):
    import pandas as pd
    import numpy as np
#     import tensorflow as tf
    # import pickle5
    
#     da1=pickle.dumps(twr,protocol=5)
    
#     tw=pickle.loads(da1)
    
    
    
    x=pd.DataFrame(np.array([[a1,a2,a3]]),columns=['student_country','totaldays_engaged','tendaysafter'])
    X_test=ct1.transform(x)    
     
#     y_pred1=tf.math.round(twr().predict(tf.convert_to_tensor(X_test,dtype=tf.float64))[0])

#     y_pred1=tf.math.round(autoencoder1.predict(tf.convert_to_tensor(X_test,dtype=tf.float64))[0])



    predictions = autoencoder1.predict(X_test1)
    error = np.mean((X_test1 - predictions), axis=1)

    th1=np.mean(error)+np.std(error)
    th2=np.mean(error)-np.std(error)

    y_pred = [1 if (e > th1 or e < th2) else 0 for e in error][0]


    
#     predictions = tf.convert_to_tensor(autoencoder1.predict(tf.convert_to_tensor(X_test,dtype=tf.float64),verbose=0)
#                                        ,dtype=tf.float64)    
#     mse = np.mean(np.power(X_test - predictions, 2), axis=1)
#     threshold=0.136
#     y_pred = [1.0 if mse > threshold else 0][0]
    
    print("y_pred is ", y_pred)
    
    return y_pred


# In[140]:


predict_student_autoencoder('IN',10,20)


# In[141]:


client.deploy('predict_student_autoencoder',
predict_student_autoencoder, 'keras Returns prediction of student conversion'
, override = True)


# In[ ]:





# In[ ]:





# In[ ]:





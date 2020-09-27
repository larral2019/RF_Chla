# -*- coding: utf-8 -*-
"""
Created on Tue May  5 02:21:14 2020

@author: LAR
"""
import pandas as pd
import numpy as np
import time
from datetime import datetime
from datetime import date,timedelta
import pylab as plt
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import load_iris
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              AdaBoostClassifier)
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from IPython.display import Image
from sklearn import tree
import pydotplus
import os
import collections
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import  classification_report
from sklearn.metrics import precision_score #精度
from sklearn.metrics import recall_score,f1_score #召回率,f1分数
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import cohen_kappa_score
from random import choice
#------------------------read file
print(__doc__)
chara=pd.read_csv('C:/Users/LAR/RFcls_chla_site_time.csv',sep=',', encoding='gbk') #,dtype='float64'
chara.set_index(["site_event"], inplace=True)





#------------------------RF

def RF_class_fit(X,y,textsize,randomstate):
    '''
    Parameters
    ----------
    X : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.

    Returns
    -------
    cfm : TYPE
        DESCRIPTION.
    importance : TYPE
        DESCRIPTION.
    score_train : TYPE
        DESCRIPTION.
    score_test : TYPE
        DESCRIPTION.
    labels_recall_score : TYPE
        DESCRIPTION.
    precision_score : TYPE
        DESCRIPTION.
    f1 : TYPE
        DESCRIPTION.

    '''
    print(__doc__)
    
    
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    X = (X - mean) / std
    
    
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=textsize,random_state= randomstate)
    
    #------------------------Parameters fitting
    #GridSearch
    param_test = {'n_estimators': list(np.linspace(400,800,4,dtype='int')),
        'max_depth': list(np.linspace(2,7,4,dtype='int'))}
    print('-1111111111111111111111111111111111111')
    '''
    n_estimators = [100, 300, 500, 800, 1200]
    max_depth = [5, 8, 15, 25, 30]
    min_samples_split = [2, 5, 10, 15, 100]
    min_samples_leaf = [1, 2, 5, 10]
    param_grid = dict(n_estimators=n_estimators, max_depth=max_depth,
                  min_samples_split=min_samples_split,
                  min_samples_leaf=min_samples_leaf)
    '''
    
    rf_clf=RandomForestClassifier( criterion='gini',bootstrap=True,oob_score=True,\
                                  random_state=randomstate ) #,class_weight='balanced',\  random_state=1,min_samples_split=5,min_samples_split=3 ,max_features=None,class_weight='balanced'
    print('-22222222222222')
    gridsearch= GridSearchCV(rf_clf,param_test,scoring='accuracy')
    print('-33333333333333')
    #train
    gridsearch.fit(X_train, y_train)
    print('-444444444444')
    #test
    
    best_score=gridsearch.best_score_
    best_para=gridsearch.best_params_
    

    #best parameters
    maxdepth=best_para['max_depth']
    nestimators=best_para['n_estimators']
    
    
    #test score using best parameter by train datasets
    rf_clf_best_fit=RandomForestClassifier( n_estimators = nestimators, max_depth=maxdepth, criterion='gini',bootstrap=True,oob_score=True,\
                                  random_state=randomstate,max_features=None)  #,min_samples_split=3  ,class_weight='balanced'
    print('-55555555555555')
    rf_clf_best_fit.fit(X_train, y_train)
    y_pred = rf_clf_best_fit.predict(X_test)
    
    
    #Evaluation test data
    cfm = confusion_matrix(y_test, y_pred)
    score_train = rf_clf_best_fit.score(X_train,y_train)
    score_test = rf_clf_best_fit.score(X_test,y_test)
    #recall_score
    labels_recall_score = recall_score(y_test,y_pred,average=None)
    #precision
    precision_scores=precision_score(y_test,y_pred,average=None)
    #kappa
    kappa_value = cohen_kappa_score(y_test,y_pred)
     #f1 score
    f1=f1_score(y_test,y_pred,average='macro')
    #importance of variables
    importance=rf_clf_best_fit.feature_importances_
    
     
    
    
    return cfm,importance,score_train,score_test , labels_recall_score,precision_scores,f1,best_para,best_score,kappa_value,textsize,randomstate


def obtain_para_score_surface( nestimators,maxdepth,X,y):

    #best parameters 
   
    para1=np.arange(1, nestimators*5, 1,dtype=int)
    para2=np.arange(1, maxdepth*5, 1,dtype=int)
    
    
    #train dataset and test dataset
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.4,random_state= 1)
    #para1=range(1,best_para['n_estimators']*2,1)
    #para2=range(1,best_para['max_depth']*5,1)
    
    #测试
    score_test_list=[]
    para1_list=[]
    para2_list=[]
    for par1 in para1:
        for par2 in para2:
            clf=RandomForestClassifier(n_estimators=par1,max_depth=par2,criterion='gini',\
                      bootstrap=True,oob_score=True,max_features=None,\
                                   random_state=1,class_weight='balanced') #,class_weight='balanced',min_samples_split=3
            clf.fit(X_train,y_train)
            y_test = clf.predict(X_test)
            score_test_list.append(clf.score(X_test,y_test))
            para1_list.append(par1)
            para2_list.append(par2)
    

    return para1_list,para2_list,score_test_list
    





#define the threshold of algal bloom 
chla_threshold=10   #ug/L

#test_size and random state defination
textchoice=[0.3,0.4,0.5,0.6]
random_state_ensemble=[1,3,5,7,10]


#main
chla_pattern=[]
for i in range(0,len(chara['chla'])):
    if chara['chla'].iloc[i]>chla_threshold:
        chla_pattern.append(1)
    else:
        chla_pattern.append(0)

chara_dele_chla=chara.drop('chla',axis=1)
chara_dele_chla=chara_dele_chla.drop('Cl',axis=1)

chara_dele_chla=chara.drop('chla',axis=1)
chara_dele_chla=chara_dele_chla.drop('Cl',axis=1)

#feature and classification
X_chla_site=np.array(chara_dele_chla.iloc[0:])
y_chla_site=np.array(chla_pattern)






#iteration for above setting
ensenble_result=[]
important_ensemble=[]
cfm_ensenmble=[]
for randomstate in random_state_ensemble:
    for i in range(0,1):
        for textsize in textchoice:
        
            #textsize=0.3   # choice(textchoice)
            #randomstate=1 #choice(random_state_ensemble)
            cfm_chla_site,importance_chla_site,score_train_chla_site,score_test_chla_site,labels_recall_score_chla_site, precision_scores_chla_site, f1_chla_site, best_para_chla_site, best_score_chla_site,kappa_value_chla_site,textsize_site,randomstate_site \
            =RF_class_fit(X_chla_site,y_chla_site,textsize,randomstate)
            temp_a=score_train_chla_site,score_test_chla_site, f1_chla_site,  best_score_chla_site,kappa_value_chla_site,textsize_site,randomstate_site
            ensenble_result.append(temp_a)
            temp_b=cfm_chla_site,labels_recall_score_chla_site, precision_scores_chla_site,best_para_chla_site
            cfm_ensenmble.append(temp_b)
            important_ensemble.append(importance_chla_site)
    
#output of model result
ensenble_result_df=pd.DataFrame(ensenble_result)
ensenble_result_df.to_csv('ensenble_result_df.csv', mode='w', header=True)

#output of important variables
important_ensemble_df=pd.DataFrame(important_ensemble)
important_ensemble_df.to_csv('important_ensemble_df.csv', mode='w', header=True)

#output of confuse matrix
cfm_ensenmble_df=pd.DataFrame(cfm_ensenmble)
cfm_ensenmble_df.to_csv('cfm_ensenmble_df.csv', mode='w', header=True)



#%%
from typing import TextIO
import pandas as pd 
import numpy as np 
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import PowerTransformer, OneHotEncoder
from sklearn.impute._iterative import IterativeImputer #for numeric values
from sklearn.impute import SimpleImputer #for categorical
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, roc_auc_score, plot_roc_curve, plot_confusion_matrix, confusion_matrix
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt 
import seaborn as sns
import lightgbm as lgb 
from category_encoders.target_encoder import TargetEncoder
import utils
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 1000)

#%%
data = pd.read_csv('./data/Leads.csv')
label = 'Converted'

import re
data = data.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]', '', x))

data.head(5)

#%%
#a bit of EDA 
#selecting types of column to process them accordingly 
cat_columns = list(data.select_dtypes(include='object').columns)
num_columns = list(set(data.columns) - set(cat_columns)) 
num_columns.remove(label) #our label should not get in here

#examine nulls
print(data.shape, data.isnull().sum())

# %%
#separate data
y = data[label]
X = data.drop([label], axis = 1)

#%%
#apply cleaning and encoding in EACH phase of the CV process 
splitter = StratifiedKFold()

f1_scores = [] #store results
auc_scores = []

for train_index, test_index in splitter.split(X = X, y = y):
    
    estim = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05)

    X_train_k, X_test_k = X.iloc[train_index], X.iloc[test_index]
    y_train_k, y_test_k = y.iloc[train_index], y.iloc[test_index]

    #creation of the pipeline each time
    X_train_k, X_test_k = utils.transform_data_oh(X_train = X_train_k, 
                                X_test = X_test_k, 
                                cat_columns = cat_columns, 
                                num_columns = num_columns 
                                )
    
    

    estim.fit(X = X_train_k, y = y_train_k)
    #get results
    preds = estim.predict(X=X_test_k)
    f1 = f1_score(y_true=y_test_k,y_pred=preds)
    auc = roc_auc_score(y_true=y_test_k,y_score=preds)
    
    f1_scores.append(f1)
    auc_scores.append(auc)

print('F1 average: ' + str(round(np.average(f1_scores), 3)))
print('AUC average: ' + str(round(np.average(auc_scores), 3)))

#%%
lgb.plot_importance(estim, max_num_features=20, figsize=(10,10), importance_type='gain')

# %%
import shap 
shap.initjs() 

# %%
#extract from the last iteration of K folds, just for demo purposes

explainer = shap.TreeExplainer(estim) 
shap_values = explainer.shap_values(X=X_train_k, y = y_train_k)

#%%
#explanation for cases that actually have been converted into sales
shap.summary_plot(shap_values=shap_values[1], features = X_train_k, feature_names = X_train_k.columns, 
                        plot_type = 'dot')

#%%
#explanation for cases that weren't converted
shap.summary_plot(shap_values=shap_values[0], features = X_train_k, feature_names = X_train_k.columns, 
                        plot_type = 'dot')






























# %%
#TODO inspect Top 5 features (Tags, Lead Profile, time spent on website, last notable activity, asymmetrique activity score)

display_df = X.iloc[train_index]

shap.dependence_plot(
                        ind = "Tags", shap_values = shap_values[1], 
                        features = X_train_k, display_features = X.iloc[train_index] #original dataset for original values
                        )

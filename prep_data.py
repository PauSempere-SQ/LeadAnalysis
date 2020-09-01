#%%
import pandas as pd 
import numpy as np 
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import PowerTransformer
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

#%%
data = pd.read_csv('./data/Leads_old.csv').drop(['Lead Number', 'Prospect ID'], axis = 1) #remove the IDs

data = data.drop(['Tags'], axis = 1)
data.to_csv('./data/Leads.csv', index = False)


# %%

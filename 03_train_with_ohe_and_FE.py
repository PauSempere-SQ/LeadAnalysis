#%%
import pandas as pd 
import numpy as np 
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import PowerTransformer, OneHotEncoder, RobustScaler, StandardScaler
from sklearn.impute._iterative import IterativeImputer #for numeric values
from sklearn.impute import SimpleImputer #for categorical
import sklearn.feature_selection as feature_selection
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, roc_auc_score, plot_roc_curve, plot_confusion_matrix, confusion_matrix
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt 
import seaborn as sns
import lightgbm as lgb 
import time
from category_encoders.target_encoder import TargetEncoder
import utils
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 1000)

#%%
data = pd.read_csv('./data/Leads.csv')
label = 'Converted'

data.head(5)

#%%
#selecting types of column to process them accordingly 
cat_columns = list(data.select_dtypes(include='object').columns)
num_columns = list(set(data.columns) - set(cat_columns)) 
num_columns.remove(label) #our label should not get in here

# %%
#split data and label
y = data[label]
X = data.drop([label], axis = 1)

#%%
#apply cleaning and encoding in EACH phase of the CV process 
splitter = StratifiedKFold()

#store results
f1_scores = [] 
auc_scores = []

s = time.clock()

#create our own stratified CV loop
for train_index, test_index in splitter.split(X = X, y = y):
    
    #create estimator
    estim = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05)

    #separate into train and test for the Knth fold in the split
    X_train_k, X_test_k = X.iloc[train_index], X.iloc[test_index]
    y_train_k, y_test_k = y.iloc[train_index], y.iloc[test_index]

    #reset names in each step of the loop
    cat_columns = list(X_train_k.select_dtypes(include='object').columns)
    num_columns = list(set(X_train_k.columns) - set(cat_columns)) 

    #generate features before normalizing and encoding
    X_train_k, X_test_k = utils.create_feat_engineering(
                                        X_train=X_train_k, X_test=X_test_k, 
                                        cat_columns = cat_columns, num_columns=num_columns
                                        )

    #print('shape after FE: ' + str(X_train_k.shape))

    #refresh numeric and categorical columns 
    cat_columns = list(X_train_k.select_dtypes(include='object').columns)
    num_columns = list(set(X_train_k.columns) - set(cat_columns)) 

    #print('cat columns after FE: ' + str(cat_columns))

    #clean, transform and encode data in each loop
    X_train_k, X_test_k = utils.transform_data_oh(X_train = X_train_k, 
                                X_test = X_test_k, 
                                cat_columns = cat_columns, 
                                num_columns = num_columns 
                                )

    print('data filled and encoded. Shape: ' + str(X_train_k.shape))
    
    X_train_k, X_test_k = utils.apply_RFE(      
                                            X_train = X_train_k, 
                                            y_train=y_train_k, 
                                            X_test=X_test_k, 
                                            verbose=False
                                        )

    #fit with the reduced data
    estim.fit(X = X_train_k, y = y_train_k)

    #get results
    preds = estim.predict(X=X_test_k)
    f1 = f1_score(y_true=y_test_k,y_pred=preds)
    auc = roc_auc_score(y_true=y_test_k,y_score=preds)
    
    f1_scores.append(f1)
    auc_scores.append(auc)

print('F1 average: ' + str(round(np.average(f1_scores), 3)))
print('AUC average: ' + str(round(np.average(auc_scores), 3)))

e = time.clock() 
print('Used ' + str(e-s))

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

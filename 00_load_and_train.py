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
data = pd.read_csv('./data/Leads.csv')
label = 'Converted'

data.head(5)

#%%
#a bit of EDA 
#selecting types of column to process them accordingly 
cat_columns = list(data.select_dtypes(include='object').columns)
num_columns = list(set(data.columns) - set(cat_columns)) 
num_columns.remove(label) #our label should not get in here

#examine nulls
print(data.shape, data.isnull().sum())

#%%
#for numeric cols, plot bloxplots hued by the target
for c in num_columns:
    plt.figure(figsize=(8,8)) #create new figure
    ax = sns.boxplot(x=label, y=c,  data=data) #plot histogram

#%%
#just our label
plt.figure(figsize=(8, 8))
ax = sns.countplot(x=label, data=data)

#%%
for c in cat_columns:
    plt.figure(figsize=(8, 8))
    ax = sns.countplot(x=c, hue=label, data=data)

# %%
#separate data
y = data[label]
X = data.drop([label], axis = 1)

#%%
#random split version
#we can do this because our dataset has no time-related features
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
num_imputer = IterativeImputer() 
cat_imputer = SimpleImputer(strategy='most_frequent') #use the mode to fill up NAs

#encode and transform
t_encoder = TargetEncoder()
num_transformer = PowerTransformer()

numeric_df = X_train.filter(num_columns)
cat_df = X_train.filter(cat_columns)

#numeric part
numeric_df = pd.DataFrame(num_imputer.fit_transform(X=numeric_df), columns=num_columns)
numeric_df = pd.DataFrame(num_transformer.fit_transform(X=numeric_df), columns=num_columns)

#cat part 
cat_df = pd.DataFrame(cat_imputer.fit_transform(X=cat_df), columns=cat_columns)
cat_df = pd.DataFrame(t_encoder.fit_transform(X=cat_df, y=y_train), columns=cat_columns)

X_train = pd.concat([cat_df, numeric_df], ignore_index=True, axis = 1)
X_train.columns = cat_columns + num_columns

#apply to test, only transforming
numeric_df = X_test.filter(num_columns)
cat_df = X_test.filter(cat_columns)

#numeric part
numeric_df = pd.DataFrame(num_imputer.transform(X=numeric_df), columns=num_columns)
numeric_df = pd.DataFrame(num_transformer.transform(X=numeric_df), columns=num_columns)

#cat part 
cat_df = pd.DataFrame(cat_imputer.transform(X=cat_df), columns=cat_columns)
cat_df = pd.DataFrame(t_encoder.transform(X=cat_df), columns=cat_columns) #without Y

X_test = pd.concat([cat_df, numeric_df], ignore_index=True, axis = 1)
X_test.columns = cat_columns + num_columns

X_test.head() 

#%%
model = lgb.LGBMClassifier()
model.fit(X = X_train, y = y_train)

#%%
plot_roc_curve(estimator=model,X=X_test, y=y_test)

preds = model.predict(X = X_test)

print(confusion_matrix(y_true=y_test,y_pred=preds))

print('\n F1 Score :' + str(f1_score(y_true=y_test,y_pred=preds)))

#%%
#apply cleaning and encoding in EACH phase of the CV process 
splitter = StratifiedKFold()

f1_scores = [] #store results
auc_scores = []

for train_index, test_index in splitter.split(X = X, y = y):
    
    estim = lgb.LGBMClassifier()

    X_train_k, X_test_k = X.iloc[train_index], X.iloc[test_index]
    y_train_k, y_test_k = y.iloc[train_index], y.iloc[test_index]

    #creation of the pipeline each time
    X_train_k, X_test_k = utils.transform_data(X_train = X_train_k, 
                                y_train = y_train_k, 
                                X_test = X_test_k, 
                                y_test = y_test_k, 
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
lgb.plot_importance(estim, figsize=(10,10), importance_type='gain')

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

# %%

# %%

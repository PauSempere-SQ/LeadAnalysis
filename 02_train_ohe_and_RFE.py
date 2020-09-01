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
from category_encoders.target_encoder import TargetEncoder
import utils
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 1000)

#%%
data = pd.read_csv('./data/Leads.csv')
label = 'Converted'

data.head()

#%%
#selecting types of column to process them accordingly 
cat_columns = list(data.select_dtypes(include='object').columns)
num_columns = list(set(data.columns) - set(cat_columns)) 
num_columns.remove(label) #our label should not get in here

#examine nulls
#print(data.shape, '\n', data.isnull().sum())

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

#create our own stratified CV loop
for train_index, test_index in splitter.split(X = X, y = y):
    
    #create estimator
    estim = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05)

    #separate into train and test for the Knth fold in the split
    X_train_k, X_test_k = X.iloc[train_index], X.iloc[test_index]
    y_train_k, y_test_k = y.iloc[train_index], y.iloc[test_index]

    #clean, transform and encode data in each loop
    X_train_k, X_test_k = utils.transform_data_oh(X_train = X_train_k, 
                                X_test = X_test_k, 
                                cat_columns = cat_columns, 
                                num_columns = num_columns 
                                )

    print('data filled and encoded. Shape: ' + str(X_train_k.shape))
    
    #reduce our dataset, improves stability, speed and interpretability
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

#%%
lgb.plot_importance(estim, max_num_features=20, figsize=(10,10), importance_type='gain')

# %%
import shap 
shap.initjs() 

# %%
#extract from the last iteration of K folds, just for demo purposes

explainer = shap.TreeExplainer(estim) 
shap_values = explainer.shap_values(X = X_train_k, y = y_train_k)


#%%
#explanation for cases that actually have been converted into sales
shap.summary_plot(
                    shap_values=shap_values[1], features = X_train_k
                    , feature_names = X_train_k.columns
                    , plot_type = 'dot'
                    )

#%%
#explanation for cases that weren't converted
shap.summary_plot(shap_values=shap_values[0], features = X_train_k, feature_names = X_train_k.columns, 
                        plot_type = 'dot')





























#%%
#inspect specific cases (right and wrong classified) with decision plots
#create predictions to check which ones are right and which ones are wrong
preds = estim.predict(X_test_k)

confusion_matrix(y_test_k,preds)

#%%
#check other cases 
from sklearn.metrics import precision_score
cm = confusion_matrix(y_test_k, preds)

TN = cm[0][0]
FP = cm[0][1]
FN = cm[1][0]
TP = cm[1][1]

print('precision: ' + str(round(precision_score(y_test_k, preds), 3)))

print('negative predictive value : ' + str(round(TN/(TN+FN), 3)))

#%%
##################################################
##                 DECISION PLOTS               ##
##################################################
right_indices = np.where(preds == y_test_k)[0]
right_cases = X_test_k.iloc[right_indices].head(10)

right_cases_display = X_test_k.loc[right_indices].head(10)

wrong_indices = np.where(preds != y_test_k)[0]
wrong_cases = X_test_k.iloc[wrong_indices].head(10)

# %%
#correct predictions for the converted cases
shap_values = explainer.shap_values(right_cases)[1] 
shap_inter_values = explainer.shap_interaction_values(right_cases)
if isinstance(shap_inter_values, list):
    shap_inter_values = shap_inter_values[1]

# %%
#importance of each feature value (transformed) in the first well classified case
shap.decision_plot(explainer.expected_value, shap_values[0], right_cases, link='logit')

#%%
#we can even inspect the first order interaction values from the tree structures
shap.decision_plot(explainer.expected_value, shap_inter_values, right_cases, link='logit')


# %%
#incorrect predictions for the converted cases
shap_values = explainer.shap_values(wrong_cases)[1] 
shap_inter_values = explainer.shap_interaction_values(wrong_cases)
if isinstance(shap_inter_values, list):
    shap_inter_values = shap_inter_values[1]

# %%
#importance of each feature value (transformed) in the first wrong classified case
shap.decision_plot(explainer.expected_value, shap_values[0], wrong_cases, link='logit')

# %%

































#%%
#threshold optimization (simple)
thresholds = np.arange(0.01, 1.0, 0.01)
scores = dict()

#get scores rather than labels
preds_proba = pd.DataFrame(estim.predict_proba(X = X_test_k), columns = ['proba_0', 'proba_1'])
preds_proba = preds_proba['proba_1'] #only proba_1 cases, probability to get the lead done

#%%
#get adjusted labels
def get_labels(y_scores, t):
    return[1 if y >= t else 0 for y in y_scores]
    
for t in thresholds:
    predicted_labels = get_labels(preds_proba, t)
    s = f1_score(y_test_k, predicted_labels) 
    scores[t] = s 

max_threshold = round(max(scores, key=scores.get), 3)
print('best threshold is: ' + str(max_threshold))


# %%
import seaborn as sns

thresholds_plot = list(scores.keys()) 
scores_plot = list(scores.values())

#plot our gain chart
sns.set(style="whitegrid")
fig = plt.gcf()
fig.set_size_inches(10, 10)
sns_plot = sns.lineplot(x=thresholds_plot, y=scores_plot)
sns_plot.text(max_threshold, max(scores_plot), "MAX PERFORMANCE", weight="bold", color = "red")
    

# %%

#%%
import pandas as pd 
import numpy as np 
import re
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import PowerTransformer, OneHotEncoder, RobustScaler
from sklearn.impute._iterative import IterativeImputer #for numeric values
from sklearn.impute import SimpleImputer #for categorical
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, plot_roc_curve, plot_confusion_matrix, confusion_matrix
from sklearn.compose import ColumnTransformer
import sklearn.feature_selection as feature_selection
import matplotlib.pyplot as plt 
import seaborn as sns
import lightgbm as lgb 
from category_encoders.target_encoder import TargetEncoder

def transform_data(X_train, y_train, X_test, y_test, cat_columns, num_columns):
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
    cat_df = pd.DataFrame(t_encoder.transform(X=cat_df), columns=cat_columns)

    X_test = pd.concat([cat_df, numeric_df], ignore_index=True, axis = 1)
    X_test.columns = cat_columns + num_columns

    #return train and test df
    return X_train, X_test

def clean_num_data(X):
    X.fillna(X.mean(), inplace=True)
    X.interpolate(method = 'linear', axis = 0, downcast='infer', inplace=True)
    X.replace([np.inf, -np.inf], 0, inplace=True)

    _float_cols = X.select_dtypes(include=np.float).columns 
    _int_cols = X.select_dtypes(include=np.int).columns

    for n in _float_cols:
        X[n] = pd.to_numeric(round(X[n], 3), downcast='float')
        X[n] = pd.to_numeric(round(X[n], 3), downcast='float')

    for n in _int_cols:
        X[n] = pd.to_numeric(X[n], downcast='integer')
        X[n] = pd.to_numeric(X[n], downcast='integer')

    return X

def transform_data_oh(X_train, X_test, cat_columns, num_columns):
    #imputers
    #num_imputer = IterativeImputer(verbose=2,n_nearest_features=10) #save time and memory to impute
    #num_imputer = SimpleImputer(strategy="mean")
    cat_imputer = SimpleImputer(strategy='most_frequent') #use the mode to fill up NAs

    #encode and transform
    num_transformer = RobustScaler()
    cat_encoder = OneHotEncoder(handle_unknown='ignore', dtype=np.int, sparse=False)
    

    numeric_df = X_train.filter(num_columns)
    cat_df = X_train.filter(cat_columns)

    #numeric part
    numeric_df = clean_num_data(numeric_df)
    #numeric_df = pd.DataFrame(num_imputer.fit_transform(X=numeric_df), columns=num_columns)
    numeric_df = pd.DataFrame(num_transformer.fit_transform(X=numeric_df), columns=num_columns)

    #cat part 
    cat_df = pd.DataFrame(cat_imputer.fit_transform(X=cat_df), columns=cat_columns)
    cat_df = pd.DataFrame(cat_encoder.fit_transform(X=cat_df))

    cat_df.columns = cat_encoder.get_feature_names(input_features=cat_columns)

    X_train = pd.concat([cat_df, numeric_df], ignore_index=True, axis = 1)
    X_train.columns = list(cat_df.columns) + num_columns

    #TEST
    numeric_df_test = X_test.filter(num_columns)
    cat_df_test = X_test.filter(cat_columns)

    numeric_df_test = clean_num_data(numeric_df_test)
    #numeric_df_test = pd.DataFrame(num_imputer.transform(X=numeric_df_test), columns=num_columns)
    numeric_df_test = pd.DataFrame(num_transformer.transform(X=numeric_df_test), columns=num_columns)

    #cat part 
    cat_df_test = pd.DataFrame(cat_imputer.transform(X=cat_df_test), columns=cat_columns)
    cat_df_test = pd.DataFrame(cat_encoder.transform(X=cat_df_test))

    cat_df_test.columns = cat_encoder.get_feature_names(input_features=cat_columns)

    X_test = pd.concat([cat_df_test, numeric_df_test], ignore_index=True, axis = 1)
    X_test.columns = list(cat_df_test.columns) + num_columns

    X_train = X_train.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]', '', x))
    X_test = X_test.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]', '', x))

    #return train and test df
    return X_train, X_test

def create_feat_engineering(X_train, X_test, cat_columns, num_columns):
    feats = []
    added_cols = []

    #use our numerical columns to find out our categorical columns
    for cat in cat_columns: 
        #create the grouping object for each categorical feature we've got
        # FROM TRAINING SET
        group_by_feat = X_train.groupby(by=cat)
        for num_feat in num_columns:
            #go over our numeric features to generate synthetic features
            col_names = [cat, 'mean_' + num_feat + '_by_' + cat, 'std_' + num_feat + '_by_' + cat, 
                                'max_' + num_feat + '_by_' + cat, 'min_' + num_feat + '_by_' + cat]

            #create mean, std, max and min 
            df_grouped = group_by_feat[num_feat].agg(['mean', 'std', 'max', 'min']).reset_index()
            #add feature names
            df_grouped.columns = col_names

            added_cols.append(col_names)

            #for train data
            X_train = pd.merge(left = X_train, right=df_grouped, how = 'left', on=cat, 
                                    suffixes = ('', '_feat'))

            #number of sigmas the amount is deviated from the mean
            X_train[num_feat + '_sigmas_on_' + cat] = round((X_train[num_feat] - X_train['mean_' + num_feat + '_by_' + cat]) / X_train['std_' + num_feat + '_by_' + cat], 2)
            X_train[num_feat + '_amplitude_on_' + cat] = round(X_train['max_' + num_feat + '_by_' + cat] - X_train['min_' + num_feat + '_by_' + cat], 2)
            
            #print('shape of test: ' + str(X_train.shape))

            #join data from train set into test set to avoid data leakage
            X_test = pd.merge(left = X_test, right=df_grouped, how = 'left', on=cat, 
                                     suffixes = ('', '_feat'))
                
            # #number of sigmas the amount is deviated from the mean
            X_test[num_feat + '_sigmas_on_' + cat] = round(abs((X_test[num_feat] - X_test['mean_' + num_feat + '_by_' + cat])) / X_test['std_' + num_feat + '_by_' + cat], 2)
            X_test[num_feat + '_amplitude_on_' + cat] = round(X_test['max_' + num_feat + '_by_' + cat] - X_test['min_' + num_feat + '_by_' + cat], 2)
                
            #print('shape of test: ' + str(X_test.shape) + '\n')

    #optimize data types before leaving 
    _float_cols = X_train.select_dtypes(include=np.float).columns 
    _int_cols = X_train.select_dtypes(include=np.int).columns

    for n in _float_cols:
        X_train[n] = pd.to_numeric(X_train[n], downcast='float')
        X_test[n] = pd.to_numeric(X_test[n], downcast='float')

    for n in _int_cols:
        X_train[n] = pd.to_numeric(X_train[n], downcast='integer')
        X_test[n] = pd.to_numeric(X_test[n], downcast='integer')

    return X_train, X_test

def apply_RFE(X_train, y_train, X_test, verbose):
    #reduce feature amount to find simpler and more robust models. 
    #improves stability and interpretation
    rfe_estimator = lgb.LGBMClassifier(n_estimators=200)

    #try RFECV to try combinations of features
    #jump in 5 feature steps for timing reasons
    rfe = feature_selection.RFECV(estimator = rfe_estimator, step = 5, n_jobs = 3, 
                                    cv = 3, 
                                    scoring='f1', 
                                    verbose = 2)

    rfe.fit(X=X_train, y=y_train.values.ravel())

    if verbose == True:
        print("Optimal number of features : %d" % rfe.n_features_)

        # Plot number of features VS. cross-validation scores
        sns.set(style="whitegrid")

        plt.figure()
        plt.xlabel("Number of features selected")
        plt.ylabel("Cross validation score")
        plt.plot(range(1, len(rfe.grid_scores_) + 1), rfe.grid_scores_)
        plt.show()

    #select data from our original dataset
    X_train = pd.DataFrame(rfe.transform(X_train), columns = X_train.columns[rfe.support_])
    X_test = pd.DataFrame(rfe.transform(X_test), columns = X_test.columns[rfe.support_])

    return X_train, X_test
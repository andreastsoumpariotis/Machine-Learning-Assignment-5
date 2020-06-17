#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 13:21:57 2020

@author: andreastsoumpariotis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
import patsy
import itertools
from IPython.display import display, HTML
from sklearn import linear_model
from sklearn.model_selection import KFold
from ipywidgets import interact
import ipywidgets as widgets
import warnings
warnings.filterwarnings('ignore')
from ipywidgets import interact
import ipywidgets as widgets
# Deps for pca/pcr
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from scipy import stats

# Question 8

# part a

np.random.seed(1)
x = np.random.normal(0, 1, 100)
e = np.random.normal(0, 1, 100)

# part b

y = 2 + 1*x + 4*(x**2) + 2*(x**3) + e
sns.scatterplot(x = x, y = y)

# part c

def get_models(k, x, y):

    combos = itertools.combinations(list(x.columns), k)

    models = []
    for x_label in combos:

        x_smf = ' + '.join(x_label)
        f = '{} ~ {}'.format(y.columns[0], x_smf)

        model = smf.ols(formula=f, data = pd.concat([x, y], axis=1)).fit()

        models += [(f, model)]
    return models


def min_rss(statsmodels):
    """Return model with lowest Residual Sum of Squares (RSS)"""
    return sorted(statsmodels, key=lambda tup: tup[1].ssr)[0]


def max_adjr2(statsmodels):
    """Return model with max R-squared"""
    return sorted(statsmodels, reverse=True, key=lambda tup: tup[1].rsquared_adj)[0]

def min_bic(statsmodels):
    """Return model with min Bayes' Information Criteria"""
    return sorted(statsmodels, reverse=False, key=lambda tup: tup[1].bic)[0]

def min_aic(statsmodels):
    """Return model with min Akaike's Information Criteria"""
    return sorted(statsmodels, reverse=False, key=lambda tup: tup[1].aic)[0]

x = pd.DataFrame({'x': x,
                  'x2': x**2,
                  'x3': x**3,
                  'x4': x**4,
                  'x5': x**5,
                  'x6': x**6,
                  'x7': x**7,
                  'x8': x**8,
                  'x9': x**9,
                  'x10': x**10})
y = pd.DataFrame({'y': y})

# get all model results
model_subsets = []
for k in range(len(x.columns)):
    k=k+1
    subset = get_models(k, x, y)
    model_subsets += [subset]
    print(format(k))

# Get predictor count
k = np.arange(1, len(x.columns)+1)

# adjr2
# ------------------------------------------------
display(HTML('<h4>Adjusted R^2</h4>'))

# Adjusted R^2
AdjR2 = [max_adjr2(m)[1].rsquared_adj for m in model_subsets]

sns.lineplot(x=k, y = AdjR2)
plt.xlabel('k')
plt.ylabel('R^2')
plt.show()

# R^2 Coeffucients
R2_Coef = [(max_adjr2(m)[1].rsquared_adj, max_adjr2(m)[1].params) for m in model_subsets]
print(max(R2_Coef)[1])


# Bayes' Information Criteria (BIC)
# ------------------------------------------------
display(HTML('<h4>Bayes\' Information Criteria</h4>'))

# BIC
bic = [min_bic(m)[1].bic for m in model_subsets]

sns.lineplot(x=k, y=bic)
plt.xlabel('k')
plt.ylabel('bic')
plt.show()

# BIC Coeffucients
bic_coef = [(min_bic(m)[1].bic, min_bic(m)[1].params) for m in model_subsets]
print(min(bic_coef)[1])


# Akaike's Information Criteria (AIC/ C_p)
# ------------------------------------------------
display(HTML('<h4>Akaike\'s Information Criteria</h4>'))

# AIC
aic = [min_aic(m)[1].aic for m in model_subsets]

sns.lineplot(x=k, y=aic)
plt.xlabel('k')
plt.ylabel('aic')
plt.show()

# AIC Coeffucients
aic_coef = [(min_aic(m)[1].aic, min_aic(m)[1].params) for m in model_subsets]
print('Model selected: \n{}'.format(min(aic_coef)[1]))

# part d

def forward_stepwise(x, y, scorer='ssr', results=[(0, [])]):
    
    p_all    = list(x.columns)
    p_used   = results[-1][1]
    p_unused = [p for p in p_all if p not in p_used]
    
    if not p_unused:
        scores = [r[0] for r in results]
        preds  = [r[1] for r in results]
        return pd.DataFrame({scorer: scores, 'predictors': preds}).drop(0)
    
    r = []
    for p in p_unused:
        f     = '{} ~ {}'.format(y.columns[0], '+'.join([p]+p_used))

        model = smf.ols(formula=f, data=pd.concat([x, y], axis=1)).fit()
        r    += [(model, [p]+p_used)]
    
    if scorer == 'ssr':
        best_model = sorted(r, key=lambda tup: tup[0].ssr)[0]
        best_score = (best_model[0].ssr, best_model[1])
    elif scorer == 'rsquared_adj':
        best_model = sorted(r, key=lambda tup: tup[0].rsquared_adj)[-1]
        best_score = (best_model[0].rsquared_adj, best_model[1])        
    elif scorer == 'bic':
        best_model = sorted(r, key=lambda tup: tup[0].bic)[0]
        best_score = (best_model[0].bic, best_model[1]) 
    elif scorer == 'aic':
        best_model = sorted(r, key=lambda tup: tup[0].aic)[0]
        best_score = (best_model[0].aic, best_model[1]) 
                        
    new_results = results + [best_score]

    return forward_stepwise(x, y, scorer, new_results)

def backward_stepwise(X, y, scorer='ssr', results=[]):
    
    p_all = list(x.columns)

    if not results:

        f     = '{} ~ {}'.format(y.columns[0], '+'.join(p_all))
        model = smf.ols(formula=f, data=pd.concat([x, y], axis=1)).fit()

        if scorer == 'ssr':
            return backward_stepwise(x, y, scorer, [(model.ssr, p_all)])
        if scorer == 'rsquared_adj':
            return backward_stepwise(x, y, scorer, [(model.rsquared_adj, p_all)])
        if scorer == 'bic':
            return backward_stepwise(x, y, scorer, [(model.bic, p_all)])
        if scorer == 'aic':
            return backward_stepwise(x, y, scorer, [(model.aic, p_all)])
    else:
        p_used = results[-1][1]

    if len(p_used) == 1:
        scores = [r[0] for r in results]
        preds  = [r[1] for r in results]
        return pd.DataFrame({scorer: scores, 'predictors': preds})    

    r = []
    for p in p_used:
        p_test = [i for i in p_used if i != p]
        f     = '{} ~ {}'.format(y.columns[0], '+'.join(p_test))

        model = smf.ols(formula=f, data=pd.concat([x, y], axis=1)).fit()
        r     += [(model, p_test)]

    if scorer == 'ssr':
        best_model = sorted(r, key=lambda tup: tup[0].ssr)[0]
        best_score = (best_model[0].ssr, best_model[1])
    elif scorer == 'rsquared_adj':
        best_model = sorted(r, key=lambda tup: tup[0].rsquared_adj)[-1]
        best_score = (best_model[0].rsquared_adj, best_model[1])        
    elif scorer == 'bic':
        best_model = sorted(r, key=lambda tup: tup[0].bic)[0]
        best_score = (best_model[0].bic, best_model[1]) 
    elif scorer == 'aic':
        best_model = sorted(r, key=lambda tup: tup[0].aic)[0]
        best_score = (best_model[0].aic, best_model[1]) 

    new_results = results + [best_score]

    return backward_stepwise(x, y, scorer, new_results)


def subset_analysis(dataframe, scorer):
    dataframe['predictors_str'] = dataframe['predictors'].astype(str)
    
    ax = sns.lineplot(x='predictors_str', y=scorer, data=dataframe, sort=False)
    plt.xticks(rotation=90)
    plt.show();
    
    if scorer == 'rsquared_adj':
        display(dataframe[dataframe[scorer] ==  dataframe[scorer].max()].drop('predictors_str', axis=1))
    else:
        display(dataframe[dataframe[scorer] ==  dataframe[scorer].min()].drop('predictors_str', axis=1))

# Forward Stepwise Selection
        
# Adjusted R^2
display(HTML('<h4>Adjusted R^2</h4>'))
scorer = 'rsquared_adj'
subset_analysis(forward_stepwise(x, y, scorer = scorer), scorer)

# BIC
display(HTML('<h4>Bayes\' Information Criteria</h4>'))
scorer = 'bic'
subset_analysis(forward_stepwise(x, y, scorer=scorer), scorer)

# AIC
display(HTML('<h4>Akaike\'s Information Criteria</h4>'))
scorer = 'aic'
subset_analysis(forward_stepwise(x, y, scorer=scorer), scorer)


# Backward Stepwise Selection

# Adjusted R^2
display(HTML('<h4>Adjusted R^2</h4>'))
scorer = 'rsquared_adj'
subset_analysis(backward_stepwise(x, y, scorer=scorer), scorer)

# BIC
display(HTML('<h4>Bayes\' Information Criteria</h4>'))
scorer = 'bic'
subset_analysis(backward_stepwise(x, y, scorer=scorer), scorer)

# AIC
display(HTML('<h4>Akaike\'s Information Criteria</h4>'))
scorer = 'aic'
subset_analysis(backward_stepwise(x, y, scorer=scorer), scorer)

# part e

def mse(y_hat, y):
    """Calculate Mean Squared Error"""
    return np.sum(np.square(y_hat - y)) / y.size

def lasso_cv(X, y, λ, k):
    """Perform the lasso with 
    k-fold cross validation to return mean MSE scores for each fold"""
    # Split dataset into k-folds
    # Note: np.array_split doesn't raise excpetion is folds are unequal in size
    X_folds = np.array_split(X, k)
    y_folds = np.array_split(y, k)
    
    MSEs = []
    for f in np.arange(len(X_folds)):
        # Create training and test sets
        X_test  = X_folds[f]
        y_test  = y_folds[f]
        X_train = X.drop(X_folds[f].index)
        y_train = y.drop(y_folds[f].index)
        
        # Fit model
        model = linear_model.Lasso(alpha=λ, fit_intercept=True, normalize=False, max_iter=1000000).fit(X_train, y_train)

        # Measure MSE
        y_hat = model.predict(X_test)
        MSEs += [mse(y_hat, y_test['y'])]
    return MSEs

lambdas = np.arange(0.01, 1, 0.01)
MSEs = []
for l in lambdas:
    MSEs += [np.mean(lasso_cv(X, y, λ=l, k=10))]

sns.scatterplot(x='λ', y='MSE', data=pd.DataFrame({'λ': lambdas, 'MSE': MSEs}));

min(zip(MSEs, lambdas))

lamb = min(zip(MSEs, lambdas))[1]
model = linear_model.Lasso(alpha=lamb, fit_intercept=True, normalize=False, max_iter=1000000).fit(X, y)
df = pd.DataFrame({'feature': X.columns, 'coefficient': model.coef_})
# plot
ax = sns.barplot(x='coefficient', y='feature', data=df);
plt.xticks(rotation=90)
plt.show();
display(df)

# part f

np.random.seed(1)
x   = np.random.normal(0, 1, 100)
eps = np.random.normal(0 ,1 , 100)

y_np = 5 + 1.2*(x**7) + eps

sns.scatterplot(x=x, y=y_np);

X = pd.DataFrame({'X': x,
                  'X2': x**2,
                  'X3': x**3,
                  'X4': x**4,
                  'X5': x**5,
                  'X6': x**6,
                  'X7': x**7,
                  'X8': x**8,
                  'X9': x**9,
                  'X10': x**10})
y = pd.DataFrame({'y': y_np})

# Best Subset Selection

def min_rss(statsmodels):
    """Return model with lowest Residual Sum of Squares (RSS)"""
    return sorted(statsmodels, key=lambda tup: tup[1].ssr)[0]

def max_adjr2(statsmodels):
    """Return model with max R-squared"""
    return sorted(statsmodels, reverse=True, key=lambda tup: tup[1].rsquared_adj)[0]

def min_bic(statsmodels):
    """Return model with min Bayes' Information Criteria"""
    return sorted(statsmodels, reverse=False, key=lambda tup: tup[1].bic)[0]

def min_aic(statsmodels):
    """Return model with min Akaike's Information Criteria"""
    return sorted(statsmodels, reverse=False, key=lambda tup: tup[1].aic)[0]


def get_models(k, X, y):
    """
    Fit all possible models that contain exactly k predictors.
    """
    # List all available predictors
    X_combos = itertools.combinations(list(X.columns), k)
    
    # Fit all models
    models = []
    for X_label in X_combos:
        # Parse patsy formula
        X_smf = ' + '.join(X_label)
        f     = '{} ~ {}'.format(y.columns[0], X_smf)
        # Fit model
        model = smf.ols(formula=f, data=pd.concat([X, y], axis=1)).fit()
        # Return results
        models += [(f, model)]
    return models


# Intended API
# ----------------------------------------------

def best_subset(X, y, scorer='ssr'):
    """Perform best subset selection using Residual Sum of Squares to
    select best model in each subset.
    Notes: highly computationally expensive for large number of features in  X
    Maxes out my laptop for p > 14"""
    # get all model results
    model_subsets = []
    for k in range(len(X.columns)):
        k=k+1
        subset = get_models(k, X, y)
        model_subsets += [subset]
        print('Best subset selected: k = {}/{}, done'.format(k, len(X.columns)))

    # Select best in each subset using chosen scorer
    if scorer == 'ssr':
        # Get best rss score for each subset
        return [min_rss(m) for m in model_subsets]
    elif scorer == 'rsquared_adj':
        # Get best rss score for each subset
        return [max_adjr2(m) for m in model_subsets]       
    elif scorer == 'bic':
        # Get best rss score for each subset
        return [min_bic(m) for m in model_subsets]
    elif scorer == 'aic':
        # Get best rss score for each subset
        return [min_aic(m) for m in model_subsets]

    
def cross_val(formula, X, y, k):
    """Perform k-fold cross validation to return mean MSE score
    Expects formula as Patsy formula"""
    # Split dataset into k-folds
    # Note: np.array_split doesn't raise excpetion is folds are unequal in size
    X_folds = np.array_split(X, k)
    y_folds = np.array_split(y, k)
    
    MSEs = []
    for f in np.arange(len(X_folds)):
        # Create training and test sets
        X_test  = X_folds[f]
        y_test  = y_folds[f]
        X_train = X.drop(X_folds[f].index)
        y_train = y.drop(y_folds[f].index)
        
        # Fit model
        model = smf.ols(formula=formula, data=pd.concat([X_train, y_train], axis=1)).fit()
        
        # Measure MSE
        y_hat = model.predict(X_test)
        MSEs += [mse(y_hat, y_test['y'])]
    return (MSEs, formula)

# get all model results
best_subset_models = best_subset(X, y, scorer='bic')

# Parse results
best_subset_formula = [f[0] for f in best_subset_models]
best_subset_mses    = [np.mean(cross_val(f, X, y, 10)[0]) for f in best_subset_formula]
df = pd.DataFrame({'formula': best_subset_formula, 'mse': best_subset_mses})

# Show chosen model
display(df[df['mse'] == df['mse'].min()])
best_subset_params = best_subset_models[2][1].params
display(best_subset_params)

# Plot mse across subsets
ax = sns.lineplot(x='formula', y='mse', data=df, sort=False)
plt.xticks(rotation=90)
plt.show();

# Lasso
def lasso_cv(X, y, λ, k):
    """Perform the lasso with 
    k-fold cross validation to return mean MSE scores for each fold"""
    # Split dataset into k-folds
    # Note: np.array_split doesn't raise excpetion is folds are unequal in size
    X_folds = np.array_split(X, k)
    y_folds = np.array_split(y, k)
    
    MSEs = []
    for f in np.arange(len(X_folds)):
        # Create training and test sets
        X_test  = X_folds[f]
        y_test  = y_folds[f]
        X_train = X.drop(X_folds[f].index)
        y_train = y.drop(y_folds[f].index)
        
        # Fit model
        model = linear_model.Lasso(alpha=λ, fit_intercept=True, normalize=True, max_iter=1000000).fit(X_train, y_train)

        # Measure MSE
        y_hat = model.predict(X_test)
        MSEs += [mse(y_hat, y_test['y'])]
    return MSEs

lambdas = np.arange(0.01, 0.1, 0.001)
MSEs    = [] 
for l in lambdas:
    MSEs += [np.mean(lasso_cv(X, y, λ=l, k=10))]

sns.scatterplot(x='λ', y='MSE', data=pd.DataFrame({'λ': lambdas, 'MSE': MSEs}));

min(zip(MSEs, lambdas))

# What coefficients does the lasso choose for the optimal lambda value?
λ = min(zip(MSEs, lambdas))[1]
model = linear_model.Lasso(alpha=λ, fit_intercept=True, normalize=True, max_iter=1000000).fit(X, y)
intercept  = pd.DataFrame({'feature': 'X0', 'coefficient': model.intercept_})
lasso_a_df = intercept.append(pd.DataFrame({'feature': X.columns, 'coefficient': model.coef_}), ignore_index=True)
ax = sns.barplot(x='coefficient', y='feature', data=lasso_a_df);
plt.xticks(rotation=90)
plt.show();

display(lasso_a_df)

lambdas = np.arange(0.000001, 0.0005, 0.00001)
MSEs = [] 
for l in lambdas:
    MSEs += [np.mean(lasso_cv(X, y, λ=l, k=10))]

sns.scatterplot(x='λ', y='MSE', data=pd.DataFrame({'λ': lambdas, 'MSE': MSEs}))
plt.show();

# What coefficients does the lasso choose for the optimal lambda value?
λ = min(zip(MSEs, lambdas))[1]
model = linear_model.Lasso(alpha=λ, fit_intercept=True, normalize=True, max_iter=1000000).fit(X, y)
intercept  = pd.DataFrame({'feature': 'X0', 'coefficient': model.intercept_})
lasso_b_df = intercept.append(pd.DataFrame({'feature': X.columns, 'coefficient': model.coef_}), ignore_index=True)
ax = sns.barplot(x='coefficient', y='feature', data=lasso_b_df);
plt.xticks(rotation=90)
plt.show();

display(lasso_b_df)

min(zip(MSEs, lambdas))

x   = np.random.normal(0, 1, 100)
X = pd.DataFrame({'X': x,
                  'X2': x**2,
                  'X3': x**3,
                  'X4': x**4,
                  'X5': x**5,
                  'X6': x**6,
                  'X7': x**7,
                  'X8': x**8,
                  'X9': x**9,
                  'X10': x**10})
X_np = np.insert(np.array(X), 0, 1, axis=1)

y_actual = 5 + 1.2*(x**7)
bss_params = np.zeros(11)
bss_params[0] = best_subset_params[0]
bss_params[2] = best_subset_params[1]
bss_params[6] = best_subset_params[2]
bss_params[7] = best_subset_params[3]

lasso_a_params = np.array(lasso_a_df['coefficient'])
lasso_b_params = np.array(lasso_b_df['coefficient'])

y_bss     = X_np @ bss_params
y_lasso_a = X_np @ lasso_a_params
y_lasso_b = X_np @ lasso_b_params

def mse(y_hat, y):
    """Calculate Mean Squared Error"""
    return np.sum(np.square(y_hat - y)) / y.size

display(HTML('<h4>Actual MSEs compared to known f(x)</h4>'))
print('Best subset selection    : {}'.format(mse(y_bss, y_actual)))
print('Lasso a (higher lambda)  : {}'.format(mse(y_lasso_a, y_actual)))
print('Lasso b (lower lambda ~0): {}'.format(mse(y_lasso_b, y_actual)))

# Question 9

import numpy as np
import pandas as pd
import statsmodels.api as sm

from operator import itemgetter

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression

from matplotlib import pyplot as plt

college = pd.read_csv("College.csv", index_col = 0)

print(college.isnull().any().any())
college.head(5)

x = college.Apps
x.mean()
median = x.median()

dummy = pd.get_dummies(x > median, prefix='Applications')

# part a
np.random.seed(0)

# training set is split 50-50
train = np.random.choice([True,False], size=len(college))
college_train = college[train]
college_test = college[~train]

# part  b

# Least Squares Model

y_train = college_train['Apps']
features = list(college)
features.remove('Apps')
x_train = college_train[features]

# Fit model and report the parameters
lm = sm.OLS(y_train,sm.add_constant(x_train)).fit()
lm.params

# MSE
y_predict = lm.predict(sm.add_constant(college_test[features]))

mse = np.mean((college_test.Apps.values-y_predict)**2)
print(mse) #1855678.638348571

# part c

# Ridge Regression Model

a = np.logspace(-4, 0, 100)

RidgeCV = RidgeCV(alphas = a, normalize=True, store_cv_values=True)
# Fit model
results = RidgeCV.fit(x_train.values, y_train.values)

cv = np.mean(results.cv_values_, axis = 0)

min_cv = np.min(cv)
min_alpha = results.alpha_

# Plot
fig, ax = plt.subplots(figsize=(12,6))
ax.plot(a, np.mean(results.cv_values_, axis=0), color='red')
ax.plot(min_alpha, min_cv, color='blue', marker='o',)
ax.set_xlabel('alpha');
ax.set_ylabel('MSE for Train Set');

# Get the ridge models predictions for the test set
ridge_predicted = results.predict(college_test[features].values)

# MSE
MSE = np.mean((college_test.Apps.values-ridge_predicted)**2)
print(MSE) #1962381.246675046

# part d

# Lasso Model

lasso = LassoCV(alphas = np.logspace(-4, 0, 100), normalize = True, cv = 5, max_iter = 100000)
lasso.fit(x_train.values, y_train.values)

mse = np.mean(lasso.mse_path_, axis = 1)

# Plot
fig, ax = plt.subplots(figsize=(12,6));
ax.plot(lasso.alphas_, mse, color='red');
ax.plot(lasso.alpha_, np.min(mse), marker='o', color='blue', markersize=8);
ax.set_xlabel('alpha');
ax.set_ylabel('MSE for Train Set');

lasso.predicted = lasso.predict(college_test[features].values)

# MSE
MSE = np.mean((college_test.Apps.values-lasso.predicted)**2)
print(MSE) #1865826.9804312338
print(pd.Series(data = np.hstack([lasso.intercept_,lasso.coef_]), index=['Intercept'] + features))

# part e

# PCR Model

scores = []

num_components = np.arange(1,len(features)+1)
num_samples = x_train.shape[0]

for n in num_components:

    pca = PCA(n_components=n)
    
    pipeline = Pipeline([('scaler', StandardScaler()), ('pca', pca), ('linear_regression', LinearRegression())])
    
    pipeline.fit(x_train.values,y_train.values)
    scores.append(-np.mean(cross_val_score(pipeline, x_train.values, y_train.values, 
                                           scoring='mean_squared_error', cv=5)))

min_index, min_score = min(enumerate(scores), key = itemgetter(1))
      
fig, ax = plt.subplots(figsize=(12,6))
ax.plot(num_components,scores, marker='o', color='red');
ax.plot(min_index+1, min_score, marker='x', color='blue', markersize = 15);
ax.set_xlabel('Number of Components');
ax.set_ylabel('Training Score');
ax.set_xlim(xmin = -1);

# part f

# PLS Model

pls_scores = []

# Get number of components and num samples of training data
num_components = list(np.arange(1,x_train.values.shape[1]+1))
num_samples = x_train.values.shape[0]

for n in num_components:
    # construct PLS obj with n_components
    pls = PLSRegression(n_components=n, scale=True)

    pls_scores.append(-np.mean(cross_val_score(pls, x_train.values, y_train.values, 
                                           scoring='mean_squared_error', cv=5)))   
# get the min cv score
min_index, min_score = min(enumerate(pls_scores), key = itemgetter(1))

# Plot CV scores across number of components 
fig, ax = plt.subplots(figsize=(8,4))

# Plot the minimum cv score and the number of components that give that score
ax.plot(num_components,pls_scores, marker='o', color='b');
ax.plot(min_index+1, min_score, marker='x', color='r', markersize = 15);
ax.set_xlabel('Number of Components');
ax.set_ylabel('5-Fold CV Training Score');
ax.set_xlim(xmin=-1);

# Question 10

import numpy as np
import pandas as pd
import statsmodels.api as sm

from tqdm import tqdm
from operator import itemgetter
from itertools import combinations

from sklearn.preprocessing import PolynomialFeatures

from matplotlib import pyplot as plt

%matplotlib inline
plt.style.use('ggplot') # emulate R's pretty plotting

# print numpy arrays with precision 4
np.set_printoptions(precision=4)

# part a

np.random.seed(0)

x = np.random.randn(1000,1)

beta = np.zeros((13,))
np.put(beta, [0,1,2,3], [2, 4, 3, 1])

e = np.random.randn(1000,1)

y = beta[0]+beta[1]*X + beta[1]*X + beta[2]*X**2 + beta[3]*X**3 + e

poly = PolynomialFeatures(degree=12, include_bias=False)
x_arr = poly.fit_transform(x)

col_names = ['Y']+['X_' + str(i) for i in range(1,len(betas))]

dataframe = pd.DataFrame(np.concatenate((y,x_arr), axis=1), columns=col_names)
dataframe.head()

# part b

train_dataframe = dataframe.sample(n=100, random_state=0)
test_dataframe = dataframe.drop(train_dataframe.index)

# part c

# Plot Train MSE

def best_subsets(dataframe, predictors, response, max_features=8):
     
    def process_linear_model(features):

        x = sm.add_constant(dataframe[features])
        y = dataframe[response]

        model = sm.OLS(y,x).fit()
        RSS = model.ssr
        return (model, RSS)

    def get_best_kth_model(k):

        results = []

        for combo in combinations(predictors, k):

            results.append(process_linear_model(list(combo)))

        return sorted(results, key= itemgetter(1)).pop(0)[0]
    
    models =[]
    for k in tqdm(range(1,max_features+1)):
        models.append(get_best_kth_model(k))
    
    return models

features = list(train_dataframe.columns[1:])
models = best_subsets(train_dataframe, features, ['Y'], max_features=12)

mse = np.array([])
for model in models:

    features = list(model.params.index[1:])
    
    x_train = sm.add_constant(train_dataframe[features])
    
    y_predicted = model.predict(x_train)
    
    mse = np.append(mse, np.mean((y_predicted - train_dataframe.Y.values)**2))

fig, ax = plt.subplots(figsize=(8,5));
ax.plot(range(1,len(features)+1), mse, marker='x', color = "blue");
ax.set_xlabel('Number of Features');
ax.set_ylabel('MSE');

# part d

# Plot Test MSE

mse = np.array([])
for model in models:

    features = list(model.params.index[1:])

    x_test = sm.add_constant(test_dataframe[features])

    y_predicted = model.predict(x_test)

    mse = np.append(mse, np.mean((y_predicted - test_dataframe.Y.values)**2))
    
fig, ax = plt.subplots(figsize=(8,5));
ax.plot(range(1,len(features)+1),mse, marker='x', color  = "blue");
ax.set_xlabel('Number of Features');
ax.set_ylabel('Test MSE');

# part f

models[2].params

coeff_norms =[]
for idx, model in enumerate(models):
    model_betas = np.pad(model.params.values[1:],(0,len(models)-idx), 'constant', constant_values=[0])
    coeff_norms.append(np.sqrt(np.sum((betas-model_betas)**2)))
    
fig, ax = plt.subplots()
ax.plot(coeff_norms,marker='o');
ax.set_ylabel('L2 Norm Differences');
ax.set_xlabel('Num Model Coeffecients');

# Question 11

import numpy as np
import pandas as pd

from operator import itemgetter

import statsmodels.api as sm

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.cross_validation import cross_val_score
from matplotlib import pyplot as plt

%matplotlib inline
plt.style.use('ggplot') # emulate R's pretty plotting

# print numpy arrays with precision 4
np.set_printoptions(precision=4)

# part a

boston = pd.read_csv('Boston.csv')
boston.head()

# Define predictor and response variables
predictors = list(boston.columns[1:])
crime = boston.crim

# Scatterplot

# A really good idea is to start with some plots to get a feel for how the predictors relate to the response
fig, axarr = plt.subplots(4,4,figsize=(8,5))

for p, predictor in enumerate(predictors):
    index = np.unravel_index(p,(4,4))
    axarr[index[0], index[1]].scatter(boston[predictor],boston.crim, facecolors='none', edgecolors='b', alpha=0.5)
    axarr[index[0], index[1]].set_xlabel(predictor)
    axarr[index[0], index[1]].set_ylabel('crim')
    
plt.tight_layout()

# Ridge Regression Model

# Remeber for ridge we need to scale the predictors so that predictors measured on different scales are
# penalized equivalently. We will pass normalize option to ridgeCV

# create a set array of regularization hyperparameters
lambdas = 10**np.linspace(-4,-1, 100)
ridge = RidgeCV(alphas=lambdas, fit_intercept=True, normalize=True, store_cv_values=True)
ridge.fit(boston[predictors],boston.crim)

# Get the alpha with the minimum cv test error
ridge_mse = np.mean(ridge.cv_values_, axis=0)
min_mse_idx, min_ridge_mse = min(enumerate(ridge_mse), key=itemgetter(1))

# Plot the ridge's test mse
fig, ax1 = plt.subplots(figsize=(8,6))
ax1.plot(lambdas, ridge_mse)
ax1.plot(lambdas[min_mse_idx], min_ridge_mse, marker='o', color='b', markersize=12);
ax1.set_xscale('log');
ax1.set_xlabel('log(alpha)');
ax1.set_ylabel('Ridge Test MSE');
plt.show()

# Print the Ridge's Test MSE at the optimal alpha
print('Ridge Regression LOOCV Test MSE Estimate = ', min_ridge_mse,'\n')

# Print the Ridge Regression Coeffecients
print(pd.Series(data = np.hstack([ridge.intercept_,ridge.coef_]), index=['Intercept'] + predictors))

# Lasso Regression

# Fit a lasso model with 100 alpha values and LOOCV. Be sure to normalize
lambdas = 10**np.linspace(-5,-2, 200)
lasso = LassoCV(alphas=lambdas, fit_intercept=True, normalize=True, cv=len(boston))
lasso.fit(boston[predictors],boston.crim)

lasso_mse = np.mean(lasso.mse_path_, axis=1)
min_lasso_mse = min(lasso_mse)

fig, ax = plt.subplots(figsize=(8,6))
ax.plot(lasso.alphas_, lasso_mse )
ax.plot(lasso.alpha_, min_lasso_mse, marker='o', color='b', markersize=12);
ax.set_xscale('log');
ax.set_xlabel('log(alpha)');
ax.set_ylabel('Lasso Test MSE');
plt.show()

print('Lasso Regression LOOCV Test MSE Estimate = ', min(lasso_mse),'\n')
# Print the Lasso Regression Coeffecients
print(pd.Series(data = np.hstack([lasso.intercept_, lasso.coef_]), index=['Intercept'] + predictors))







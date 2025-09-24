import statsmodels.api as sm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LassoCV, RidgeCV, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.cross_decomposition import PLSRegression
import itertools

RANDOM_SEED = 7406

def main():
    RNG = np.random.RandomState(RANDOM_SEED)
    train, test = read_and_split_data("fat.csv")
    model_names = ["Full", "5-Best (p-score)", "5-Best (AIC)", "Stepwise", "LASSO", "Ridge", "PCA", "PLS"]
    full_lr_model, full_mse = all_features_lr(train, test)
    (k_p_model, k_p_mse), (k_aic_model, k_aic_mse) = best_k_models(train, test)
    step_model, step_mse = stepwise(train, test)
    lasso_model, lasso_mse, lasso_lambda = lasso(train, test, plot=False, rng=RNG)
    ridge_model, ridge_mse, ridge_lambda = ridge(train, test)
    pca_model, pca_mse, pca_trans = pca_regr(train, test, plot=False)
    pls_model, pls_mse = pls_regr(train, test, plot=False)

    mses = [full_mse, k_p_mse, k_aic_mse, step_mse, lasso_mse, ridge_mse, pca_mse, pls_mse]
    results = pd.DataFrame(mses, columns=["MSE"], index=model_names)
    print(results)

def read_and_split_data(filename: str):
    data = pd.read_csv(filename)
    data = data.drop(columns = ["siri", "density", "free"])
    test_rows = [1, 21, 22, 57, 70, 88, 91, 94, 121, 127, 149, 151, 159, 162,
                164, 177, 179, 194, 206, 214, 215, 221, 240, 241, 243]
    train_data = data.drop(test_rows, axis = 0)
    test_data = data.iloc[test_rows]
    return (train_data, test_data)

def all_features_lr(data, test):
    X = data.iloc[:,1:]
    y = data.iloc[:,0]
    model = LinearRegression().fit(X, y)
    mse = calc_mse(model, test.iloc[:,1:], test.iloc[:,0])
    return model, mse

def best_k_models(data, test):
    X = data.iloc[:,1:]
    y = data.iloc[:,0]
    p_best_features = SelectKBest(score_func=f_regression, k=5).fit(X, y)
    model_pscore = LinearRegression().fit(data.loc[:, p_best_features.get_feature_names_out()], y)
    mse_pscore = calc_mse(model_pscore, test.loc[:,model_pscore.feature_names_in_], test.iloc[:,0])

    selected_aic_feat = full_param_search(X, y, combo_size=5)
    model_aic = LinearRegression().fit(data.loc[:, selected_aic_feat], y)
    mse_aic = calc_mse(model_aic, test.loc[:,model_aic.feature_names_in_], test.iloc[:,0])
    return ((model_pscore, mse_pscore), (model_aic, mse_aic))

def full_param_search(X, y, combo_size):
    combinations = itertools.combinations(X.columns, r = combo_size)
    best_combo = None
    lowest_score = None
    for combo in combinations:
        this_aic = calc_aic(X.loc[:, combo], y)
        if (lowest_score is None) or (this_aic < lowest_score):
            lowest_score = this_aic
            best_combo = combo
    return best_combo

def stepwise(data, test):
    X = data.iloc[:,1:]
    y = data.iloc[:,0]
    dummy = pd.Series(np.ones_like(y))
    features = X.columns.values.tolist()
    best_aic = calc_aic(dummy, y)
    current_model = ["init"]
    next_model = []

    while next_model != current_model:
        current_model = next_model.copy()
        possible_models = [(current_model, best_aic)]
        if len(current_model) > 1:
            for feature in current_model:
                test_model = current_model.copy()
                test_model.remove(feature)
                test_aic = calc_aic(X.loc[:,feature], y)
                possible_models.append((test_model, test_aic))
        unused_features = [f for f in features if f not in current_model]
        for feature in unused_features:
            test_model = current_model.copy()
            test_model.append(feature)
            test_aic = calc_aic(X.loc[:,feature], y)
            possible_models.append((test_model, test_aic))
        
        next_model_idx = np.argmin([m[1] for m in possible_models])
        next_model = possible_models[next_model_idx][0].copy()
        best_aic = possible_models[next_model_idx][1]
    
    model = LinearRegression().fit(X.loc[:,next_model], y)
    mse = calc_mse(model, test.loc[:, model.feature_names_in_], test.iloc[:,0])
    return (model, mse)

def lasso(data, test, plot = False, rng=None):
    scaled_data = StandardScaler().fit_transform(data)
    scaled_X = scaled_data[:,1:]
    y = data.iloc[:,0]
    lasso_cv = LassoCV(cv=5, random_state=rng).fit(scaled_X, y)
    if plot:
        lambdas, coefs, _ = lasso_cv.path(scaled_X,y, alphas = lasso_cv.alphas_)
        plot_coef_path(lambdas, coefs, "LASSO", best_lambda = lasso_cv.alpha_)
    selected_coefficients = pd.DataFrame(list(zip(data.columns[1:], lasso_cv.coef_)), columns = ["Feature", "Coefficient"])
    selected_coefficients = selected_coefficients[selected_coefficients["Coefficient"] != 0]
    
    X = data.loc[:,selected_coefficients["Feature"]]
    model = LinearRegression().fit(X, y)
    mse = calc_mse(model, test.loc[:,model.feature_names_in_], test.iloc[:,0])
    return (model, mse, lasso_cv.alpha_)

def ridge(data, test):
    scaler = StandardScaler()
    X = scaler.fit_transform(data.iloc[:,1:])
    y = data.iloc[:,0]
    ridge_cv = RidgeCV().fit(X, y)
    mse = calc_mse(ridge_cv, scaler.transform(test.iloc[:,1:]), test.iloc[:,0])
    return (ridge_cv, mse, ridge_cv.alpha_)
    
def pca_regr(data, test, plot = False):
    scaler = StandardScaler()
    scaled_X = scaler.fit_transform(data.iloc[:,1:])
    y = data.iloc[:,0]
    pca = PCA()
    red_X = pca.fit_transform(scaled_X)
    if plot:
        plot_pca_exp_var(pca)
    n_pcs = next(i for i, val in enumerate(np.cumsum(pca.explained_variance_ratio_)) if val > 0.95) + 1

    model = LinearRegression().fit(red_X[:, :n_pcs], y)
    test_X = pca.transform(scaler.transform(test.iloc[:,1:]))[:,:n_pcs]
    mse = calc_mse(model, test_X, test.iloc[:,0])
    return (model, mse, pca)

def pls_regr(data, test, plot = False):
    scaler = StandardScaler()
    scaled_X = scaler.fit_transform(data.iloc[:,1:])
    y = data.iloc[:,0]

    cv_mse = []
    for i in range(1, scaled_X.shape[1]+1):
        pls = PLSRegression(n_components=i)
        score = -1 * cross_val_score(pls, scaled_X, y, scoring='neg_mean_squared_error').mean()
        cv_mse.append(score)

    if(plot):
        plot_pls_cv(cv_mse)

    n_comp = np.argmin(cv_mse)
    model = PLSRegression(n_components=n_comp).fit(scaled_X, y)
    mse = calc_mse(model, scaler.transform(test.iloc[:,1:]), test.iloc[:,0])
    return (model, mse)

def plot_pls_cv(res):
    plt.plot(range(1, len(res)+1), res, 'o-')
    plt.xlabel('Number of PLS Components')
    plt.ylabel('Mean CV MSE')
    plt.show()

def plot_pca_exp_var(pca):
    plt.plot(np.cumsum(pca.explained_variance_ratio_), 'o-')
    plt.axhline(y = 0.95, color = 'r', ls = '--')
    plt.axhline(y = 0.99, color = 'k', ls = '--')
    plt.xlabel('Cumulative Explained Variance')
    plt.ylabel('Number of PC')
    plt.show()

def plot_coef_path(lambdas, coefs, regr_name, best_lambda = None):
    lambdas = np.log10(lambdas)
    plt.figure()
    for i in range(coefs.shape[0]):
        l1 = plt.plot(lambdas, coefs[i,:])
    
    if best_lambda:
        l2 = plt.axvline(x = -np.log10(best_lambda), color = 'k', ls = '--')
    
    plt.xlabel("log10(lambda)")
    plt.ylabel("Coefficients")
    plt.title("{} Coefficient Path".format(regr_name))
    if(best_lambda):
        plt.legend((l1[-1], l2), (regr_name, "Optimal Lambda"), loc = "upper left")
    plt.show()

def calc_mse(model, test_data, actual):
    predictions = model.predict(test_data)
    mse = mean_squared_error(actual, predictions)
    return mse

def calc_aic(X, y, model = None, ic = 'aic'):
    if model is None:
        if X.ndim == 1:
            X = X.values.reshape(-1,1)
        model = LinearRegression().fit(X, y)
    
    loglik = log_likelihood(model, X, y)
    n = X.shape[0]
    k = X.shape[1] + 1
    penalty = 2*k

    if ic == 'bic':
        penalty *= np.log(n)

    return (-2*loglik + penalty)

def log_likelihood(model, X, y):
    fitted = model.predict(X)
    ssr = np.sum((y - fitted)**2)
    n = X.shape[0]
    loglik = (-(n/2)*(1+np.log(2*np.pi))) - ((n/2)*(np.log(ssr/n)))
    return loglik

if __name__ == "__main__":
    main()
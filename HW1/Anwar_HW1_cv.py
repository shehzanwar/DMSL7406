import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

N_ROUNDS = 100
R_SEED = 7406

def read_data() -> tuple:
    ziptrain = pd.read_csv("ISYE7406/HW1/zip.train.csv", header=None)
    ziptest = pd.read_csv("ISYE7406/HW1/zip.test.csv", header=None)
    col_names = ["Y"]
    col_names.extend([str(i) for i in range(ziptrain.shape[1]-1)])
    ziptrain.columns = col_names
    ziptrain = ziptrain[ziptrain["Y"].isin([2,7])]
    ziptest.columns = col_names
    ziptest = ziptest[ziptest["Y"].isin([2,7])]
    return (ziptrain, ziptest)

def monteCarlo(df, ntest=-1, rand=None) -> tuple:
    y = df["Y"]
    x = df.drop(columns=["Y"])
    if ntest == -1:
        ntest = np.floor(df.shape[0] * 0.3)
    split = train_test_split(x, y, test_size=ntest, random_state=rand)
    return split

def LR_error(train_x, test_x, train_y, test_y) -> float:
    model = LinearRegression().fit(train_x, train_y)
    preds = model.predict(test_x)
    preds = 2+5*(preds>=4.5)
    error = np.mean(preds != test_y)
    return error

def knn_error(train_x, test_x, train_y, test_y, k=1) -> float:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(train_x, train_y)
    preds = model.predict(test_x)
    error = np.mean(preds != test_y)
    return error

def main():
    rand = np.random.RandomState(R_SEED)
    train, test = read_data()
    ntest = test.shape[0]
    df = pd.concat([train, test])
    col_names = ["LinearReg", "KNN1", "KNN3", "KNN5", "KNN7", "KNN9", "KNN11", "KNN13", "KNN15"]
    err_test = pd.DataFrame(None, columns=col_names)

    for i in range(N_ROUNDS):
        current_row = []
        train_x, test_x, train_y, test_y = monteCarlo(df, ntest, rand)
        lin_error = LR_error(train_x, test_x, train_y, test_y)
        current_row.append(lin_error)
        k_choice = list(range(1, 16, 2))
        for k in k_choice:
            knn_err = knn_error(train_x, test_x, train_y, test_y, k)
            current_row.append(knn_err)
        current_row = pd.DataFrame([current_row], columns=col_names)
        err_test = pd.concat([err_test, current_row], ignore_index=True)
    
    summary = err_test.describe()
    summary = summary.apply(lambda x: np.square(x) if x.name == 'std' else x)
    summary = summary.rename(index={"std":"var"})
    print(summary)

if __name__ == "__main__":
    main()
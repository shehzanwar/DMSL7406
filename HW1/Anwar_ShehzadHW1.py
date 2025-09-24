import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

RANDOM_SEED = 7406
NUM_ITERATIONS = 100
TRAIN_SIZE = 1376
TEST_SIZE = 345

def run_analysis():
    rng = np.random.RandomState(RANDOM_SEED)

    # Load and filter datasets
    train_df, test_df = load_datasets()
    
    # Prepare data for training and testing
    X_train = train_df.drop(columns=["label"])
    y_train = train_df["label"]
    X_test = test_df.drop(columns=["label"])
    y_test = test_df["label"]
    
    # Compute training errors
    print("Training Errors on ziptrain27:")
    lin_train_err = compute_linear_error(X_train, y_train, X_train, y_train)
    print(f"Linear Regression: {lin_train_err:.6f}")
    k_values = [1, 3, 5, 7, 9, 11, 13, 15]
    for k_val in k_values:
        knn_train_err = compute_knn_error(k_val, X_train, y_train, X_train, y_train)
        print(f"KNN (k={k_val}): {knn_train_err:.6f}")
    
    # Compute testing errors
    print("\nTesting Errors on ziptest27:")
    lin_test_err = compute_linear_error(X_train, y_train, X_test, y_test)
    print(f"Linear Regression: {lin_test_err:.6f}")
    for k_val in k_values:
        knn_test_err = compute_knn_error(k_val, X_train, y_train, X_test, y_test)
        print(f"KNN (k={k_val}): {knn_test_err:.6f}")
    
    # Monte Carlo Cross-Validation
    combined_data = pd.concat([train_df, test_df], ignore_index=True)
    methods = ["LinearReg", "KNN_1", "KNN_3", "KNN_5", "KNN_7", "KNN_9", "KNN_11", "KNN_13", "KNN_15"]
    test_errors = pd.DataFrame(columns=methods)
    
    for iteration in range(NUM_ITERATIONS):
        row_data = []
        X_train_cv, X_test_cv, y_train_cv, y_test_cv = random_split(combined_data, TEST_SIZE, rng)
        lin_err = compute_linear_error(X_train_cv, y_train_cv, X_test_cv, y_test_cv)
        row_data.append(lin_err)
        for k_val in k_values:
            knn_err = compute_knn_error(k_val, X_train_cv, y_train_cv, X_test_cv, y_test_cv)
            row_data.append(knn_err)
        new_row = pd.DataFrame([row_data], columns=methods)
        test_errors = pd.concat([test_errors, new_row], ignore_index=True)
    
    # Compute summary statistics
    stats = test_errors.describe()
    stats = stats.apply(lambda row: row**2 if row.name == 'std' else row)
    stats = stats.rename(index={'std': 'var'})
    print("\nCross-Validation Results (100 iterations):")
    print(stats)
    
    # Identify optimal k
    knn_means = stats.loc['mean', [f"KNN_{k}" for k in k_values]]
    optimal_k = k_values[np.argmin(knn_means)]
    print(f"\nOptimal KNN k value: {optimal_k} (Mean Test Error: {knn_means[f'KNN_{optimal_k}']:.6f})")

def load_datasets() -> tuple:
    # Load datasets with correct file paths
    zip_train_df = pd.read_csv(r"ISYE7406\HW1\zip.train.csv", header=None)
    zip_test_df = pd.read_csv(r"ISYE7406\HW1\zip.test.csv", header=None)

    # Assign column names
    feature_cols = ["label"] + [f"pixel_{i}" for i in range(zip_train_df.shape[1] - 1)]
    zip_train_df.columns = feature_cols
    zip_test_df.columns = feature_cols
    
    # Filter for labels 2 and 7
    filtered_train = zip_train_df[zip_train_df["label"].isin([2, 7])]
    filtered_test = zip_test_df[zip_test_df["label"].isin([2, 7])]

    return (filtered_train, filtered_test)

def random_split(dataframe, test_size, rng) -> tuple:
    labels = dataframe["label"]
    features = dataframe.drop(columns=["label"])
    split_result = train_test_split(features, labels, test_size=test_size, random_state=rng)
    return split_result

def compute_linear_error(train_features, train_labels, test_features, test_labels) -> float:
    lin_model = LinearRegression().fit(train_features, train_labels)
    predictions = lin_model.predict(test_features)
    classified_preds = np.where(predictions >= 4.5, 7, 2)
    error_rate = np.mean(classified_preds != test_labels)
    return error_rate

def compute_knn_error(k_value, train_features, train_labels, test_features, test_labels) -> float:
    knn_model = KNeighborsClassifier(n_neighbors=k_value)
    knn_model.fit(train_features, train_labels)
    predictions = knn_model.predict(test_features)
    error_rate = np.mean(predictions != test_labels)
    return error_rate

if __name__ == "__main__":
    run_analysis()
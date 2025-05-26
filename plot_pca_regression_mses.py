# Load some dummy data frame.
import pandas as pd
import numpy as np
df = pd.DataFrame({
    'A': np.random.normal(0, 10, size=100), #np.random.randint(0, 100, size=100),
    'B': np.random.normal(0, 10, size=100), #np.random.randint(0, 100, size=100),
    'C': np.random.normal(0, 10, size=100), #np.random.randint(0, 100, size=100),
    'D': np.random.normal(0, 10, size=100), #np.random.randint(0, 100, size=100),
})

# To make it a tiny bit more interesting, add a dependent variable.
# df['B'] = df['A'] * 0.75 + np.random.normal(0, 10, size=100)
df['E'] = df['A'] + df['B'] * 0.5 + np.random.normal(0, 10, size=100)
# df['E'] = np.random.randint(0, 100, size=100)

# ----

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def run_regression(df, y, pca_vectors, n_components):
    # Transform the data using the PCA vectors.
    X_transformed = df.values @ pca_vectors[:, :n_components]
    
    # Fit the linear regression model.
    model = LinearRegression()
    model.fit(X_transformed, y)
    
    return model

def plot_regression_mses_over_pca_components(df, y, train_ratio=0.8):
    """
    Plot the Mean Squared Errors (MSE) for the training and validation data
    of regression models over different numbers of PCA components.

    Parameters:
    df (pd.DataFrame): The input data frame with features.
    y (np.ndarray): The target variable as a NumPy array.
    train_ratio (float): The ratio of data to be used for training (default is 0.8).
    """

    # Normalize the data frame.
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    
    # Split the data into training and validation sets.
    X_train, X_val, y_train, y_val = train_test_split(df_scaled.values, y, train_size=train_ratio, random_state=42)

    mse_train = []
    mse_val = []

    # Perform PCA on the training data.
    pca = PCA(n_components=df.shape[1])
    pca.fit(X_train)
    pca_vectors_matrix = pca.components_.T  # Transpose to get features as rows.

    # Iterate over the number of PCA components.
    for n_components in range(1, df.shape[1] + 1):
        # Run regression with the specified number of PCA components.
        model = run_regression(pd.DataFrame(X_train), y_train, pca_vectors_matrix, n_components)

        # Calculate MSE for training and validation sets.
        predictions_train = model.predict(X_train @ pca_vectors_matrix[:, :n_components])
        predictions_val = model.predict(X_val @ pca_vectors_matrix[:, :n_components])

        mse_train.append(mean_squared_error(y_train, predictions_train))
        mse_val.append(mean_squared_error(y_val, predictions_val))

    # Print the MSEs as a DataFrame.
    mse_df = pd.DataFrame({
        'Number of PCA Components': range(1, df.shape[1] + 1),
        'Training MSE': mse_train,
        'Validation MSE': mse_val
    })

    # Plot the MSEs.
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, df.shape[1] + 1), mse_train, label='Training MSE', marker='o')
    plt.plot(range(1, df.shape[1] + 1), mse_val, label='Validation MSE', marker='o')
    plt.xlabel('Number of PCA Components')
    plt.ylabel('Mean Squared Error')
    plt.title('MSE over Number of PCA Components')
    plt.legend()
    plt.grid()
    plt.show()

    return mse_df

# Run the function to plot MSEs over PCA components.
y = df['A'].values + np.random.normal(0, 1, size=100)
mse_df = plot_regression_mses_over_pca_components(df, y)
mse_df

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_absolute_error

# If we use L2 regularization (ridge regression), we end up with a model with smaller coefficients. 
# In other words, L2 regularization shrinks all the coefficients but rarely turns them into zero.
def calculate_mae_with_ridge(X_train, y_train, X_val, y_val, X_test, y_test, degree, alpha=1.0):
    """
    Trains a polynomial regression model with L2 regularization and returns MAE for train, validation, and test sets.
    
    Parameters:
    alpha: float, regularization strength (larger values mean more regularization)
    """
    model = Pipeline([
        #('scaler', StandardScaler()),
        ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
        ('ridge', Ridge(alpha=alpha))  # Usamos Ridge en lugar de LinearRegression, R
    ])
    
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    mae_train = mean_absolute_error(y_train, y_train_pred)
    mae_val = mean_absolute_error(y_val, y_val_pred)
    mae_test = mean_absolute_error(y_test, y_test_pred)

    return mae_train, mae_val, mae_test, model


# If we use L1 regularization (lasso regression), you end up with a model with fewer coefficients. 
# In other words, L1 regularization turns some of the coefficients into zero. 
def calculate_mae_with_lasso(X_train, y_train, X_val, y_val, X_test, y_test, degree, alpha=1.0):
    """
    Trains a polynomial regression model with L1 regularization and returns MAE for train, validation, and test sets.
    
    Parameters:
    alpha: float, regularization strength (larger values mean more regularization)
    """
    model = Pipeline([
        ('scaler', StandardScaler()),  # Escalamos los datos para mejorar la convergencia de Lasso
        ('poly', PolynomialFeatures(degree=degree, include_bias=False)), # Convierte X en [X, X², X³, ...]
        ('lasso', Lasso(alpha=alpha, max_iter=10000))  # Aumentamos max_iter para convergencia
    ])
    
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    mae_train = mean_absolute_error(y_train, y_train_pred)
    mae_val = mean_absolute_error(y_val, y_val_pred)
    mae_test = mean_absolute_error(y_test, y_test_pred)

    return mae_train, mae_val, mae_test, model


def find_best_regresion_model(X_train, y_train, X_val, y_val, X_test, y_test, degrees, alphas, calculate_mae_with_regression):
    best_degree = None
    best_alpha = None
    best_train_mae = float('inf')
    best_mae_val = float('inf')
    best_test_mae = float('inf')
    best_model = None
    
    for degree in degrees:
        for alpha in alphas:
            mae_train, mae_val, mae_test, model = calculate_mae_with_regression(X_train, y_train, X_val, y_val, X_test, y_test, degree, alpha)
            
            # print(f"Degree: {d}, Alpha: {alpha:.4f}, Train MAE: {mae_train:.2f}, Val MAE: {mae_val:.2f}, Test MAE: {mae_test:.2f}")
            
            if mae_val < best_mae_val:
                best_model = model
                best_train_mae = mae_train
                best_mae_val = mae_val
                best_test_mae = mae_test
                best_degree = degree
                best_alpha = alpha
                

    print("Best Model Found:")
    print(f"Degree: {best_degree}")
    print(f"Alpha: {best_alpha:.4f}")
    print(f"Train MAE: {best_train_mae:.2f}")
    print(f"Validation MAE: {best_mae_val:.2f}")
    print(f"Test MAE: {best_test_mae:.2f}")

    return best_model, best_degree, best_alpha, best_train_mae, best_mae_val, best_test_mae




from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load and split dataset
X, y = load_diabetes(return_X_y=True)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)


# If we use L2 regularization (ridge regression), we end up with a model with smaller coefficients. 
degrees = range(1, 11)
alphas = [0.001, 0.01, 0.1, 1, 10, 100]  # Diferentes valores de alpha para probar
print("Finding best Ridge regression model (L2 Norm)...")
find_best_regresion_model(X_train, y_train, X_val, y_val, X_test, y_test, degrees, alphas, calculate_mae_with_ridge)

# If we use L1 regularization (lasso regression), you end up with a model with fewer coefficients. 
degrees = range(1, 6)  # Con L1, grados altos pueden ser problemáticos
alphas = [0.0001, 0.001, 0.01, 0.1, 1, 10]  # Alpha necesita valores más pequeños que con L2
print("Finding best Lasso regression model (L1 Norm)...")
find_best_regresion_model(X_train, y_train, X_val, y_val, X_test, y_test,degrees, alphas, calculate_mae_with_lasso)



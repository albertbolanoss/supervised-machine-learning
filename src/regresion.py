from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_absolute_error
from dataclasses import dataclass
import pandas as pd

@dataclass
class ModelEvaluation:
    mae_train: float
    mae_val: float
    mae_test: float
    y_train_pred: list
    y_val_pred: list
    y_test_pred: list
    model: Pipeline
    degree: int
    alpha: float

# If we use L2 regularization (ridge regression), we end up with a model with smaller coefficients. 
# In other words, L2 regularization shrinks all the coefficients but rarely turns them into zero.
def calculate_mae_with_ridge(X_train, y_train, X_val, y_val, X_test, y_test, degree, alpha=1.0):
    """
    Trains a polynomial regression model with L2 regularization and returns MAE for train, validation, and test sets.
    
    Parameters:
    alpha: float, regularization strength (larger values mean more regularization)
    """
    model = Pipeline([
        ('scaler', StandardScaler()),
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

    return ModelEvaluation(mae_train, mae_val, mae_test, y_train_pred, y_val_pred, y_test_pred, model, degree, alpha)


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

    return ModelEvaluation(mae_train, mae_val, mae_test, y_train_pred, y_val_pred, y_test_pred, model, degree, alpha)


def find_best_regresion_model(X_train, y_train, X_val, y_val, X_test, y_test, degrees, alphas, calculate_mae_with_regression):
    best_mae_val = float('inf')
    best_model_evaluation = None
    
    for degree in degrees:
        for alpha in alphas:
            model_evaluation = calculate_mae_with_regression(X_train, y_train, X_val, y_val, X_test, y_test, degree, alpha)
            
            # print(f"Degree: {d}, Alpha: {alpha:.4f}, Train MAE: {mae_train:.2f}, Val MAE: {mae_val:.2f}, Test MAE: {mae_test:.2f}")
            
            if model_evaluation.mae_val < best_mae_val:
                best_model_evaluation = model_evaluation
                best_mae_val = model_evaluation.mae_val
                

    return best_model_evaluation

def export_predictions_to_csv(X, y, y_pred, split_name: str, feature_names: list, filename_prefix: str = "predictions"):
    """
    Exporta un CSV con las columnas del dataset, el valor real y el valor predicho.
    
    Parameters:
    - X: np.ndarray o pd.DataFrame, features
    - y: array-like, valores reales
    - y_pred: array-like, predicciones del modelo
    - split_name: str, uno de 'train', 'val' o 'test'
    - feature_names: list of str, nombres de las columnas
    - filename_prefix: str, prefijo del archivo de salida
    """
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    df['prediction'] = y_pred
    df['delta'] = y - y_pred
    filename = f"{filename_prefix}_{split_name}.csv"
    df.to_csv(filename, index=False)



from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

# Load and split dataset
X, y = load_diabetes(return_X_y=True)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)


# If we use L2 regularization (ridge regression), we end up with a model with smaller coefficients. 
# degrees = range(1, 11)
# alphas = [0.001, 0.01, 0.1, 1, 10, 100]  # Diferentes valores de alpha para probar
# print("Finding best Ridge regression model (L2 Norm)...")
# best_model_evaluation = find_best_regresion_model(X_train, y_train, X_val, y_val, X_test, y_test, degrees, alphas, calculate_mae_with_ridge)

# If we use L1 regularization (lasso regression), you end up with a model with fewer coefficients. 
degrees = range(1, 6)  # Con L1, grados altos pueden ser problemáticos
alphas = [0.0001, 0.001, 0.01, 0.1, 1, 10]  # Alpha necesita valores más pequeños que con L2
print("Finding best Lasso regression model (L1 Norm)...")
best_model_evaluation = find_best_regresion_model(X_train, y_train, X_val, y_val, X_test, y_test,degrees, alphas, calculate_mae_with_lasso)


print("Best Model Found:")
print(f"Degree: {best_model_evaluation.degree}")
print(f"Alpha: {best_model_evaluation.alpha:.4f}")
print(f"Train MAE: {best_model_evaluation.mae_train:.2f}")
print(f"Validation MAE: {best_model_evaluation.mae_val:.2f}")
print(f"Test MAE: {best_model_evaluation.mae_test:.2f}")

feature_names = load_diabetes().feature_names
filename_prefix = "datasets/processed/diabetes"

export_predictions_to_csv(X_train, y_train, best_model_evaluation.y_train_pred, "train", feature_names, filename_prefix)
export_predictions_to_csv(X_val, y_val, best_model_evaluation.y_val_pred, "val", feature_names, filename_prefix)
export_predictions_to_csv(X_test, y_test, best_model_evaluation.y_test_pred, "test", feature_names, filename_prefix)
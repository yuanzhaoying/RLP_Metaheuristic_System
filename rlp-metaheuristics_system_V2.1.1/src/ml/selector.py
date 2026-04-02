from __future__ import annotations
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os


class AlgorithmSelector:
    def __init__(
        self,
        model_type: str = "random_forest",
        n_estimators: int = 500,
        max_depth: int = 10,
        random_state: int = 42
    ):
        self.model_type = model_type
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.algorithm_names = []
        self.feature_names = []
        self.is_fitted = False

    def _create_model(self):
        if self.model_type == "random_forest":
            base_model = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state,
                n_jobs=-1
            )
        elif self.model_type == "gradient_boosting":
            base_model = GradientBoostingRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        return MultiOutputRegressor(base_model)

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame
    ):
        self.algorithm_names = list(y.columns)
        self.feature_names = list(X.columns)

        X_scaled = self.scaler.fit_transform(X)
        self.model = self._create_model()
        self.model.fit(X_scaled, y)
        self.is_fitted = True

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        return predictions

    def select_best(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        predictions = self.predict(X)
        best_indices = np.argmin(predictions, axis=1)
        best_predictions = predictions[np.arange(len(X)), best_indices]

        return best_indices, best_predictions

    def get_algorithm_name(self, index: int) -> str:
        if 0 <= index < len(self.algorithm_names):
            return self.algorithm_names[index]
        raise IndexError(f"Invalid algorithm index: {index}")

    def get_feature_importance(self) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        importances = []
        for estimator in self.model.estimators_:
            importances.append(estimator.feature_importances_)

        avg_importance = np.mean(importances, axis=0)

        return pd.DataFrame({
            "feature": self.feature_names,
            "importance": avg_importance
        }).sort_values("importance", ascending=False)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "algorithm_names": self.algorithm_names,
            "feature_names": self.feature_names,
            "model_type": self.model_type,
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth
        }
        joblib.dump(model_data, path)

    def load(self, path: str):
        model_data = joblib.load(path)
        self.model = model_data["model"]
        self.scaler = model_data["scaler"]
        self.algorithm_names = model_data["algorithm_names"]
        self.feature_names = model_data["feature_names"]
        self.model_type = model_data["model_type"]
        self.n_estimators = model_data["n_estimators"]
        self.max_depth = model_data["max_depth"]
        self.is_fitted = True


def nested_cv_evaluation(
    X: pd.DataFrame,
    y: pd.DataFrame,
    model_type: str = "random_forest",
    outer_folds: int = 5,
    inner_folds: int = 3,
    n_estimators: int = 500,
    max_depth: int = 10
) -> Dict:
    outer_cv = KFold(n_splits=outer_folds, shuffle=True, random_state=42)

    selector_scores = []
    sbs_scores = []
    vbs_scores = []
    accuracy_scores = []

    algorithm_names = list(y.columns)
    y_values = y.values

    sbs_index = np.argmin(y_values.mean(axis=0))
    sbs_performance = y_values[:, sbs_index]

    for train_idx, test_idx in outer_cv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        selector = AlgorithmSelector(
            model_type=model_type,
            n_estimators=n_estimators,
            max_depth=max_depth
        )
        selector.fit(X_train, y_train)

        predicted_best, _ = selector.select_best(X_test)

        y_test_values = y_test.values
        selected_performance = y_test_values[
            np.arange(len(test_idx)),
            predicted_best
        ]

        selector_scores.append(selected_performance.mean())
        sbs_scores.append(sbs_performance[test_idx].mean())

        vbs_per_instance = y_test_values.min(axis=1)
        vbs_scores.append(vbs_per_instance.mean())

        actual_best = np.argmin(y_test_values, axis=1)
        correct_predictions = np.sum(predicted_best == actual_best)
        accuracy = correct_predictions / len(test_idx)
        accuracy_scores.append(accuracy)

    results = {
        "selector_mean": np.mean(selector_scores),
        "selector_std": np.std(selector_scores),
        "sbs_mean": np.mean(sbs_scores),
        "sbs_std": np.std(sbs_scores),
        "vbs_mean": np.mean(vbs_scores),
        "vbs_std": np.std(vbs_scores),
        "selection_accuracy_mean": np.mean(accuracy_scores),
        "selection_accuracy_std": np.std(accuracy_scores),
        "selector_improvement_over_sbs": (
            (np.mean(sbs_scores) - np.mean(selector_scores)) / np.mean(sbs_scores)
        ) if np.mean(sbs_scores) > 0 else 0,
        "gap_to_vbs": (
            (np.mean(selector_scores) - np.mean(vbs_scores)) / np.mean(vbs_scores)
        ) if np.mean(vbs_scores) > 0 else 0
    }

    return results


def evaluate_selector(
    selector: AlgorithmSelector,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame
) -> Dict:
    predicted_indices, predicted_values = selector.select_best(X_test)

    y_test_values = y_test.values
    selected_performance = y_test_values[
        np.arange(len(X_test)),
        predicted_indices
    ]

    vbs_per_instance = y_test_values.min(axis=1)

    sbs_index = np.argmin(y_test_values.mean(axis=0))
    sbs_performance = y_test_values[:, sbs_index]

    correct_predictions = np.sum(
        predicted_indices == np.argmin(y_test_values, axis=1)
    )
    accuracy = correct_predictions / len(X_test)

    return {
        "selected_performance_mean": float(selected_performance.mean()),
        "selected_performance_std": float(selected_performance.std()),
        "sbs_performance_mean": float(sbs_performance.mean()),
        "vbs_performance_mean": float(vbs_performance.mean()),
        "improvement_over_sbs": float(
            (sbs_performance.mean() - selected_performance.mean()) / sbs_performance.mean()
        ) if sbs_performance.mean() > 0 else 0,
        "gap_to_vbs": float(
            (selected_performance.mean() - vbs_per_instance.mean()) / vbs_per_instance.mean()
        ) if vbs_per_instance.mean() > 0 else 0,
        "selection_accuracy": accuracy,
        "misprediction_penalty": float(
            (selected_performance.mean() - vbs_per_instance.mean()) / vbs_per_instance.mean()
        ) if vbs_per_instance.mean() > 0 else 0
    }


def prepare_ml_data(
    results_df: pd.DataFrame,
    features_df: pd.DataFrame,
    performance_col: str = "best_objective",
    instance_col: str = "instance_id"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    print(f"\nPreparing ML data...")
    print(f"Results columns: {list(results_df.columns)}")
    print(f"Results shape: {results_df.shape}")

    if 'instance_id' not in results_df.columns:
        print("ERROR: 'instance_id' column not found in results!")
        print("Available columns:", list(results_df.columns))
        return pd.DataFrame(), pd.DataFrame()

    results_df = results_df.copy()
    results_df['instance_name'] = results_df[instance_col].apply(
        lambda x: os.path.basename(x) if isinstance(x, str) else x
    )

    aggregated = results_df.groupby(["instance_name", "algorithm_name"])[performance_col].median().reset_index()

    y = aggregated.pivot(index="instance_name", columns="algorithm_name", values=performance_col)
    y = y.dropna()

    if 'instance_id' in features_df.columns:
        features_df = features_df.copy()
        features_df['instance_name'] = features_df['instance_id'].apply(
            lambda x: os.path.basename(x) if isinstance(x, str) else x
        )
        features_indexed = features_df.set_index('instance_name')
    else:
        features_indexed = features_df

    common_instances = list(set(y.index) & set(features_indexed.index))
    if not common_instances:
        print(f"WARNING: No matching instances found!")
        print(f"Results instances: {list(y.index[:5])}")
        print(f"Features instances: {list(features_indexed.index[:5])}")
        return pd.DataFrame(), pd.DataFrame()

    y = y.loc[common_instances]
    X = features_indexed.loc[common_instances]

    feature_cols = [c for c in X.columns if c not in ['instance_id', 'instance_name', 'horizon']]
    X = X[feature_cols]

    return X.reset_index(drop=True), y.reset_index(drop=True)

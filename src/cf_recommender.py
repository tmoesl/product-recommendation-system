"""
cf_recommender.py

Collaborative Filtering Recommendation System Module

This module provides a class `CFRecommendationSystem` for managing various recommendation system algorithms, 
including kNN-based and SVD-based methods. It supports training, evaluation, cross-validation, hyperparameter tuning, and 
generating personalized recommendations.

Classes:
    AlgorithmMismatchError: Custom exception for algorithm mismatch.
    ModelNotTrainedError: Custom exception for untrained model.
    CFRecommendationSystem: Manages recommendation system algorithms.

CFRecommendationSystem Methods:
    __init__: Initialize the recommendation system with data, algorithm class, parameters, and random state.
    fit_knn: Train the kNN-based recommendation model.
    fit_svd: Train the SVD-based recommendation model.
    evaluate: Evaluate the model on training and test sets, and display performance metrics.
    cross_val: Perform cross-validation on the provided dataset using a specified algorithm and evaluation metrics.
    predict: Generate and display rating predictions for user-item interactions.
    tune_hyperparameters: Perform hyperparameter tuning using GridSearchCV for the specified algorithm.
    recommend: Generate and display top N product recommendations for each user.
"""

# ---------------------------------------------------------
# IMPORT LIBRARIES
# ---------------------------------------------------------

# Import required libraries
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import pandas as pd
from IPython.display import Markdown, display

# Import Dataset from the data module
from surprise.dataset import Dataset

# Import GridSearchCV from the model_selection module
from surprise.model_selection import GridSearchCV, cross_validate

# Import KNN-based and matrix factorization-based algorithms from the submodule
from surprise.prediction_algorithms import (
    SVD,
    AlgoBase,
    KNNBaseline,
    KNNBasic,
    KNNWithMeans,
    KNNWithZScore,
    SVDpp,
)

# Import modules and required functions from the src directory
from src.model_eval_functions import evaluate_model, get_recommendations
from src.utils import display_eval_metrics

# ---------------------------------------------------------
# CLASS DEFINITIONS
# ---------------------------------------------------------


# Class for error handling
class AlgorithmMismatchError(Exception):
    """Custom exception for algorithm mismatch."""

    pass


# Class for error handling
class ModelNotTrainedError(Exception):
    """Custom exception for untrained model."""

    pass


# Class for managing recommendation system algorithms
class CFRecommendationSystem:
    """
    Manages various recommendation system algorithms including kNN-based and SVD-based methods.
    Supports training, evaluation, hyperparameter tuning, and generating personalized recommendations.
    """

    __slots__ = (
        "data",
        "algo_name",
        "algo_class",
        "params_grid",
        "random_state",
        "model",
        "best_score",
        "best_params",
        "gs",
    )

    # Initialize the recommendation system with data, algorithm class, parameters, and random state
    def __init__(
        self,
        data: pd.DataFrame,
        algo_class: Type[AlgoBase],
        params_grid: Optional[Dict[str, Any]] = None,
        random_state: int = 42,
    ):
        """
        Initialize the CFRecommendationSystem.

        Args:
            data (pd.DataFrame): Input data.
            algo_class (Type[AlgoBase]): Algorithm class to use.
            params_grid (Dict[str, Any], optional): Parameter grid for tuning. Defaults to None.
            random_state (int, optional): Random state for reproducibility. Defaults to 42.
        """

        self.data = data
        self.algo_name = algo_class.__name__
        self.algo_class = algo_class
        self.params_grid = params_grid or {}
        self.random_state = random_state
        self.model: Optional[AlgoBase] = None
        self.best_score: Optional[float] = None
        self.best_params: Optional[Dict[str, Any]] = None
        self.gs: Optional[GridSearchCV] = None

    # Function to train the kNN-based recommendation model
    def fit_knn(self, trainset: Dataset, use_tuned_params: bool = False) -> None:
        """
        Train the kNN-based recommendation model.

        Args:
            trainset (Dataset): Training dataset.
            use_tuned_params (bool): Whether to use tuned parameters. Defaults to False.

        Raises:
            AlgorithmMismatchError: If the algorithm class is not kNN-based
        """
        if not issubclass(
            self.algo_class, (KNNBasic, KNNBaseline, KNNWithMeans, KNNWithZScore)
        ):
            raise AlgorithmMismatchError(
                "The algo_class is not kNN inspired. Please select a different algo_class."
            )

        if use_tuned_params and self.best_params is not None:
            model_params = {
                "k": self.best_params.get("k", 40),
                "min_k": self.best_params.get("min_k", 1),
                "sim_options": self.best_params.get("sim_options", {}),
                "random_state": self.random_state,
                "verbose": False,
            }
        else:
            model_params = {
                "k": self.params_grid.get("k", 40),
                "min_k": self.params_grid.get("min_k", 1),
                "sim_options": self.params_grid.get("sim_options", {}),
                "random_state": self.random_state,
                "verbose": False,
            }

        self.model = self.algo_class(**model_params)
        self.model.fit(trainset)
        print(f"Parameters: {model_params}")

    # Function to train the SVD-based recommendation model
    def fit_svd(self, trainset: Dataset, use_tuned_params: bool = False) -> None:
        """
        Train the SVD-based recommendation model.

        Args:
            trainset (Dataset): Training dataset.
            use_tuned_params (bool): Whether to use tuned parameters. Defaults to False.

        Raises:
            AlgorithmMismatchError: If the algorithm class is not SVD-based.
        """
        if not issubclass(self.algo_class, (SVD, SVDpp)):
            raise AlgorithmMismatchError(
                "The algo_class is not kNN inspired. Please select a different algo_class."
            )

        if use_tuned_params and self.best_params is not None:
            model_params = {
                "n_factors": self.best_params.get("n_factors", 100),
                "n_epochs": self.best_params.get("n_epochs", 20),
                "lr_all": self.best_params.get("lr_all", 0.005),
                "reg_all": self.best_params.get("reg_all", 0.02),
                "random_state": self.random_state,
            }
        else:
            model_params = {
                "n_factors": self.params_grid.get("n_factors", 100),
                "n_epochs": self.params_grid.get("n_epochs", 20),
                "lr_all": self.params_grid.get("lr_all", 0.005),
                "reg_all": self.params_grid.get("reg_all", 0.02),
                "random_state": self.random_state,
            }

        self.model = self.algo_class(**model_params)
        self.model.fit(trainset)
        print(f"Parameters: {model_params}")

    # Function to evaluate the model on training and test sets, and display performance metrics
    def evaluate(
        self, trainset: Dataset, testset: Dataset, k: int = 10, th: float = 3.5
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Evaluate the model on training and test sets, and display performance metrics.

        Args:
            trainset (Dataset): Training dataset.
            testset (Dataset): Test dataset.
            k (int): Number of recommendations to generate. Defaults to 10.
            th (float): Rating threshold. Defaults to 3.5.

        Returns:
            Tuple[Dict[str, float], Dict[str, float]]: A tuple containing dictionaries of train and test metrics.
        """
        if self.model is None:
            raise ModelNotTrainedError(
                "Model is not trained. Please call fit_knn or fit_svd first."
            )

        # Evaluate on training set
        train_predictions = self.model.test(trainset.build_testset())
        train_pred_met, train_rank_met = evaluate_model(
            train_predictions, k=k, threshold=th
        )

        # Evaluate on test set
        test_predictions = self.model.test(testset)
        test_pred_met, test_rank_met = evaluate_model(
            test_predictions, k=k, threshold=th
        )

        # Combine and display results
        display_eval_metrics(
            train_pred_met,
            train_rank_met,
            test_pred_met,
            test_rank_met,
        )

        return {**train_rank_met, **train_pred_met}, {**test_rank_met, **test_pred_met}

    # Function to perform cross validation on a dataset for specified algorithm and metrics
    def cross_val(
        self, data: Dataset, measures: Optional[List[str]] = None, cv: int = 5
    ) -> Dict[str, List[float]]:
        """
        Perform cross-validation on the provided dataset using a specified algorithm and evaluation metrics.

        Args:
            data (Dataset): A Surprise library dataset containing user-item interactions.
            measures (list, optional): List of performance measures to evaluate during cross-validation. Defaults to ["rmse", "mae"].
            cv (int, optional): Number of cross-validation folds. Defaults to 5.

        Returns:
            dict: A dictionary containing cross-validation results for each evaluation measure.
        """

        if self.model is None:
            raise ModelNotTrainedError(
                "Model is not trained. Please call fit_knn or fit_svd first."
            )

        if measures is None:
            measures = ["rmse", "mae"]  # Default measures

        # Perform cross-validation
        return cross_validate(
            algo=self.model,
            data=data,
            measures=measures,
            cv=cv,
            n_jobs=-1,
            verbose=True,
        )

    # Function to generate and display rating predictions for user-item interactions
    def predict(
        self,
        int_data: Dict[str, List[Union[Tuple[str, float], str]]],
        has_interacted: bool,
    ) -> None:
        """
        Generate and display rating predictions for user-item interactions.

        Args:
            int_data (Dict[str, List[Union[Tuple[str, float], str]]]): User-item interaction data.
            has_interacted (bool): Whether the predictions are for interacted or non-interacted items. Defaults to False.
        """
        if self.model is None:
            raise ModelNotTrainedError(
                "Model is not trained. Please call fit_knn or fit_svd first."
            )

        phrase = "" if has_interacted else "Non-"
        display(Markdown(f"**Rating Estimates for {phrase}Interacted Products**"))

        for user, interactions in int_data.items():
            predictions = []
            for interaction in interactions:
                if has_interacted:
                    iid, rui = interaction  # product ID, rating
                    pred = self.model.predict(
                        uid=user, iid=iid, r_ui=rui, verbose=False
                    )
                else:
                    iid = interaction  # product ID
                    pred = self.model.predict(uid=user, iid=iid, verbose=False)
                predictions.append(pred)

            for pred in predictions:
                print(pred)

    # Function to perform hyperparameter tuning using GridSearchCV for the specified algorithm
    def tune_hyperparameters(
        self,
        param_grid: Dict[str, Any],
        measures: Optional[List[str]] = None,
        cv: int = 5,
    ) -> None:
        """
        Perform hyperparameter tuning using GridSearchCV for the specified algorithm.

        Args:
            param_grid (Dict[str, Any]): Parameter grid for tuning.
            measures (List[str]): List of measures to use for evaluation. Defaults to None.
            cv (int): Number of cross-validation folds. Defaults to 5.
        """
        if measures is None:
            measures = ["rmse"]  # Default measure

        self.gs = GridSearchCV(
            self.algo_class, param_grid=param_grid, measures=measures, cv=cv, n_jobs=-1
        )
        self.gs.fit(self.data)
        self.best_score = self.gs.best_score[measures[0]]
        self.best_params = self.gs.best_params[measures[0]]

        # Best RMSE score
        print(f"{measures[0].upper()}: {self.best_score:.3f}")
        print(f"Parameters: {self.best_params}")

    # Function to generate and display top N product recommendations for each user
    def recommend(
        self,
        data: pd.DataFrame,
        int_data: Dict[str, List[Tuple[str, float]]],
        top_n: int = 5,
    ) -> None:
        """
        Generate top-N product recommendations for each user and display them.

        Args:
            data (pd.DataFrame): Full dataset.
            int_data (Dict[str, List[Tuple[str, float]]]): User interaction data.
            top_n (int): Number of top recommendations to generate. Defaults to 5.
        """
        if self.model is None:
            raise ModelNotTrainedError(
                "Model is not trained. Please call fit_knn or fit_svd first."
            )

        for user in int_data.keys():
            recommendations = get_recommendations(
                data=data, algo=self.model, user_id=user, top_n=top_n
            )
            display(Markdown(f"**Recommendations for User: {user}**"))
            display(recommendations)


# ---------------------------------------------------------
# END OF SCRIPT
# ---------------------------------------------------------

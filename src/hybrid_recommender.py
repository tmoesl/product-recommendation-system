"""
hybrid_recommender.py

Hybrid Recommendation System Module

This module provides a class `HybridRecommendationSystem` that combines collaborative filtering (CF) scores with rank-based scores (e.g. Bayesian average). 
It supports generating personalized recommendations by blending CF and rank-based approaches, evaluating the performance of the hybrid model on training and test datasets, and providing top-N recommendations for each user.

Classes:
    WeightSumError: Custom exception for weight sum mismatch.
    ModelNotAvailableError: Custom exception for unavailable model.
    HybridRecommendationSystem: Manages hybrid recommendation systems, combines CF and rank-based scores, evaluates model performance, and generates top-N recommendations.

HybridRecommendationSystem Methods:
    __init__: Initialize the HybridRecommendationSystem with a trained collaborative filtering model and rank-based model.
    compute_hybrid_scores: Compute hybrid scores for user-item interactions by combining CF scores with rank scores.
    evaluate: Evaluate the trained model on training and test sets, and display performance metrics.
    predict: Generate and display rating predictions for user-item interactions.
    recommend: Generate and display top-N product recommendations for each user.
    _calculate_cf_scores: Helper function to calculate CF scores for a given user.
    _calculate_hybrid_score: Helper function to calculate the hybrid score.
    _calculate_hybrid_predictions: Helper function to calculate hybrid predictions.
"""

# ---------------------------------------------------------
# IMPORT LIBRARIES
# ---------------------------------------------------------

# Import required libraries
from typing import Dict, List, Tuple, Union

import pandas as pd
from IPython.display import Markdown, display
from surprise.dataset import Dataset
from surprise.prediction_algorithms.predictions import Prediction
from surprise.prediction_algorithms.algo_base import AlgoBase

# Import modules and required functions from the src directory
from src.model_eval_functions import evaluate_model, get_recommendations
from src.utils import display_eval_metrics

from src.cf_recommender import CFRecommendationSystem
from src.rank_recommender import RankRecommendationSystem

# ---------------------------------------------------------
# CLASS DEFINITIONS
# ---------------------------------------------------------


# Class for error handling
class WeightSumError(Exception):
    """Custom exception for weight sum mismatch."""

    pass


# Class for error handling
class ModelNotAvailableError(Exception):
    """Custom exception for unavailable model."""

    pass


# Class for managing hybrid recommendation systems
class HybridRecommendationSystem:
    """
    A hybrid recommendation system that combines collaborative filtering (CF) scores with rank-based (e.g. Bayesian average) scores.

    This class supports generating personalized recommendations by blending CF and rank-based approaches,
    evaluating the performance of the hybrid model on training and test datasets, and providing top-N recommendations for each user.
    """

    __slots__ = (
        "model_cf",
        "model_rank",
        "model_rank_dict",
        "global_rating",
        "weight_cf",
        "weight_rank",
        "algo_name",
        "key_column",
    )

    # Initialize the recommendation system with a collaborative filtering model, rank-based model, and weights
    def __init__(
        self,
        model_cf: CFRecommendationSystem,
        model_rank: RankRecommendationSystem,
        weight_cf: float = 0.8,
        weight_rank: float = 0.2,
    ):
        """
        Initialize the HybridRecommendationSystem with a trained collaborative filtering model and rank-based model.

        Args:
            model_cf (CFRecommendationSystem): An instance of the CFRecommendationSystem class containing the trained CF model.
            model_rank (RankRecommendationSystem): An instance of the RankRecommendationSystem class containing the rank-based scores.
            weight_cf (float, optional): Weight assigned to CF scores. Defaults to 0.8.
            weight_rank (float, optional): Weight assigned to rank-based scores. Defaults to 0.2.

        Raises:
            WeightSumError: If the sum of weights does not equal 1.
            ModelNotAvailableError: If model not available.
        """
        self.model_cf = model_cf.model
        self.model_rank = model_rank.scores
        self.key_column = model_rank.key_column
        self.model_rank_dict = self.model_rank[self.key_column].to_dict()
        self.global_rating = self.model_rank[self.key_column].mean()
        self.weight_cf = weight_cf
        self.weight_rank = weight_rank
        self.algo_name = f"{model_cf.algo_name} | {model_rank.algo_name}"

        # Error handling
        if abs(self.weight_cf + self.weight_rank - 1) > 1e-9:
            raise WeightSumError("Weights must sum to 1.")

        if self.model_cf is None or self.model_rank is None:
            raise ModelNotAvailableError(
                "Collaborative filtering model or Rank-based model is not available."
            )

    # Function to calculate hybrid scores
    def compute_hybrid_scores(
        self, data: pd.DataFrame, int_data: Dict[str, List[Tuple[str, float]]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute hybrid scores for user-item interactions by combining CF scores with rank scores.

        Args:
            data (pd.DataFrame): The input DataFrame containing user-product interactions.
            int_data (Dict[str, List[Tuple[str, float]]]): A dictionary containing users and their interactions.

        Returns:
            Dict[str, Dict[str, float]]: A dictionary of combined hybrid scores for each user.
        """
        # Initialize dictionaries to store CF scores and hybrid scores
        cf_scores = {}
        hybrid_scores = {}

        # Calculate hybrid scores
        for user in int_data.keys():
            # Get CF scores
            cf_scores[user] = self._calculate_cf_scores(
                data=data, algo=self.model_cf, user_id=user
            )

            # Initialize hybrid scores for current user
            hybrid_scores[user] = {}

            # Calculate the hybrid score for each item
            for item, cf_score in cf_scores[user].items():
                hybrid_scores[user][item] = self._calculate_hybrid_score(cf_score, item)

        return hybrid_scores

    # Function to evaluate the model on training and test sets, and display performance metrics
    def evaluate(
        self, trainset: Dataset, testset: Dataset, k: int = 10, th: float = 3.5
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Evaluate the trained model on training and test sets, and display performance metrics.

        Args:
            trainset (Dataset): Training dataset.
            testset (Dataset): Test dataset.
            k (int): Number of recommendations to generate. Defaults to 10.
            th (float): Rating threshold. Defaults to 3.5.

        Returns:
            Tuple[Dict[str, float], Dict[str, float]]: A tuple containing dictionaries of train and test metrics.
        """

        # Evaluate on training set
        train_predictions = self.model_cf.test(trainset.build_testset())
        hybrid_train_predictions = self._calculate_hybrid_predictions(train_predictions)

        # Evaluate on test set
        test_predictions = self.model_cf.test(testset)
        hybrid_test_predictions = self._calculate_hybrid_predictions(test_predictions)

        # Evaluate the hybrid predictions on both sets
        train_pred_met, train_rank_met = evaluate_model(
            hybrid_train_predictions, k=k, threshold=th
        )

        test_pred_met, test_rank_met = evaluate_model(
            hybrid_test_predictions, k=k, threshold=th
        )

        # Combine and display results
        display_eval_metrics(
            train_pred_met,
            train_rank_met,
            test_pred_met,
            test_rank_met,
        )

        return {**train_rank_met, **train_pred_met}, {**test_rank_met, **test_pred_met}

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
        phrase = "" if has_interacted else "Non-"
        display(Markdown(f"**Rating Estimates for {phrase}Interacted Products**"))

        predictions = []
        # Loop through each user and their interactions
        for user, interactions in int_data.items():
            for interaction in interactions:
                if has_interacted:
                    iid, rui = interaction  # product ID, true rating
                    cf_prediction = self.model_cf.predict(
                        uid=user, iid=iid, r_ui=rui, verbose=False
                    )
                else:
                    iid = interaction  # product ID
                    cf_prediction = self.model_cf.predict(
                        uid=user, iid=iid, verbose=False
                    )

                # Calculate hybrid score
                hybrid_score = self._calculate_hybrid_score(cf_prediction.est, iid)
                predictions.append((cf_prediction, hybrid_score))

        for cf_prediction, hybrid_score in predictions:
            # Format the output to match the Surprise 'predict style'
            r_ui_formatted = (
                f"{cf_prediction.r_ui:.2f}"
                if cf_prediction.r_ui is not None
                else "None"
            )
            print(
                f"user: {cf_prediction.uid:<14} item: {cf_prediction.iid:<10} "
                f"r_ui = {r_ui_formatted:<6} est = {hybrid_score:.2f}   {cf_prediction.details}"
            )

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
            data (pd.DataFrame): Full dataset
            int_data (Dict[str, List[Tuple[str, float]]]): User interaction data.
            top_n (int): Number of top recommendations to generate. Defaults to 5.
        """
        # Compute hybrid scores for all users
        hybrid_scores = self.compute_hybrid_scores(data, int_data)

        # Generate and display the top-N recommendations based on combined scores
        for user, scores in hybrid_scores.items():
            # Rank products by descending predicted ratings (combined score)
            top_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]

            # Convert recommendations to DataFrame and round predicted ratings
            recommendations_df = pd.DataFrame(
                top_items, columns=["prod_id", "estimated_ratings"]
            ).round({"estimated_ratings": 2})

            # Display the recommendations
            display(Markdown(f"**Recommendations for User: {user}**"))
            display(recommendations_df)

    # Helper function to calculate CF scores for a given user
    def _calculate_cf_scores(
        self, data: pd.DataFrame, algo: AlgoBase, user_id: str
    ) -> Dict[str, float]:
        """
        Calculate CF scores for a given user.

        Args:
            data (pd.DataFrame): The input DataFrame containing user-product interactions.
            algo (AlgoBase): The collaborative filtering algorithm.
            user_id (str): The ID of the user.

        Returns:
            Dict[str, float]: A dictionary of CF scores for the user.
        """
        # Get CF-based recommendations for the current user
        cf_recommendations = get_recommendations(data=data, algo=algo, user_id=user_id)

        # Create a dictionary of CF scores for the user
        cf_scores = dict(
            zip(
                cf_recommendations["prod_id"],
                cf_recommendations["estimated_ratings"],
            )
        )

        return cf_scores

    # Helper function to calculate the hybrid score
    def _calculate_hybrid_score(self, cf_score: float, item_id: str) -> float:
        """
        Calculate the hybrid score by combining CF score and rank score.

        Args:
            cf_score (float): Collaborative Filtering score.
            item_id (str): Item ID to fetch the rank score.

        Returns:
            float: Combined hybrid score.
        """
        rank_score = self.model_rank_dict.get(item_id, self.global_rating)
        return self.weight_cf * cf_score + self.weight_rank * rank_score

    # Helper function to calculate hybrid predictions
    def _calculate_hybrid_predictions(
        self, predictions: List[Prediction]
    ) -> List[Prediction]:
        """
        Calculate hybrid predictions by combining CF scores and rank scores.

        Args:
            predictions (List[Prediction]): List of CF predictions.

        Returns:
            List[Prediction]: List of hybrid predictions.
        """
        hybrid_predictions = []
        for pred in predictions:
            hybrid_score = self._calculate_hybrid_score(pred.est, pred.iid)
            hybrid_predictions.append((pred.uid, pred.iid, pred.r_ui, hybrid_score, {}))
        return hybrid_predictions


# ---------------------------------------------------------
# END OF SCRIPT
# ---------------------------------------------------------

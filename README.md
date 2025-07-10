#NFL Game Outcome Predictor

A Python-based machine learning pipeline for forecasting NFL game outcomes using historical data. This repository includes:

Data Processing & Feature Engineering
Ingests games.csv, computes game-level win flags, aggregates per-team metrics (win percentage, net points, recent form), and merges them into a modeling dataset.

Model Training & Evaluation
Trains and validates RandomForest classifiers with time-series cross-validation, performs randomized hyperparameter search, and reports test accuracy alongside feature importances.

Interactive Prediction CLI
Provides a command-line script under general game predictor/ that prompts for any two teams and uses the tuned model to predict head-to-head winners.

Exploratory Analysis Notebook
superbowl2025.ipynb walks through EDA, visualization of team trends, and initial model prototyping.

Built with pandas, scikit-learn, and matplotlib, NFL Predictor delivers end-to-end reproducibility—from raw data to actionable matchup predictions.

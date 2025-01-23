# IPL Match Outcome Prediction:

https://iplpredictionpd.streamlit.app/

This project focuses on predicting the outcome of Indian Premier League (IPL) cricket matches based on the current match scenario. 
Utilizing detailed historical data and machine learning techniques, specifically Logistic Regression and Random Forest classifiers, 
the project aims to provide accurate predictions that can be valuable for enthusiasts, analysts, and stakeholders.

## Table of Contents

- [Overview](#overview)
- [Datasets](#datasets)
- [Features](#features)
- [Machine Learning Models](#machine-learning-models)
- [Results](#results)
- [Dependencies](#dependencies)
- [Future Work](#future-work)

## Overview

Cricket, especially the T20 format like IPL, is a game of uncertainties. Predicting the outcome of a match involves analyzing 
various factors such as player performance, pitch conditions, and in-game dynamics. This project leverages historical IPL match 
data to train machine learning models that can predict the probable winner of a match based on the current state of the game.

## Datasets

The project utilizes two primary datasets:

1. **Matches Dataset (`matches.csv`)**: Contains detailed information about IPL matches, including teams, scores, outcomes, player of the match, venue, and more.

2. **Deliveries Dataset (`deliveries.csv`)**: Contains ball-by-ball data for each match, providing granular details like batsman, bowler, runs scored, extras, dismissals, etc.

*Both datasets should be placed in the same directory as the project notebook.*

## Features

Key features engineered and utilized in the modeling process include:

- **Current Score**: Total runs scored by the batting team at the current point in the match.
- **Overs Completed**: Number of overs bowled.
- **Wickets Fallen**: Number of wickets lost by the batting team.
- **Runs Remaining**: Runs needed by the batting team to win.
- **Balls Remaining**: Balls left in the innings.
- **Current Run Rate (CRR)**: Runs scored per over till the current point.
- **Required Run Rate (RRR)**: Runs required per over to win.
- **Team Encodings**: Representing teams in a numerical format suitable for modeling.

## Machine Learning Models

Two machine learning algorithms were employed:

1. **Logistic Regression**:
   - A statistical model that uses a logistic function to model a binary dependent variable.
   - Suitable for binary classification tasks like predicting win or loss.

2. **Random Forest Classifier**:
   - An ensemble learning method that constructs multiple decision trees during training and outputs the mode of the classes for classification tasks.
   - Handles overfitting better and captures complex patterns in the data.

Both models were trained and evaluated to compare performance, with hyperparameter tuning applied to optimize their predictive capabilities.

## Results

- **Accuracy**:
- Both models achieved commendable accuracy, with the Random Forest Classifier slightly outperforming Logistic Regression.

- **Confusion Matrix**:
- Showed the number of correct and incorrect predictions, indicating areas where the model performs well and where it struggles.

- **ROC Curve and AUC**:
- The Random Forest Classifier exhibited a higher Area Under the Curve (AUC), suggesting better discrimination between the classes.

*Detailed results, including graphs and evaluation metrics, are available in the project notebook.*

## Dependencies

Ensure the following Python libraries are installed:

- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `jupyter`

You can install these dependencies using `pip`.


## Future Work:


### Incorporate Real-Time Data:
Integrate live match data to provide real-time predictions.

### Expand Feature Set:
Include more features like player form, pitch conditions, and weather data.

### Advanced Modeling Techniques:
Explore deep learning models and ensemble methods to enhance prediction accuracy.

### User Interface:
Develop a web or mobile application to make the prediction model accessible to a broader audience.

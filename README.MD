# UDACITY Data Scientist Nanodegree

## Capstone Project - NHL Game Prediction


## Installations

This project requires Python 3.x and the following additional python libraries:

```python
import pandas as pd
import numpy as np
from functools import reduce
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.datasets import make_classification
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
import seaborn as sns
%matplotlib inline
import matplotlib.pyplot as plt
import warnings
```

You will also require software that can run and execute an iPython Notebook


## Project Motivation

The aim of this project is to create a model that can predict NHL Hickey games.

The process used is to clean and merge three csv files containing team, goalie and skater statistics related to games and then use a machine learning algorithm to train a classification model.  Multiple classifiers are used in order to see which performs best.
This model will then classify the results as a win or a loss based on the features provided. 

## File Descriptions

The files related to this project are as follows:

1. This README.md file, which contains information relating to the project aims and system requirements.
2. NHL_predictions.ipynb - Jupyter Notebook containing all python code required for the project
3. game_team_stats.csv - file containing team statistics
4. game_goalie_stats.csv - file containing goalie statistics
5. game_skater_stats.csv - file containing skater statistics

In addition there is a Medium blog post that presents the findings of this project.  The post can be found [here](https://www.kaggle.com/datasets/martinellis/nhl-game-data).

## How to interact with the project

The code used to run the entire project is held in NHL_predictions.ipynb.  This workbook should be opened and all code should be run.  Commentary and in-code comments are included in the notebook where appropriate.

## Results of Analysis

The detailed results analysis is included in the python notebook along with the code, and is also explained fully in the Medium blog post.  A brief summary for each question is noted below.

1. Using 4 different machine learning classifiers, we identified a model that used the Random Forest classifier that gave the best results, with f1_score, precision and recall values of 0.91, 0.92 and 0.89 respectively.
2. By using a Recursive Feature Eliminator  we refined the model to use less features, and upon testing this we identified a beetter model using the smaller feature set and the Artificial Neural Network classifier.  This produced f1_score, precision and recall values of 0.94, 0.92 and 0.96 respectively.

## Licensing, Authors, Acknowledgements, etc

The dataset used in this analysis is provided by Martin Ellis and is hosted on the Kaggle website [here](https://medium.com/@kane_62029/diamonds-in-the-rough-or-how-i-learned-to-stop-worrying-and-embrace-data-science-2f1a03383eaf).
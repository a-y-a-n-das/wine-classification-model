# Wine Classification Model

This project involves training a classification model to predict the type of wine based on several chemical properties. The dataset contains various features related to the chemical composition of wines, such as alcohol content, ash content, and phenols, which are used to predict the class of the wine.

## Model Description

The classification model used in this project is the **K-Nearest Neighbors (KNN)** algorithm, implemented using **scikit-learn**. KNN is a simple, yet effective, machine learning algorithm that classifies data points based on the majority class among their nearest neighbors. The model was trained using the processed wine dataset, and hyperparameters such as the number of neighbors (`k`) were optimized for better performance.


## Dataset

The dataset used in this project is from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/109/wine). The dataset contains **178 rows** and **13 attributes** including the wine’s chemical composition and its corresponding class.

**Features (Attributes):**
- Alcohol
- Ash
- Alcalinity_of_ash
- Magnesium
- Total_phenols
- Flavanoids
- Nonflavanoid_phenols
- Proanthocyanins
- Color_intensity
- Hue
- Diluted_wines
- Proline

**Class**: The target variable, which represents the wine type. There are three possible classes:
- Class 1
- Class 2
- Class 3

You can access the dataset [here](https://archive.ics.uci.edu/dataset/109/wine).

## Project Overview

This project aimed to build a machine-learning model that can classify wines into three classes based on their chemical properties.

### Preprocessing and Feature Selection

Initially, we had the following features in the dataset:
- Alcohol
- Ash
- Color_intensity
- Flavanoids
- Hue
- Malicacid
- Proanthocyanins
- Class
- Diluted_wines

However, after performing exploratory data analysis (EDA), it was observed that many of these features had overlapping distributions across classes (as seen in the histograms). As a result, we decided to drop the following features to avoid redundancy and improve the model’s performance:

- **Alcohol**
- **Ash**
- **Color_intensity**
- **Flavanoids**
- **Hue**
- **Malicacid**
- **Proanthocyanins**
- **Class**
- **Diluted_wines**

### Model Evaluation

After preprocessing, we trained a classification model and evaluated its performance using precision, recall, and F1-score metrics. The results were as follows:

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 1     | 0.90      | 1.00   | 0.95     | 9       |
| 2     | 1.00      | 0.94   | 0.97     | 18      |
| 3     | 1.00      | 1.00   | 1.00     | 17      |

- **Accuracy**: 98%
- **Macro Average**: Precision: 0.97, Recall: 0.98, F1-Score: 0.97
- **Weighted Average**: Precision: 0.98, Recall: 0.98, F1-Score: 0.98

The model performs exceptionally well, achieving an accuracy of **98%**.

## Instructions

### Dependencies

- Python 3.x
- Pandas
- Scikit-learn
- Matplotlib
- NumPy

## Conclusion

The project demonstrates the effectiveness of using chemical properties of wine to predict the wine class. After preprocessing the dataset and removing redundant features, the model achieved an accuracy of **98%**. The model's performance was evaluated using various metrics, with excellent results in precision, recall, and F1-score across all classes. This indicates that the classifier is highly capable of predicting the wine class based on the selected features.

Future work could explore different feature engineering techniques, alternative classifiers, or hyperparameter tuning to further improve the model's performance.

## Acknowledgements

Dataset source: [Wine Dataset - UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/109/wine)

Special thanks to the creators and maintainers of the UCI Machine Learning Repository for providing access to this valuable dataset.


# Wine Classification Model

This project involves training a classification model to predict the type of wine based on several chemical properties. The dataset contains various features related to the chemical composition of wines, such as alcohol content, ash content, and phenols, which are used to predict the class of the wine.

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
- imbalanced-learn (for SMOTE and handling class imbalance)

You can install the necessary dependencies by running:

```bash
pip install -r requirements.txt

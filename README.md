# CLASSIFY THE NEWSPAPER TITLE BY ESSEMBLE LEARNING TECHNIQUE
This Project contains the code, data, and analysis developed to classify the subtitle of newspaper. Our goal is to separate the short sentences into the individual group based on their contents.

Text Classification is one of the fundamental and important problems in the field of Natural Language Processing (NLP). The goal of this problem is to classify text segments into different groups or labels based on their content. Common applications of Text Classification include email classification (spam vs. spam), news classification, sentiment analysis, and many others.

## Overview
In this project, we will build a Text Classification program that involves classifying an abstract of a publication (scientific article) into different topics.
The program will be built on the Python platform and use popular machine learning libraries such as scikit-learn, numpy, etc.
Accordingly, the general Input/Output of the program will include:
* **Input**: An abstract of the publication (scientific article).
* **Output**: Topic of that abstract (e.g. physics, mathematics, computer science, etc.).

* **Data Exploration & Preprocessing**:
Analyzing hourly load data alongside temperature and GHI from multiple sites. We employ harmonic transformations to capture cyclic patterns and aggregate weather data to account for regional effects.

* **Model Selection & Training**:
Implementing `Text Embedding, Random Forest, Gradient Boost, XGBoost` that are fine-tuned over key hyperparameters.

* **Model Evaluation & Forecast Generation**:
Evaluating performance using metrics such as MAE, MSE, and MAPE. The final model produces reliable forecasts.

## Repository Structure

```
.
├── LICENSE (not applicable)
├── README.md
├── datasets
│   ├── full_dataset (>2GB)
│   ├── dataset2k.csv
│   └── train_data.csv
|   |__ test_data.csv
├── main.py (might not be used)
├── notebooks
│   ├── feature_extraction.ipynb
|   ├── feature_exploration.ipynb
│   └── xgboost.ipynb
|   |__ randomforest.ipynb
|   |__ gradientboost.ipynb
├── requirements.txt
└── utils
    ├── visualization.py
    |__ EmbeddingVectorization.py
```

## Installation
1. Clone the Repository:

```
git clone https://github.com/isenginer/AIVN_Project4.1_EnsembleLearning.git
```

2. Install Dependencies (Optional):

This section is for the library reference only. If your IDE has sufficient library, this section is not required to apply.
```
pip install -r requirements.txt
```

## Usage
### Interactive Analysis
Open the Jupyter notebooks in the notebooks directory to explore the data, visualize key insights, and review the model development process:

```
jupyter ./notebooks/interactive_eda.ipynb
```

### Running the Scripts
Use the following command to run:

```
python main.py
```

### Methodology
Our approach can be summarized as follows:
* **Data Preprocessing**:
We standardize features and create new ones to capture temporal patterns (e.g., sine/cosine transformations of the hour) and aggregate multi-site weather data.

* **Modeling**:
The model algorithm is employed due to its ability to model non-linear relationships and its built-in regularization. Model hyperparameters (number of trees, tree depth, learning rate, etc.) are fine-tuned using a time-series aware cross-validation strategy.

* **Validation**:
Model performance is evaluated using metrics such as MAE, MSE, and MAPE, ensuring the forecasts are both accurate and robust.

## License
The project is public and no license required.
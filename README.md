# End-to-End Machine Learning Prediction App with Flask

This project is an end-to-end machine learning application built using **Flask**. It allows users to input data through a web interface and get predictions about student performance based on a trained machine learning model.

## Features

- Web interface for data input
- Data preprocessing using `pandas` and `numpy`
- Machine learning model prediction using `scikit-learn`
- Flask web framework for serving the application

## How to Run

1. Ensure you have Python and Flask installed on your system.
2. Navigate to the directory containing the `app.py` file.
3. Run the Flask application using the command:
    ```bash
    python app.py
    ```
4. Open your web browser and go to `http://127.0.0.1:5000/` to view the home page of your application.
5. To input data and get predictions, navigate to `http://127.0.0.1:5000/predictdata`.

## HTML Pages

- `index.html`: A simple welcome page.
- `home.html`: A page where users can input their data.

## Notebook Folder

The `notebook` folder contains Jupyter notebooks that were used during the development and experimentation phase of the project. These notebooks provide a detailed view of the steps taken to preprocess the data, train the models, and evaluate their performance. These notebooks include:

- `1. EDA STUDENT PERFORMANCE.ipynb`: This notebook contains exploratory data analysis (EDA) on the student performance dataset. It includes data visualization and statistical analysis to understand the distribution and relationships between different features.
- `2. MODEL TRAINING.ipynb`: This notebook contains code for training and evaluating different machine learning models. It includes data preprocessing, model training, and performance evaluation using various metrics.

### Business Problem

The business problem addressed in the `2. MODEL TRAINING.ipynb` notebook is predicting students' performance based on various features such as gender, race/ethnicity, parental level of education, lunch, test preparation course, and scores in math, reading, and writing. The goal is to build a machine learning model that can accurately predict students' math scores based on these features. This can help educators identify students who may need additional support and resources to improve their performance.

### Steps in the Notebook

1. **Import Data and Required Packages**: Importing necessary libraries such as `pandas`, `numpy`, `matplotlib`, `seaborn`, and machine learning libraries like `scikit-learn`, `catboost`, and `xgboost`.
2. **Data Preprocessing**: Loading the dataset, handling missing values, encoding categorical variables, and scaling numerical features.
3. **Model Training**: Training various machine learning models, including Linear Regression, Lasso, Ridge, K-Neighbors Regressor, Decision Tree, Random Forest Regressor, XGBRegressor, CatBoosting Regressor, and AdaBoost Regressor.
4. **Model Evaluation**: Evaluating the performance of the trained models using metrics such as Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R2 Score.
5. **Results Visualization**: Visualizing the results using scatter plots and regression plots to compare the actual and predicted values.

## src Folder

The `src` folder contains the source code for the machine learning pipeline and other utility functions used in the project. Here is a brief description of the key files and their purposes:

- `pipeline/`: This directory contains the pipeline code for data preprocessing and prediction.
  - `predict_pipeline.py`: Contains the `CustomData` and `PredictPipeline` classes for data handling and prediction.
  - `train_pipeline.py`: Contains the code for training the machine learning model, including data preprocessing, model training, and saving the trained model.
- `components/`: This directory contains the components for data ingestion, data transformation, and model training.
  - `data_ingestion.py`: Contains the `DataIngestion` class for reading the dataset, splitting it into training and testing sets, and saving these sets as CSV files.
  - `data_transformation.py`: Contains the `DataTransformation` class for transforming the data, including handling missing values and scaling features.
  - `model_trainer.py`: Contains the `ModelTrainer` class for training and evaluating different machine learning models and saving the best model.
- `utils.py`: This file contains utility functions used throughout the project. It includes functions for saving and loading objects, as well as evaluating machine learning models.

These files and directories provide the necessary functionality for data preprocessing, model training, and prediction in the project.
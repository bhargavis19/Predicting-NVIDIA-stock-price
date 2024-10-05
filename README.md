# Predicting-NVIDIA-stock-price

This project aims to predict stock price movements using a machine learning model, specifically a `RandomForestClassifier`. The model is trained and evaluated on historical stock market data to predict whether the stock price will rise or fall.

## Project Overview

The notebook performs the following steps:

1. **Data Preparation**: Prepares the dataset for training, including handling missing values and feature engineering.
2. **Model Training**: Trains a `RandomForestClassifier` using historical stock data.
3. **Backtesting**: Evaluates the model on unseen data using a custom backtest function.
4. **Prediction**: Makes predictions and evaluates the precision of the model.

## Requirements

To run the notebook, you will need the following Python libraries:

- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `seaborn`

You can install these dependencies using pip:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

## Notebook Contents

- **Feature Engineering**: The notebook defines key features based on historical stock data to predict future movements.
- **Random Forest Model**: The `RandomForestClassifier` is used for classification tasks, and it is trained using a set of selected predictors.
- **Backtesting Function**: The `backtest` function tests the model's performance on historical data and compares the predicted outcomes with actual market movements.
- **Evaluation**: The model is evaluated based on precision and other metrics.

## Key Functions

- `predict()`: Trains the model on training data and generates predictions for the test data.
- `backtest()`: Simulates predictions on unseen data, allowing you to test the model's performance before deploying it.

## Usage

1. **Data**: Load your historical stock price data (preferably for NVIDIA) in CSV format or any other suitable format.
2. **Training**: Run the notebook to train the model using the provided features.
3. **Prediction**: Use the trained model to predict future stock movements.

## Results

The notebook outputs the following results:

- Confusion matrix visualizations to assess prediction performance.
- Precision score of the model.
- Count of correct and incorrect predictions.

## Conclusion

This project demonstrates a simple yet effective approach to predicting stock prices using machine learning. The `RandomForestClassifier` offers robust predictions based on historical market data, and the notebook's backtest function ensures that the model is evaluated thoroughly before being applied to future predictions.

---

**Note**: This project is for educational purposes only and should not be used for actual financial trading without further validation.

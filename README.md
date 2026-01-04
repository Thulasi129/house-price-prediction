# House Price Prediction

This project predicts house prices based on various features using a machine learning model. It includes a web application to predict house prices and a notebook with the exploratory data analysis and model training process.

## Project Structure

```
.
├── data
│   └── house_prices.csv
├── notebooks
│   └── House_Price_Prediction.ipynb
├── results
│   ├── actual_vs_predicted.png
│   ├── correlation_matrix.png
│   ├── distributions.png
│   ├── feature_importance.csv
│   ├── feature_importance.png
│   ├── model_columns.joblib
│   ├── model_performance.csv
│   ├── model_rmse_comparison.png
│   ├── model.joblib
│   ├── pairplot.png
│   ├── price_by_location.png
│   ├── scaler.joblib
│   └── summary_stats.txt
├── src
│   ├── app.py
│   ├── generate_data.py
│   └── run_prediction.py
├── README.md
└── requirements.txt
```

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/Thulasi129/house-price-prediction.git
    cd house-price-prediction
    ```

2.  Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To run the web application:

```bash
python src/app.py
```

Open your browser and go to `http://127.0.0.1:5000` to predict house prices.

## Notebook

The `notebooks/House_Price_Prediction.ipynb` notebook contains the exploratory data analysis, feature engineering, model training, and evaluation.

## Results

The `results` folder contains the outputs of the model training and analysis, such as:
-   `model.joblib`: The trained model.
-   `scaler.joblib`: The scaler used for data preprocessing.
-   `model_performance.csv`: The performance metrics of the model.
-   Various plots and visualizations.
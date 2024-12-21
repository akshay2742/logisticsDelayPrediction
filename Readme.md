# Logistics Delay Prediction Documentation

## Overview

This project aims to predict whether a shipment will be delayed or on time based on historical data. It utilizes machine learning algorithms to build a classification model and exposes an API for making predictions.

## Project Structure

```
logistics_delay_prediction/
│
├── src/
│   ├── data_preparation.py        # Data cleaning and preparation
│   ├── model.py                   # Model training and evaluation
│   ├── train_and_predict.py       # Training and prediction logic
│   └── api.py                     # FastAPI for serving predictions
│
├── notebooks/
│   └── exploratory_analysis.ipynb  # Jupyter notebook for EDA
│
├── data/
│   └── shipment_data.xlsx         # Dataset for training and testing
│
├── requirements.txt                # Required Python packages
└── Problem_Statement1.txt          # Problem statement and tasks
```

## Setup Instructions

### Prerequisites

- Python 3.6 or higher

- pip (Python package installer)

### Installation

1. **Clone the repository**:

   - git clone https://github.com/akshay2742/logisticsDelayPrediction.git
   - cd logistics_delay_prediction

2. **Install required packages**:

   - pip install -r requirements.txt

3. **Download the dataset**:

   - Ensure that the dataset `shipment_data.xlsx` is placed in the `data/` directory.

## Data Preparation & Exploration

- The data preparation is handled in `data_preparation.py`, where the dataset is cleaned, missing values are handled, and categorical variables are encoded.

- Exploratory Data Analysis (EDA) is performed in the Jupyter notebook `exploratory_analysis.ipynb`, which provides insights into the dataset and visualizations.

## Model Development

- The model is built in `model.py`, where different machine learning algorithms (e.g., Random Forest, Logistic Regression) are implemented.

- The model is trained and evaluated using metrics such as accuracy, precision, recall, and F1 score.

## API Usage

### Starting the API

To start the FastAPI server, run the following command:
python src/api.py
The server will start at http://127.0.0.1:8000.

### Making Predictions

You can make predictions by sending a POST request to the `/predict` endpoint with the shipment details in JSON format. Here’s an example using `curl`:

```json
{
  "origin": "Mumbai",
  "destination": "Delhi",
  "vehicle_type": "Truck",
  "distance": 1400.0,
  "weather": "Clear",
  "traffic": "Moderate"
}
```

```bash
curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d '{
  "origin": "Mumbai",
  "destination": "Delhi",
  "vehicle_type": "Truck",
  "distance": 1400.0,
  "weather": "Clear",
  "traffic": "Moderate"
}'
```

### Request Body

The request should contain the following fields:

- `origin`: Origin city (string)

- `destination`: Destination city (string)

- `vehicle_type`: Type of vehicle (string)

- `distance`: Distance in kilometers (float)

- `weather`: Weather conditions (string)

- `traffic`: Traffic conditions (string)

### Response

The API will return a JSON response with the prediction and probability:

```
{
    "delay_predicted": "Yes",
    "probability": 0.85
}
```

## Conclusion

This project demonstrates the ability to handle data, build a classification model, and deploy it as an API. The documentation provides a clear guide for setting up and using the system effectively.

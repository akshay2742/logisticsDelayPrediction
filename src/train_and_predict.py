from data_preparation import DataPreparation
from model import DelayPredictor
import joblib
import os

def train_model():
    # Initialize data preparation
    data_prep = DataPreparation()
    
    # Load and prepare data
    df = data_prep.load_and_clean_data("data/shipment_data.xlsx")
    X_train, X_test, y_train, y_test = data_prep.prepare_features(df)
    
    # Train models
    predictor = DelayPredictor()
    results = predictor.train_and_evaluate(X_train, X_test, y_train, y_test)
    
    # Print results
    print("\nModel Performance:")
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
    
    # Save the best model and data preparation object
    os.makedirs("models", exist_ok=True)
    predictor.save_model("models/best_model.pkl")
    joblib.dump(data_prep, "models/data_prep.pkl")
    
    print(f"\nBest model ({predictor.best_model_name}) has been saved!")
    return predictor, data_prep

def predict_shipment(predictor, data_prep, shipment_details):
    """
    Make prediction for a single shipment
    """
    # Transform input data using the saved label encoders
    transformed_data = {
        'Origin': data_prep.le_dict['Origin'].transform([shipment_details['origin']])[0],
        'Destination': data_prep.le_dict['Destination'].transform([shipment_details['destination']])[0],
        'Vehicle_Type': data_prep.le_dict['Vehicle_Type'].transform([shipment_details['vehicle_type']])[0],
        'Distance': shipment_details['distance'],
        'Weather_Conditions': data_prep.le_dict['Weather_Conditions'].transform([shipment_details['weather']])[0],
        'Traffic_Conditions': data_prep.le_dict['Traffic_Conditions'].transform([shipment_details['traffic']])[0]
    }
    
    # Create feature array
    features = [[
        transformed_data['Origin'],
        transformed_data['Destination'],
        transformed_data['Vehicle_Type'],
        transformed_data['Distance'],
        transformed_data['Weather_Conditions'],
        transformed_data['Traffic_Conditions']
    ]]
    
    # Make prediction
    prediction = predictor.best_model.predict(features)[0]
    probability = predictor.best_model.predict_proba(features)[0][1]
    
    return "Delayed" if prediction == 1 else "On Time", probability

if __name__ == "__main__":
    # Train the model
    predictor, data_prep = train_model()
    
    # Example prediction
    sample_shipment = {
        'origin': 'Mumbai',
        'destination': 'Delhi',
        'vehicle_type': 'Truck',
        'distance': 1400.0,
        'weather': 'Clear',
        'traffic': 'Moderate'
    }
    
    prediction, probability = predict_shipment(predictor, data_prep, sample_shipment)
    print(f"\nPrediction for sample shipment:")
    print(f"Status: {prediction}")
    print(f"Delay Probability: {probability:.2f}")
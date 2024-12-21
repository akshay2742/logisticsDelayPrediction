from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

class DelayPredictor:
    def __init__(self):
        self.models = {
            'random_forest': RandomForestClassifier(random_state=42),
            'logistic_regression': LogisticRegression(random_state=42),
            'decision_tree': DecisionTreeClassifier(random_state=42)
        }
        self.best_model = None
        self.best_model_name = None
        
    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        results = {}
        
        for name, model in self.models.items():
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            results[name] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred)
            }
        
        # Select best model based on F1 score
        self.best_model_name = max(results, key=lambda k: results[k]['f1'])
        self.best_model = self.models[self.best_model_name]
        
        return results
    
    def save_model(self, filepath):
        joblib.dump(self.best_model, filepath)
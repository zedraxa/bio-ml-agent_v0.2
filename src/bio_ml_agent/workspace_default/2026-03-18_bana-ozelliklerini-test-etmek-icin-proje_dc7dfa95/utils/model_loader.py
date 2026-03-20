import joblib
import pandas as pd
import numpy as np
import os

def load_and_predict(model_path, X_new):
    """
    Loads a saved model and makes predictions on new data.

    Args:
        model_path (str): Path to the saved model (.pkl file).
        X_new (pd.DataFrame or np.array): New data for prediction.

    Returns:
        np.array: Predicted labels or values.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    
    print(f"Loading model from: {model_path}")
    model = joblib.load(model_path)
    
    print("Making predictions...")
    predictions = model.predict(X_new)
    
    return predictions

def load_model(model_path):
    """
    Loads a saved model from a specified path.

    Args:
        model_path (str): Path to the saved model (.pkl file).

    Returns:
        object: The loaded model object.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    
    print(f"Loading model from: {model_path}")
    model = joblib.load(model_path)
    return model

# Example Usage (not to be run directly by the agent, just for reference):
if __name__ == '__main__':
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    # Create a dummy model and save it
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])
    y_series = pd.Series(y, name='target')

    X_train, X_test, y_train, y_test = train_test_split(X_df, y_series, test_size=0.3, random_state=42)

    dummy_model_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(random_state=42))
    ])
    dummy_model_pipeline.fit(X_train, y_train)

    # Save the dummy model
    if not os.path.exists('results'):
        os.makedirs('results')
    joblib.dump(dummy_model_pipeline, 'results/dummy_model.pkl')
    print("Dummy model saved to results/dummy_model.pkl")

    # Simulate new data
    new_data = pd.DataFrame(np.random.rand(5, 5), columns=[f'feature_{i}' for i in range(5)])

    # Load and predict
    try:
        predictions = load_and_predict('results/dummy_model.pkl', new_data)
        print("\nPredictions on new data:\n", predictions)
    except FileNotFoundError as e:
        print(e)
    
    # Clean up
    os.remove('results/dummy_model.pkl')
    os.rmdir('results')
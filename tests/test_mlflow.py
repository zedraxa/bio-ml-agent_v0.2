import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from bio_ml_agent.ml.model_compare import ModelComparator
from bio_ml_agent.mlflow_tracker import get_shared_tracker

def main():
    print("Testing MLflow Tracker...")
    X, y = make_classification(n_samples=100, n_features=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    comp = ModelComparator(task_type="classification", cv_folds=2)
    # We will test just one or two models to be fast
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    comp.models = {
        "LR": Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression())])
    }
    
    comp.run(X_train, X_test, y_train, y_test)
    comp.plot_comparison("test_results")
    
    print("Test run finished.")

if __name__ == "__main__":
    main()

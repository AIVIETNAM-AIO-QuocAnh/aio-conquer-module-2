from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def get_model(model_key: str, random_state: int = 42):
    models_map = {
        "knn": KNeighborsClassifier(n_jobs=-1),
        "logistic_regression": LogisticRegression(
            random_state=random_state, 
            n_jobs=-1
        ),
        "random_forest": RandomForestClassifier(
            random_state=random_state, 
            n_jobs=-1
        )
    }

    if model_key not in models_map:
        raise ValueError(f"Invalid model '{model_key}'. Choose one of: {list(models_map.keys())}")

    return models_map[model_key]
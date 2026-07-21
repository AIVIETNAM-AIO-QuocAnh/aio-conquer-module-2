from typing import Dict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from src.models import get_model

def build_pipeline(
    model_key: str, 
    use_scaler: bool = False, 
    use_pca: bool = False, 
    n_components: float = 0.95, 
    random_state: int = 42
) -> Pipeline:
    
    """
    Constructs a unified pipeline based on configuration flags.
    
    Parameters:
    -----------
    model_key : str - Model identifier ('knn', 'logistic_regression', 'random_forest').
    use_scaler : bool, default=False - Whether to apply StandardScaler.
    use_pca : bool, default=False - Whether to apply PCA for dimensionality reduction.
    n_components : float, default=0.95 - Variance ratio to retain when using PCA.
    random_state : int, default=42 - Seed to ensure reproducibility for PCA and the model.
        
    Returns:
    --------
    sklearn.pipeline.Pipeline
        A complete processing pipeline comprising preprocessing and the model.
    """

    steps = []

    # Feature Scaling
    if use_scaler:
        steps.append(('scaler', StandardScaler()))

    # PCA Dimensionality Reduction
    if use_pca:
        steps.append(('pca', PCA(n_components=n_components, random_state=random_state)))

    # Classifier Model
    model = get_model(model_key, random_state=random_state)
    steps.append(('model', model))

    return Pipeline(steps)


def build_pipeline_from_setting(
    model_key: str, 
    setting_id: str, 
    n_components: float = 0.95, 
    random_state: int = 42
) -> Pipeline:
    
    """
    Helper function to initialize a pipeline directly from an experimental setting ID.
    
    Supported setting IDs:
    - 'no_scaler_no_pca' : Case 1 (No Scaler & No PCA)
    - 'no_scaler_pca'    : Case 2 (No Scaler & PCA)
    - 'scaler_no_pca'    : Case 3 (Scaler & No PCA)
    """

    settings_map: Dict[str, Dict[str, bool]] = {
        "no_scaler_no_pca": {"use_scaler": False, "use_pca": False},
        "no_scaler_pca":    {"use_scaler": False, "use_pca": True},
        "scaler_no_pca":    {"use_scaler": True,  "use_pca": False},
    }

    if setting_id not in settings_map:
        valid_keys = list(settings_map.keys())
        raise ValueError(
            f"Invalid setting ID '{setting_id}'. "
            f"Choose one of: {valid_keys}"
        )

    config = settings_map[setting_id]
    
    return build_pipeline(
        model_key=model_key,
        use_scaler=config["use_scaler"],
        use_pca=config["use_pca"],
        n_components=n_components,
        random_state=random_state
    )
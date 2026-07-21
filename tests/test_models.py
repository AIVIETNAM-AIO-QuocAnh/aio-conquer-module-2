import numpy as np
from sklearn.model_selection import GridSearchCV
from src.config import load_config
from src.pipelines.builder import build_pipeline_from_setting
from src.data.blocks import load_block

def run_smoke_test():
    print("=== Start Smoke Test for Models & Pipelines ===")
    config = load_config()
    
    # Load Block 0
    print("Loading Block 0 data...")
    X_train, y_train, X_test, y_test = load_block(0, config)
    
    # Subsampling
    X_train_sub, y_train_sub = X_train[:300], y_train[:300]
    X_test_sub, y_test_sub = X_test[:100], y_test[:100]

    model_keys = ["knn", "logistic_regression", "random_forest"]
    settings = ["no_scaler_no_pca", "no_scaler_pca", "scaler_no_pca"]

    for model_key in model_keys:
        print(f"\n----------------------------------------")
        print(f"Testing Model: {model_key.upper()}")
        
        # Get param_grid from yaml
        param_grid = config["models"][model_key]["param_grid"]
        
        for setting in settings:
            print(f"  -> Testing Pipeline Setting: {setting}")
            
            # Build pipeline base on settings
            pipeline = build_pipeline_from_setting(
                model_key=model_key,
                setting_id=setting,
                n_components=0.90,
                random_state=42
            )
            
            search = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grid,
                cv=2,
                scoring='accuracy',
                n_jobs=-1
            )
            
            # Training
            search.fit(X_train_sub, y_train_sub)
            acc = search.score(X_test_sub, y_test_sub)
            
            print(f"     [OK] Best Params: {search.best_params_}")
            print(f"     [OK] Test Acc: {acc:.4f}")

    print("\n=== SUCCESS: All 3 models passed smoke test across all 3 pipeline settings! ===")

if __name__ == "__main__":
    run_smoke_test()

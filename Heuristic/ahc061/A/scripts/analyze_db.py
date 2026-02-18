import optuna

DB_PATH = "sqlite:///optuna_ahc.db"
STUDY_NAME = "ahc_tuning_27p_v2"

def analyze():
    print(f"Loading study: {STUDY_NAME}")
    study = optuna.load_study(study_name=STUDY_NAME, storage=DB_PATH)
    
    print(f"Best value: {study.best_value}")

    print("\nCalculating Parameter Importance...")
    try:
        importances = optuna.importance.get_param_importances(study)
        print("\nParameter Importance:")
        for k, v in importances.items():
            print(f"  {k}: {v:.4f}")
    except Exception as e:
        print(f"Could not calculate importance: {e}")

if __name__ == "__main__":
    analyze()

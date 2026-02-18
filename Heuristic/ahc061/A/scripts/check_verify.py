import optuna
import sys

DB_PATH = "sqlite:///optuna_ahc.db"
STUDY_NAME = "verify_tuning"

def check():
    try:
        study = optuna.load_study(study_name=STUDY_NAME, storage=DB_PATH)
        print(f"Study: {STUDY_NAME}")
        print(f"Number of trials: {len(study.trials)}")
        for t in study.trials:
            print(f"Trial {t.number}: state={t.state}, value={t.value}")
    except Exception as e:
        print(f"Error loading study: {e}")

if __name__ == "__main__":
    check()

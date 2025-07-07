from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import config

def get_classical_models():
    """Retorna um dicionário de modelos clássicos não instanciados."""
    models = {
        'RandomForest': RandomForestClassifier(random_state=config.RANDOM_STATE, n_jobs=1),
        'XGBoost': XGBClassifier(random_state=config.RANDOM_STATE, eval_metric='mlogloss', n_jobs=1)
    }
    return models
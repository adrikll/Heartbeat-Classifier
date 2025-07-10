import os
import json
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from scikeras.wrappers import KerasClassifier
import config
from dataloader import load_and_prepare_data
from models.neural_networks import create_mlp, create_cnn
from sklearn.utils import class_weight
import numpy as np

def optimize():
    """Executa o Random Search para encontrar os melhores hiperparâmetros e os salva em um arquivo JSON."""
    
    #carrega os dados de treino
    (X_train, y_train), _, _ = load_and_prepare_data()

    weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights_dict = dict(enumerate(weights))
    print(f"Pesos de classe calculados: {class_weights_dict}")

    models_and_params = {
        'RandomForest': {
            'model': RandomForestClassifier(random_state=config.RANDOM_STATE, class_weight='balanced', n_jobs= 1),
            'params': {'n_estimators': [100, 200, 300], 'max_depth': [10, 20, 30, None], 'min_samples_split': [2, 5, 10]}
        },
        'XGBoost': {
            'model': XGBClassifier(random_state=config.RANDOM_STATE, eval_metric='mlogloss', n_jobs= 1),
            'params': {'n_estimators': [100, 200 ,300], 'learning_rate': [0.01, 0.1, 0.2], 'max_depth': [3, 5, 7]}
        },
        'MLP': {
            'model': KerasClassifier(model=create_mlp, verbose=0, class_weight=class_weights_dict, loss="sparse_categorical_crossentropy"),
            'params': {'batch_size': [64, 128, 256], 'epochs': [10, 20, 30], 'optimizer': ['adam', 'rmsprop']}
        },
        'CNN': {
             'model': KerasClassifier(model=create_cnn, verbose=0, class_weight=class_weights_dict, loss="sparse_categorical_crossentropy"),
            'params': {'batch_size': [64, 128, 256], 'epochs': [10, 15, 20], 'optimizer': ['adam', 'rmsprop']}
        }
    }
    
    best_params_all_models = {}

    for model_name, mp in models_and_params.items():
        print(f"\n--- Otimizando {model_name} ---")
        search = RandomizedSearchCV(estimator=mp['model'], param_distributions=mp['params'], n_iter=config.N_ITER_SEARCH, cv=config.CV_FOLDS, verbose=2, random_state=config.RANDOM_STATE, n_jobs=1)
        search.fit(X_train, y_train)
        best_params_all_models[model_name] = search.best_params_
        print(f"Melhores parâmetros para {model_name}: {search.best_params_}")

    print("\nOtimização concluída. Salvando melhores parâmetros...")
    if not os.path.exists(config.OUTPUT_DIR):
        os.makedirs(config.OUTPUT_DIR)

    with open(config.BEST_PARAMS_FILE, 'w') as f:
        json.dump(best_params_all_models, f, indent=4)
    
    print(f"Melhores parâmetros salvos em: {config.BEST_PARAMS_FILE}")

if __name__ == '__main__':
    optimize()
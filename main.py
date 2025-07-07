import os
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from scikeras.wrappers import KerasClassifier
from sklearn.utils import class_weight  

import config
from dataloader import load_and_prepare_data
from models.neural_networks import create_mlp, create_cnn
from utils import evaluate_and_compare_models
import warnings

warnings.filterwarnings('ignore', category=UserWarning, module='xgboost')

def run_pipeline():
    """Orquestra o pipeline completo de treinamento e avaliação usando class_weight."""
    
    if not os.path.exists(config.OUTPUT_DIR):
        os.makedirs(config.OUTPUT_DIR)
        
    # 1. Carregar Dados (agora sem SMOTE)
    print("[ETAPA 1/3] Carregando e preparando dados...")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_and_prepare_data()

    # 2. Calcular Pesos das Classes
    # Esta etapa calcula os pesos para penalizar mais os erros nas classes minoritárias.
    print("\nCalculando os pesos das classes para o treinamento...")
    weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights_dict = dict(enumerate(weights))
    print(f"Pesos de classe calculados: {class_weights_dict}")

    # 3. Definir Modelos e Espaço de Busca
    # ATUALIZAÇÃO: Adicionamos o 'class_weight' aos modelos que o suportam diretamente.
    models_and_params = {
        'RandomForest': {
            'model': RandomForestClassifier(
                random_state=config.RANDOM_STATE,
                class_weight='balanced',  # <-- MUDANÇA AQUI: Scikit-learn usa o modo 'balanced'
                n_jobs=-1
            ),
            'params': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10]
            }
        },
        'XGBoost': {
            # Nota: A ponderação de classes no XGBoost é mais complexa.
            # Para este exemplo, confiaremos na robustez do algoritmo.
            'model': XGBClassifier(random_state=config.RANDOM_STATE, use_label_encoder=False, eval_metric='mlogloss', n_jobs=-1),
            'params': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            }
        },
        'MLP': {
            'model': KerasClassifier(
                model=create_mlp,
                verbose=0,
                # <-- MUDANÇA AQUI: Passando os pesos para o Keras
                class_weight=class_weights_dict
            ),
            'params': {
                'batch_size': [64, 128, 256],
                'epochs': [10, 20, 30],
                'optimizer': ['adam', 'rmsprop']
            }
        },
        'CNN': {
             'model': KerasClassifier(
                model=create_cnn,
                verbose=0,
                # <-- MUDANÇA AQUI: Passando os pesos para o Keras
                class_weight=class_weights_dict
            ),
            'params': {
                'batch_size': [64, 128, 256],
                'epochs': [10, 15, 20],
                'optimizer': ['adam', 'rmsprop']
            }
        }
    }
    
    # 4. Executar o Random Search e Treinar
    print("\n[ETAPA 2/3] Otimizando hiperparâmetros com Random Search...")
    results = {}
    
    for model_name, mp in models_and_params.items():
        print(f"\n--- Otimizando {model_name} ---")
        
        search = RandomizedSearchCV(
            estimator=mp['model'],
            param_distributions=mp['params'],
            n_iter=config.N_ITER_SEARCH,
            cv=config.CV_FOLDS,
            verbose=2,
            random_state=config.RANDOM_STATE,
            n_jobs=1 # Mantendo n_jobs=1 para evitar erros de memória
        )
        
        # O fit agora é feito no conjunto de treino original (não balanceado)
        search.fit(X_train, y_train)
        
        results[model_name] = {
            'best_score': search.best_score_,
            'best_params': search.best_params_,
            'best_estimator': search.best_estimator_
        }
        print(f"Melhor score (validação cruzada) para {model_name}: {search.best_score_:.4f}")
        print(f"Melhores parâmetros: {search.best_params_}")

    # 5. Avaliar e Comparar os Melhores Modelos
    print("\n[ETAPA 3/3] Avaliando os melhores modelos no conjunto de teste...")
    evaluate_and_compare_models(results, X_test, y_test)
    
    print("\n--- Pipeline concluído! ---")

if __name__ == '__main__':
    run_pipeline()
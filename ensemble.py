import os
import time
import joblib
import json
import numpy as np
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.utils import to_categorical

import config
from dataloader import load_and_prepare_data
from utils import plot_confusion_matrix, plot_roc_curves, CLASSES

def train_final_ensemble():
    """
    Script final que carrega os melhores hiperparâmetros já otimizados
    e treina o modelo de Stacking.
    """
    MODEL_NAME = 'Stacking_Ensemble_Final'

    #Carregar Dados e Hiperparâmetros Otimizados
    print("\nCarregando dados e melhores hiperparâmetros...")

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_and_prepare_data()

    try:
        with open(config.BEST_PARAMS_FILE, 'r') as f:
            best_params = json.load(f)
        print("Hiperparâmetros otimizados carregados com sucesso.")
    except FileNotFoundError:
        print(f"ERRO: Arquivo '{config.BEST_PARAMS_FILE}' não encontrado.")
        print("Por favor, execute o script '1_optimize_hyperparameters.py' primeiro para gerar este arquivo.")
        return

    #Construção do Modelo Stacking com os Melhores Parâmetros
    print("\nConstruindo o modelo Stacking com os melhores estimadores...")

    params_rf = best_params.get('RandomForest')
    params_xgb = best_params.get('XGBoost')

    if not params_rf or not params_xgb:
        print("ERRO: Hiperparâmetros para RandomForest ou XGBoost não encontrados no arquivo JSON.")
        return

    base_estimators = [
        ('rf', RandomForestClassifier(random_state=config.RANDOM_STATE, class_weight='balanced', n_jobs=1, **params_rf)),
        ('xgb', XGBClassifier(random_state=config.RANDOM_STATE, eval_metric='mlogloss', n_jobs=1, **params_xgb))
    ]

    #definição do meta-modelo
    meta_model = LogisticRegression(max_iter=1000, n_jobs=1)

    #cria classificador Stacking
    stacking_model = StackingClassifier(
        estimators=base_estimators, final_estimator=meta_model, cv=5, n_jobs=1
    )
    print("Modelo Stacking construído.")

    #treinamento
    print("\nTreinando o modelo Stacking...")
    start_time = time.time()
    stacking_model.fit(X_train, y_train)
    end_time = time.time()
    print(f"Treinamento concluído em {(end_time - start_time):.2f} segundos.")

    #avaliação
    print("\nAvaliando o modelo...")
    model_output_dir = os.path.join(config.OUTPUT_DIR, MODEL_NAME)
    if not os.path.exists(model_output_dir):
        os.makedirs(model_output_dir)

    y_pred_probs = stacking_model.predict_proba(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    plot_confusion_matrix(y_test, y_pred, MODEL_NAME, model_output_dir)
    y_test_one_hot = to_categorical(y_test, num_classes=config.NUM_CLASSES)
    plot_roc_curves(y_test_one_hot, y_pred_probs, MODEL_NAME, model_output_dir)
    print(f"Gráficos de análise salvos na pasta: {model_output_dir}")

    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n--- Relatório Final para {MODEL_NAME} ---")
    print(f"Acurácia no Teste: {accuracy:.4f}")
    print(classification_report(y_test, y_pred, target_names=CLASSES))
    
    model_save_path = os.path.join(model_output_dir, "final_ensemble_model.joblib")
    joblib.dump(stacking_model, model_save_path)

if __name__ == '__main__':
    train_final_ensemble()
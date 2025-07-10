import os
import json
import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical
from sklearn.utils import class_weight
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import config
from dataloader import load_and_prepare_data
from models.neural_networks import create_mlp, create_cnn
from utils import plot_learning_curves, plot_confusion_matrix, plot_roc_curves, CLASSES

def train_and_evaluate():
    """Carrega os melhores hiperparâmetros, treina os modelos finais e gera uma análise detalhada."""
    print("Treinamento Final")

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_and_prepare_data()

    #melhores hiperparâmetros
    with open(config.BEST_PARAMS_FILE, 'r') as f:
        best_params = json.load(f)

    #balanceamento das classes usando class_weights
    weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights_dict = dict(enumerate(weights))
    
    final_results = []

    #treina e avalia cada modelo
    for model_name, params in best_params.items():
        print(f"\n--- Processando modelo: {model_name} ---")
        
        #cria um diretório de saída para este modelo
        model_output_dir = os.path.join(config.OUTPUT_DIR, model_name)
        if not os.path.exists(model_output_dir):
            os.makedirs(model_output_dir)

        #instancia o modelo com os melhores parâmetros
        if model_name == 'RandomForest':
            model = RandomForestClassifier(random_state=config.RANDOM_STATE, class_weight='balanced', n_jobs=1, **params)
            model.fit(X_train, y_train)
        
        elif model_name == 'XGBoost':
            model = XGBClassifier(random_state=config.RANDOM_STATE, eval_metric='mlogloss', n_jobs=1, **params)
            model.fit(X_train, y_train)
            
        elif model_name in ['MLP', 'CNN']:
            if model_name == 'MLP':
                model = create_mlp(optimizer=params.get('optimizer', 'adam'))
            else: #CNN
                model = create_cnn(optimizer=params.get('optimizer', 'adam'))
            
            history = model.fit(
                X_train, y_train,
                epochs=params.get('epochs', config.NN_EPOCHS),
                batch_size=params.get('batch_size', config.NN_BATCH_SIZE),
                validation_data=(X_val, y_val),
                class_weight=class_weights_dict,
                verbose=1
            )
            plot_learning_curves(history, model_name, model_output_dir)

        #avaliação no conjunto de teste
        print("Realizando predições no conjunto de teste...")
        if model_name in ['MLP', 'CNN']:
            y_pred_probs = model.predict(X_test)
        else:
            y_pred_probs = model.predict_proba(X_test)


        y_pred = np.argmax(y_pred_probs, axis=1)
        
        plot_confusion_matrix(y_test, y_pred, model_name, model_output_dir)
        y_test_one_hot = to_categorical(y_test, num_classes=config.NUM_CLASSES)
        plot_roc_curves(y_test_one_hot, y_pred_probs, model_name, model_output_dir)

        accuracy = accuracy_score(y_test, y_pred)
        print(f"Acurácia no Teste para {model_name}: {accuracy:.4f}")
        print(classification_report(y_test, y_pred, target_names=CLASSES))

        final_results.append({'Modelo': model_name, 'Acurácia Teste': accuracy})

    #salva o resumo
    summary_df = pd.DataFrame(final_results)
    print("\nResumo Comparativo -----------------------------------------------------")
    print(summary_df.to_string())
    summary_df.to_csv(os.path.join(config.OUTPUT_DIR, 'resumo_final.csv'), index=False)
    print(f"\nPipeline concluída!'{config.OUTPUT_DIR}'.")

if __name__ == '__main__':
    train_and_evaluate()
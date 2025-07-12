import os
import json
import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical
from sklearn.utils import class_weight
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from tensorflow.keras.callbacks import EarlyStopping

import config
from dataloader import load_and_prepare_data
from models.neural_networks import create_mlp, create_cnn
from utils import plot_learning_curves, plot_confusion_matrix, plot_roc_curves, CLASSES

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

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
                lr = params.get('learning_rate', 0.001)
                model = create_mlp(optimizer_name=params.get('optimizer', 'adam'), learning_rate=lr)
            else: # CNN
                lr = params.get('learning_rate', 0.001)
                model = create_cnn(optimizer_name=params.get('optimizer', 'adam'), learning_rate=lr)
            
            early_stopping = EarlyStopping(
                monitor='val_loss',  #monitora a perda no conjunto de validação
                patience=3,          
                verbose=1,
                restore_best_weights=True
            )
            
            history = model.fit(
                X_train, y_train,
                epochs=params.get('epochs', config.NN_EPOCHS),
                batch_size=params.get('batch_size', config.NN_BATCH_SIZE),
                validation_data=(X_val, y_val),
                class_weight=class_weights_dict,
                verbose=1,
                callbacks=[early_stopping]
            )
            plot_learning_curves(history, model_name, model_output_dir)

            history_path = os.path.join(model_output_dir, 'training_history.json')
            with open(history_path, 'w') as f:
                json.dump(history.history, f, cls=NpEncoder, indent=4)
            print(f"Histórico de treinamento salvo em: {history_path}")

        

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
        report_str = classification_report(y_test, y_pred, target_names=CLASSES)
        print(f"Acurácia no Teste para {model_name}: {accuracy:.4f}")
        print(report_str)

        report_dict = classification_report(y_test, y_pred, target_names=CLASSES, output_dict=True)
        report_json_path = os.path.join(model_output_dir, 'classification_report.json')
        report_txt_path = os.path.join(model_output_dir, 'classification_report.txt')
        with open(report_json_path, 'w') as f:
            json.dump(report_dict, f, indent=4)
        with open(report_txt_path, 'w') as f:
            f.write(f"Acurácia no Teste: {accuracy:.4f}\n\n")
            f.write(report_str)
        print(f"Relatório de classificação salvo em: {model_output_dir}")
        
        final_results.append({'Modelo': model_name, 'Acurácia Teste': accuracy})

    #salva o resumo
    summary_df = pd.DataFrame(final_results)
    print("\nResumo Comparativo -----------------------------------------------------")
    print(summary_df.to_string())
    summary_df.to_csv(os.path.join(config.OUTPUT_DIR, 'resumo_final.csv'), index=False)
    print(f"\nPipeline concluída!'{config.OUTPUT_DIR}'.")

if __name__ == '__main__':
    train_and_evaluate()
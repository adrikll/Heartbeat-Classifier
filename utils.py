import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import config
import os

CLASSES = ['Normal', 'Supraventricular', 'Ventricular', 'Fusion', 'Unknown']

def evaluate_and_compare_models(results, X_test, y_test):
    """
    Avalia os melhores modelos encontrados e gera um relatório comparativo.
    """
    print("\nAvaliação Final no Conjunto de Teste -----------")
    
    summary = []
    
    for model_name, result in results.items():
        best_model = result['best_estimator']
        
        y_pred = best_model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=CLASSES, output_dict=True)
        
        print(f"\n--- {model_name} ---")
        print(f"Acurácia no Teste: {accuracy:.4f}")
        print(classification_report(y_test, y_pred, target_names=CLASSES))
        
        summary.append({
            'Modelo': model_name,
            'Acurácia Teste': accuracy,
            'Melhores Parâmetros': result['best_params']
        })
        
    summary_df = pd.DataFrame(summary)
    print("\nResumo Comparativo dos Modelos ----------")
    print(summary_df.to_string())
    
    summary_df.to_csv(f"{config.OUTPUT_DIR}resumo_comparativo.csv", index=False)
    print(f"\nRelatório comparativo salvo em: {config.OUTPUT_DIR}resumo_comparativo.csv")


def plot_learning_curves(history, model_name, save_dir):
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Acurácia de Treino')
    plt.plot(history.history['val_accuracy'], label='Acurácia de Validação')
    plt.title(f'Curva de Acurácia - {model_name}')
    plt.xlabel('Época'); plt.ylabel('Acurácia'); plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Perda de Treino')
    plt.plot(history.history['val_loss'], label='Perda de Validação')
    plt.title(f'Curva de Perda - {model_name}')
    plt.xlabel('Época'); plt.ylabel('Perda'); plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'learning_curve.png'))
    plt.close()

def plot_confusion_matrix(y_true, y_pred, model_name, save_dir):
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title(f'Matriz de Confusão - {model_name}')
    plt.xlabel('Classe Prevista'); plt.ylabel('Classe Verdadeira')
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
    plt.close()

def plot_roc_curves(y_true_one_hot, y_pred_probs, model_name, save_dir):
    
    fpr, tpr, roc_auc = {}, {}, {}
    n_classes = y_true_one_hot.shape[1]
    plt.figure(figsize=(10, 8))
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_one_hot[:, i], y_pred_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], label=f'{CLASSES[i]} (AUC = {roc_auc[i]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('Taxa de Falsos Positivos'); plt.ylabel('Taxa de Verdadeiros Positivos')
    plt.title(f'Curva ROC Multi-classe - {model_name}'); plt.legend(loc="lower right")
    plt.savefig(os.path.join(save_dir, 'roc_curve.png'))
    plt.close()
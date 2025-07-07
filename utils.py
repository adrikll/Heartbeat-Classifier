import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
import config

CLASSES = ['Normal', 'Supraventricular', 'Ventricular', 'Fusion', 'Unknown']

def evaluate_and_compare_models(results, X_test, y_test):
    """
    Avalia os melhores modelos encontrados e gera um relatório comparativo.
    """
    print("\n--- Avaliação Final no Conjunto de Teste ---")
    
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
        
    # Imprimir tabela de resumo
    summary_df = pd.DataFrame(summary)
    print("\n--- Resumo Comparativo dos Modelos ---")
    print(summary_df.to_string())
    
    # Salvar resumo em CSV
    summary_df.to_csv(f"{config.OUTPUT_DIR}resumo_comparativo.csv", index=False)
    print(f"\nRelatório comparativo salvo em: {config.OUTPUT_DIR}resumo_comparativo.csv")
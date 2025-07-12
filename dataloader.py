import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import config

from sklearn.model_selection import train_test_split
import config

def load_and_prepare_data():
    """
    Carrega os dados e os divide em conjuntos de treino, validação e teste.
    """
    print("Carregando datasets...")
    df_train_full = pd.read_csv(config.TRAIN_FILE, header=None)
    df_test = pd.read_csv(config.TEST_FILE, header=None)

    # Separar features e rótulos do conjunto de teste final
    X_test = df_test.iloc[:, :-1].values
    y_test = df_test.iloc[:, -1].values

    # Separar o dataframe de treino completo em um novo treino e um conjunto de validação
    X_train, X_val, y_train, y_val = train_test_split(
        df_train_full.iloc[:, :-1],
        df_train_full.iloc[:, -1],
        test_size=config.VALIDATION_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=df_train_full.iloc[:, -1]
    )

    scaler = StandardScaler()
    
    #normaliza os dados de treino
    scaler.fit(X_train)
    
    #transforma todos os conjuntos de dados com o mesmo normalizador
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    print(f"Tamanho do treino: {len(X_train_scaled)}")
    print("Dados carregados, divididos e normalizados com sucesso.")

    #retorna os dados normalizados como arrays numpy
    return (X_train_scaled, y_train.values), (X_val_scaled, y_val.values), (X_test_scaled, y_test)

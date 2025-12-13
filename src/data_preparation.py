import pandas as pd
from sklearn.model_selection import train_test_split


def preparar_datos_modelo(df):
    """
    Prepara el dataset para el modelado:
    - Limpieza de nulos
    - Definición de variable objetivo (Cobertura 4G)
    - Codificación de variables categóricas
    - División train / test
    """

    # Copia del dataframe
    df = df.copy()

    # Variable objetivo: Cobertura 4G
    y = df["COBERTUTA 4G"]

    # Variables predictoras (eliminamos columnas de cobertura)
    X = df.drop(columns=[
        "COBERTURA 2G",
        "COBERTURA 3G",
        "COBERTURA HSPA+, HSPA+DC",
        "COBERTUTA 4G",
        "COBERTURA LTE",
        "COBERTURA 5G"
    ])

    # Eliminar valores nulos
    X = X.dropna()
    y = y.loc[X.index]

    # Codificación de variables categóricas
    X = pd.get_dummies(X, drop_first=True)

    # División entrenamiento / prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    return X_train, X_test, y_train, y_test

import pandas as pd
from sklearn.model_selection import train_test_split

def preparar_datos_modelo(df):

    df = df.copy()

    # Variable objetivo
    y = df["COBERTUTA 4G"]

    # Variables predictoras
    X = df.drop(columns=[
        "COBERTURA 2G",
        "COBERTURA 3G",
        "COBERTURA HSPA+, HSPA+DC",
        "COBERTUTA 4G",
        "COBERTURA LTE",
        "COBERTURA 5G"
    ])

    # Eliminar columnas de códigos (no predictivas)
    X = X.drop(columns=[
        "COD DEPARTAMENTO",
        "COD MUNICIPIO",
        "COD CENTRO POBLADO"
    ], errors="ignore")

    # Rellenar nulos
    X = X.fillna("DESCONOCIDO")

    # Codificación categórica
    X = pd.get_dummies(X, drop_first=True)

    # División train / test
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test

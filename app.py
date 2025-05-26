#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
svm_classification_viz.py

Carga datos de clientes, crea la etiqueta 'HighSpender', entrena un SVM
y muestra la matriz de confusión y el reporte de clasificación
como gráficos en lugar de texto en consola.
"""

import sys
from pathlib import Path
from typing import NoReturn

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.exceptions import NotFittedError
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Ruta al archivo y constantes
DATA_PATH = Path("Mall_Customers.csv")
SPENDING_THRESHOLD = 50
TEST_SIZE = 0.30
RANDOM_STATE = 42


def load_and_prepare(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Carga CSV, crea 'HighSpender' y separa X, y."""
    if not path.exists():
        print(f"ERROR: no se encontró {path}", file=sys.stderr)
        sys.exit(1)
    df = pd.read_csv(path)
    df["HighSpender"] = (df["Spending Score (1-100)"] > SPENDING_THRESHOLD).astype(int)
    X = df[["Age", "Annual Income (k$)"]]
    y = df["HighSpender"]
    return X, y


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> tuple[SVC, StandardScaler]:
    """Escala datos y entrena SVM, devuelve modelo y scaler."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    clf = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=RANDOM_STATE)
    clf.fit(X_scaled, y_train)
    return clf, scaler


def plot_confusion_matrix(cm: list[list[int]]) -> None:
    """Dibuja la matriz de confusión."""
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation="nearest", cmap=plt.get_cmap())
    plt.title("Matriz de Confusión")
    plt.colorbar()
    ticks = [0, 1]
    plt.xticks(ticks, ["No HighSpender", "HighSpender"])
    plt.yticks(ticks, ["No HighSpender", "HighSpender"])
    for i in ticks:
        for j in ticks:
            plt.text(j, i, cm[i][j], ha="center", va="center", fontsize=12)
    plt.xlabel("Predicción")
    plt.ylabel("Etiqueta Real")
    plt.tight_layout()
    plt.show()


def plot_classification_report(y_true: pd.Series, y_pred: pd.Series) -> None:
    """Convierte el reporte en tabla y la muestra."""
    report_dict = classification_report(y_true, y_pred, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose().round(2)
    plt.figure(figsize=(6, report_df.shape[0] * 0.5))
    plt.axis("off")
    table = plt.table(
        cellText=report_df.values,
        colLabels=report_df.columns,
        rowLabels=report_df.index,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    plt.title("Reporte de Clasificación", pad=20)
    plt.tight_layout()
    plt.show()


def main() -> NoReturn:
    """Punto de entrada: carga datos, entrena SVM y grafica resultados."""
    X, y = load_and_prepare(DATA_PATH)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    clf, scaler = train_model(X_train, y_train)

    try:
        X_test_scaled = scaler.transform(X_test)
    except Exception as e:
        print(f"ERROR al escalar datos de test: {e}", file=sys.stderr)
        sys.exit(1)

    y_pred = clf.predict(X_test_scaled)

    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm)

    plot_classification_report(y_test, y_pred)


if __name__ == "__main__":
    main()
"""Ingeniería de datos: Bronze -> Silver.

Cada transformación es una función pura: recibe un DataFrame y devuelve uno
nuevo. El orden importa y está fijado en build_silver().
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from flows.common.logging import get_logger
from flows.common.paths import (
    BRONZE,
    BRONZE_LABELED_NAME,
    SILVER,
    SILVER_FINAL_NAME,
    SILVER_NAME,
)

log = get_logger(__name__)

EXPECTED_COLUMNS = [
    "PassengerId",
    "Survived",
    "Pclass",
    "Name",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Ticket",
    "Fare",
    "Cabin",
    "Embarked",
]
DECKS = list("ABCDEFGTX")
TITLE_MAP = {
    "Master": "Mr",
    "the Countess": "Mrs",
    "Dona": "Mrs",
    "Mme": "Mrs",
    "Lady": "Miss",
    "Mlle": "Miss",
    "Ms": "Miss",
}
MAIN_TITLES = ["Mr", "Mrs", "Miss"]


def validate_bronze(df: pd.DataFrame) -> pd.DataFrame:
    missing = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Columnas faltantes en Bronze: {missing}")
    if df["PassengerId"].duplicated().any():
        raise ValueError("Existen valores duplicados en PassengerId")
    return df


def cast_types(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["Survived", "Pclass", "SibSp", "Parch"]:
        if col in out.columns and out[col].notna().all():
            out[col] = out[col].astype(int)
    for col in ["Age", "Fare"]:
        if col in out.columns:
            out[col] = out[col].astype(float)
    return out


def add_person(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega 'Person': 'child' si Age < 16; si no, el valor de Sex."""
    out = df.copy()
    out["Person"] = np.where(out["Age"] < 16, "child", out["Sex"])
    return out


def add_cabin_features(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega 'TCabin' (primera letra de Cabin, 'X' si es nula) y 'HasCabin' (0/1)."""
    out = df.copy()
    out["TCabin"] = out["Cabin"].astype("string").str[0].fillna("X").astype(str)
    out["HasCabin"] = out["Cabin"].notna().astype(int)
    return out


def add_family_features(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega 'Familia' = SibSp + Parch y 'Alone' ('with Family' si Familia > 0, 'Alone' si no)."""
    out = df.copy()
    out["Familia"] = out["SibSp"] + out["Parch"]
    out["Alone"] = np.where(out["Familia"] > 0, "with Family", "Alone")
    return out


def add_survivor_label(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega 'Survivor' mapeando 0 -> 'No' y 1 -> 'Yes'."""
    out = df.copy()
    if "Survived" in out.columns:
        out["Survivor"] = out["Survived"].map({0: "No", 1: "Yes"})
    return out


def add_title(df: pd.DataFrame) -> pd.DataFrame:
    """Extrae el título de 'Name' en 'TTitulo'."""
    out = df.copy()
    out["TTitulo"] = out["Name"].str.extract(r",\s*([^.]+)\.", expand=False).str.strip()
    return out


def consolidate_titles(df: pd.DataFrame) -> pd.DataFrame:
    """Consolida títulos poco frecuentes y agrupa los restantes en 'Otro'."""
    out = df.copy()
    replaced = out["TTitulo"].replace(TITLE_MAP)
    out["TTitulo"] = replaced.where(replaced.isin(MAIN_TITLES), "Otro")
    return out


def impute_embarked(df: pd.DataFrame) -> pd.DataFrame:
    """Rellena los faltantes de 'Embarked' con la moda."""
    out = df.copy()
    mode_val = out["Embarked"].mode()[0]
    out["Embarked"] = out["Embarked"].fillna(mode_val)
    return out


def impute_age(df: pd.DataFrame) -> pd.DataFrame:
    """Imputa 'Age' con el promedio por 'Sex' conservando 'AgeOriginal' y 'AgeImputada'."""
    out = df.copy()
    out["AgeOriginal"] = out["Age"]
    out["AgeImputada"] = out["Age"].isna()
    std_before = out["AgeOriginal"].std()
    promedios = out.groupby("Sex")["Age"].mean()
    out["Age"] = out["Age"].fillna(out["Sex"].map(promedios))
    std_after = out["Age"].std()
    log.info("Desviación estándar de Age antes=%.2f, después=%.2f", std_before, std_after)
    return out


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Crea variables numéricas NSex, NEmbarked, NCabin y NTitulo."""
    out = df.copy()
    out["NSex"] = out["Sex"].map({"female": 0, "male": 1})
    out["NEmbarked"] = out["Embarked"].map({"S": 0, "C": 1, "Q": 2})
    cabin_map = {letra: i for i, letra in enumerate(DECKS)}
    out["NCabin"] = out["TCabin"].map(cabin_map)
    out["NTitulo"] = out["TTitulo"].map({"Mr": 0, "Mrs": 1, "Miss": 2, "Otro": 3})
    return out


def drop_original_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Devuelve una COPIA sin Name, Sex, Embarked y Cabin."""
    cols_to_drop = [c for c in ["Name", "Sex", "Embarked", "Cabin"] if c in df.columns]
    return df.drop(columns=cols_to_drop).copy()


def validate_silver(df: pd.DataFrame) -> pd.DataFrame:
    required_no_nulls = [
        "Survived",
        "Pclass",
        "Age",
        "Fare",
        "Embarked",
        "Person",
        "Alone",
        "TCabin",
        "TTitulo",
        "NSex",
        "NEmbarked",
        "NCabin",
        "NTitulo",
    ]
    null_cols = [col for col in required_no_nulls if col not in df.columns or df[col].isna().any()]
    if null_cols:
        raise ValueError(f"Columnas faltantes o con nulos en Silver: {null_cols}")
    return df


def build_silver(bronze: pd.DataFrame) -> pd.DataFrame:
    """Orden deliberado: Person usa la edad ORIGINAL, así que va antes de imputar."""
    return (
        bronze.pipe(validate_bronze)
        .pipe(cast_types)
        .pipe(add_person)
        .pipe(add_cabin_features)
        .pipe(add_family_features)
        .pipe(add_survivor_label)
        .pipe(add_title)
        .pipe(consolidate_titles)
        .pipe(impute_embarked)
        .pipe(impute_age)
        .pipe(encode_categoricals)
    )


def run(in_dir: Path = BRONZE, out_dir: Path = SILVER) -> Path:
    bronze = pd.read_csv(Path(in_dir) / BRONZE_LABELED_NAME)
    silver = validate_silver(build_silver(bronze))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    silver.to_parquet(out_dir / SILVER_NAME, index=False)
    drop_original_columns(silver).to_parquet(out_dir / SILVER_FINAL_NAME, index=False)
    log.info("Silver listo: %d filas, %d columnas", *silver.shape)
    return out_dir / SILVER_NAME

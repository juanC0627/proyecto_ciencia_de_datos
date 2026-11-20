"""Agregaciones: Silver -> Gold. Cada función devuelve una tabla lista para un informe."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from scipy import stats

from flows.common.logging import get_logger
from flows.common.paths import GOLD, SILVER, SILVER_NAME

log = get_logger(__name__)


def group_stats(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Estadísticas descriptivas por grupos (Paso 28)."""
    return {
        "age_by_survival": df.groupby("Survived")["Age"].describe().reset_index(),
        "survival_by_sex": df.groupby("Sex")["Survived"]
        .agg(["mean", "sum", "count"])
        .reset_index(),
        "survival_by_pclass": df.groupby("Pclass")["Survived"]
        .agg(["mean", "sum", "count"])
        .reset_index(),
        "survival_by_alone": df.groupby("Alone")["Survived"]
        .agg(["mean", "sum", "count"])
        .reset_index(),
    }


def chi2_sex_survival(df: pd.DataFrame) -> dict:
    """Prueba chi-cuadrado de independencia entre Sex y Survived (Paso 29)."""
    tabla = pd.crosstab(df["Sex"], df["Survived"])
    chi2, p, gl, _ = stats.chi2_contingency(tabla)
    return {"chi2": float(chi2), "p_value": float(p), "dof": int(gl)}


def survival_by_class_sex(df: pd.DataFrame) -> pd.DataFrame:
    """Tasa de supervivencia por Pclass y Sex (Paso 34)."""
    piv = df.pivot_table(values="Survived", index="Pclass", columns="Sex", aggfunc="mean")
    return piv.rename(columns=str).reset_index()


def survival_by_embarked(df: pd.DataFrame) -> pd.DataFrame:
    """Tasa y cantidad de pasajeros por puerto de embarque (Paso 35.1)."""
    return (
        df.groupby("Embarked")["Survived"]
        .agg(survival_rate="mean", count="count")
        .reset_index()
    )


def survival_by_embarked_class(df: pd.DataFrame) -> pd.DataFrame:
    """Tasa de supervivencia por Embarked estratificada por Pclass (Paso 35.2)."""
    piv = df.pivot_table(values="Survived", index="Embarked", columns="Pclass", aggfunc="mean")
    return piv.rename(columns=str).reset_index()


def add_fare_quartile(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega 'FareQ' con cuartiles de Fare calculados dentro de cada Pclass (Paso 36.1)."""
    out = df.copy()
    out["FareQ"] = out.groupby("Pclass")["Fare"].transform(
        lambda s: pd.qcut(s, 4, labels=False, duplicates="drop")
    )
    return out


def survival_by_fare_quartile_class(df: pd.DataFrame) -> pd.DataFrame:
    """Tasa de supervivencia por cuartil de tarifa dentro de cada Pclass (Paso 36.2)."""
    with_q = add_fare_quartile(df)
    piv = with_q.pivot_table(values="Survived", index="FareQ", columns="Pclass", aggfunc="mean")
    return piv.rename(columns=str).reset_index()


def survival_by_family(df: pd.DataFrame) -> pd.DataFrame:
    """Tasa de supervivencia y cantidad de pasajeros por tamaño de familia (Paso 37)."""
    return (
        df.groupby("Familia")["Survived"]
        .agg(survival_rate="mean", count="count")
        .reset_index()
    )


def outlier_mask_iqr(s: pd.Series) -> pd.Series:
    """Máscara booleana de valores atípicos según regla 1.5 * IQR (Paso 38.1)."""
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return (s < lower) | (s > upper)


def outlier_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Resumen de atípicos para Fare, Age (AgeOriginal si existe) y Familia (Paso 38.2)."""
    age_col = "AgeOriginal" if "AgeOriginal" in df.columns else "Age"
    rows = []
    for label, col in [("Fare", "Fare"), ("Age", age_col), ("Familia", "Familia")]:
        serie = df[col].dropna()
        mask = outlier_mask_iqr(serie)
        n_outliers = int(mask.sum())
        row = {
            "variable": label,
            "n_valid": len(serie),
            "n_outliers": n_outliers,
            "pct_outliers": round((n_outliers / len(serie)) * 100, 2) if len(serie) > 0 else 0.0,
            "survival_rate_outliers": None,
            "survival_rate_non_outliers": None,
        }
        if label == "Fare":
            full_mask = outlier_mask_iqr(df["Fare"])
            if full_mask.any():
                row["survival_rate_outliers"] = float(df.loc[full_mask, "Survived"].mean())
            if (~full_mask).any():
                row["survival_rate_non_outliers"] = float(df.loc[~full_mask, "Survived"].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def run(in_dir: Path | None = None, out_dir: Path = GOLD) -> Path:
    df = pd.read_parquet((in_dir or SILVER) / SILVER_NAME)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        **group_stats(df),
        "survival_by_class_sex": survival_by_class_sex(df),
        "survival_by_embarked": survival_by_embarked(df),
        "survival_by_embarked_class": survival_by_embarked_class(df),
        "survival_by_fare_quartile_class": survival_by_fare_quartile_class(df),
        "survival_by_family": survival_by_family(df),
        "outlier_summary": outlier_summary(df),
    }
    for name, table in tables.items():
        table.to_parquet(out_dir / f"{name}.parquet", index=False)
    (out_dir / "chi2_sex_survival.json").write_text(
        json.dumps(chi2_sex_survival(df), indent=2), encoding="utf-8"
    )
    log.info("Gold listo: %d tablas en %s", len(tables) + 1, out_dir)
    return out_dir

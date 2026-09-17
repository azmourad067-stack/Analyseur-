"""HorseProno Quinté V1 — challenger expérimental spécialisé Quinté.

Principes:
- n'entraîne la cible que sur les supports Quinté officiels;
- utilise uniquement de l'historique strictement antérieur pour les statistiques d'entités;
- optimise une cible top-5, puis produit un classement/top-7;
- ne déduit jamais qu'une course est un Quinté à partir du nombre de partants.

Dépend de quinte_candidate.py (prepare, predict_scores) fourni dans le paquet d'analyse.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from quinte_candidate import prepare, predict_scores

DEFAULT_C = 0.005  # choisi sur fenêtre de validation 01–22/08/2026, pas sur le test 23/08–08/09


def add_race_id(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "race_id" not in df.columns:
        needed = {"race_date", "meeting_number", "race_number"}
        missing = needed - set(df.columns)
        if missing:
            raise ValueError(f"Colonnes manquantes pour construire race_id: {sorted(missing)}")
        df["race_id"] = (
            "R" + df["meeting_number"].astype(int).astype(str)
            + "C" + df["race_number"].astype(int).astype(str)
            + "_" + pd.to_datetime(df["race_date"]).dt.strftime("%Y-%m-%d")
        )
    return df


def valid_quinte_rows(history: pd.DataFrame, supports: pd.DataFrame) -> pd.DataFrame:
    history = add_race_id(history)
    supports = add_race_id(supports)
    q = history[history["race_id"].isin(set(supports["race_id"]))].copy()
    if "is_non_runner" in q:
        q = q[~q["is_non_runner"].fillna(False).astype(bool)]
    good = []
    for rid, g in q.groupby("race_id", sort=False):
        pos = pd.to_numeric(g["finish_position"], errors="coerce")
        top = sorted(pos[pos.between(1, 5)].astype(int).tolist())
        if top == [1, 2, 3, 4, 5] and not g["horse_number"].duplicated().any():
            good.append(rid)
    return q[q["race_id"].isin(good)].copy().reset_index(drop=True)


def train(args):
    history = pd.read_csv(args.history)
    history["race_date"] = pd.to_datetime(history["race_date"]).dt.normalize()
    supports = pd.read_csv(args.supports)
    q = valid_quinte_rows(history, supports)
    if q.empty:
        raise RuntimeError("Aucun support Quinté officiel avec arrivée top-5 complète.")

    # Les statistiques cheval/jockey/entraîneur peuvent apprendre de toutes les courses passées,
    # mais les lignes servant de cible d'apprentissage sont UNIQUEMENT les Quintés officiels.
    x = prepare(history, q)
    y = pd.to_numeric(q["finish_position"], errors="coerce").between(1, 5).astype(int)

    if args.train_end:
        fit = q["race_date"] <= pd.Timestamp(args.train_end)
    else:
        fit = pd.Series(True, index=q.index)
    if fit.sum() < 100:
        raise RuntimeError("Trop peu de partants Quinté pour entraîner le challenger.")

    scaler = StandardScaler().fit(x.loc[fit])
    z = scaler.transform(x.loc[fit])
    model = LogisticRegression(C=args.C, max_iter=3000, random_state=42)
    model.fit(z, y.loc[fit])

    artifact = {
        "model_name": "HorseProno_Quinte_V1_specialist",
        "status": "experimental_challenger",
        "target": "finish_position in 1..5",
        "C": args.C,
        "training_supports": int(q.loc[fit, "race_id"].nunique()),
        "training_end": str(q.loc[fit, "race_date"].max().date()),
        "features": list(x.columns),
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "coef": model.coef_[0].tolist(),
        "intercept": float(model.intercept_[0]),
        "notes": [
            "Targets learned only from official Quinté supports.",
            "Entity/history features are strictly prior-date via prepare().",
            "Score is a ranking score, not a calibrated probability of winning the Quinté.",
        ],
    }
    Path(args.model).write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(f"Modèle écrit: {args.model} — {artifact['training_supports']} Quintés")


def predict(args):
    history = pd.read_csv(args.history)
    race = pd.read_csv(args.race)
    history["race_date"] = pd.to_datetime(history["race_date"]).dt.normalize()
    race["race_date"] = pd.to_datetime(race["race_date"]).dt.normalize()
    race = add_race_id(race)
    if "is_non_runner" in race:
        race = race[~race["is_non_runner"].fillna(False).astype(bool)].copy()
    art = json.loads(Path(args.model).read_text(encoding="utf-8"))
    feats = prepare(history, race)
    race["quinte_v1_score"] = predict_scores(art, feats)
    race = race.sort_values(["race_id", "quinte_v1_score", "horse_number"], ascending=[True, False, True])
    race["quinte_rank"] = race.groupby("race_id").cumcount() + 1
    race["in_top7"] = race["quinte_rank"] <= 7
    race.to_csv(args.output, index=False)
    print(race[["race_id", "horse_number", "horse_name", "quinte_rank", "quinte_v1_score"]].head(7).to_string(index=False))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--history", required=True)
    t.add_argument("--supports", required=True, help="CSV des vrais supports Quinté officiels")
    t.add_argument("--model", default="quinte_v1_artifact.json")
    t.add_argument("--train-end", default=None)
    t.add_argument("--C", type=float, default=DEFAULT_C)
    t.set_defaults(func=train)
    r = sub.add_parser("predict")
    r.add_argument("--history", required=True)
    r.add_argument("--race", required=True)
    r.add_argument("--model", default="quinte_v1_artifact.json")
    r.add_argument("--output", default="pronostic_quinte_v1.csv")
    r.set_defaults(func=predict)
    args = p.parse_args(); args.func(args)

if __name__ == "__main__":
    main()

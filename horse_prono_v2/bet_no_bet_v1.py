from __future__ import annotations

"""
HorseProno BET / NO BET V1
==========================

Couche de décision au-dessus de M8 #8 et Neural #9.

Principes :
- aucun pari n'est autorisé par défaut ;
- les signaux sont calculés uniquement à partir des prédictions disponibles avant course ;
- le Simple Placé est le signal le mieux validé actuellement ;
- le Simple Gagnant reste volontairement conservateur ;
- Trio / Multi / Quinté utilisent surtout la stabilité du classement M8/Neural.

Repères forward observés au 27/09/2026 :
- 344 courses avec résultat exploitable ;
- Top1 M8 placé : 68,3 % sans filtre ;
- M8/Neural même Top1 + même Top3 : 74,3 % ;
- + P(placé) M8 Top1 >= 0,70 : 78,8 % sur 100 courses ;
- règle forte par blocs temporels 12-15 / 16-18 / 19-20 septembre :
  78,4 % / 76,2 % / 81,0 %.
Ces taux sont descriptifs et ne garantissent pas les résultats futurs.
"""

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np
import pandas as pd

PLACE_STRONG_MIN_PROB = 0.70
PLACE_WATCH_MIN_PROB = 0.55
WIN_MIN_EDGE = 0.02
WIN_MIN_PROB = 0.25

REFERENCE_STATS = {
    "forward_result_races": 344,
    "place_baseline_rate": 0.683,
    "place_full_agreement_rate": 0.743,
    "place_strong_rule_rate": 0.788,
    "place_strong_rule_n": 100,
    "trio_top7_baseline_rate": 0.558,
    "trio_top7_full_agreement_rate": 0.612,
}


@dataclass
class BetDecision:
    game: str
    decision: str
    confidence: int
    horse_number: int | None = None
    horse_name: str | None = None
    reason: str = ""
    evidence: str = ""
    stake_level: str = "AUCUNE"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _num(value: Any, default: float = 0.0) -> float:
    try:
        value = float(value)
        if not np.isfinite(value):
            return default
        return value
    except Exception:
        return default


def _int(value: Any, default: int | None = None) -> int | None:
    try:
        if pd.isna(value):
            return default
        return int(value)
    except Exception:
        return default


def _rank_frame(
    frame: pd.DataFrame,
    probability_col: str,
    rank_col: str,
) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()

    out = frame.copy()
    out[probability_col] = pd.to_numeric(out.get(probability_col), errors="coerce")
    out["horse_number"] = pd.to_numeric(out.get("horse_number"), errors="coerce")
    out = out[out["horse_number"].notna()].copy()
    out["horse_number"] = out["horse_number"].astype(int)

    if rank_col not in out.columns:
        out = out.sort_values(
            [probability_col, "horse_number"],
            ascending=[False, True],
            na_position="last",
        ).reset_index(drop=True)
        out[rank_col] = np.arange(1, len(out) + 1)
    else:
        out[rank_col] = pd.to_numeric(out[rank_col], errors="coerce")
        out = out.sort_values(
            [rank_col, probability_col, "horse_number"],
            ascending=[True, False, True],
            na_position="last",
        ).reset_index(drop=True)

    return out


def _horse_set(frame: pd.DataFrame, rank_col: str, n: int) -> set[int]:
    if frame.empty or rank_col not in frame.columns:
        return set()
    rows = frame[pd.to_numeric(frame[rank_col], errors="coerce") <= n]
    return set(
        pd.to_numeric(rows["horse_number"], errors="coerce")
        .dropna()
        .astype(int)
        .tolist()
    )


def build_signals(
    m8_predictions: pd.DataFrame,
    neural_predictions: pd.DataFrame,
    *,
    field_size: int | None = None,
    discipline: str | None = None,
) -> dict[str, Any]:
    """Construit les signaux pré-course communs à tous les types de pari."""
    m8 = _rank_frame(m8_predictions, "place_probability", "m8_place_rank")
    neural = _rank_frame(
        neural_predictions,
        "neural_top3_probability",
        "neural_rank",
    )

    if m8.empty:
        raise ValueError("Classement M8 vide.")
    if neural.empty:
        raise ValueError("Classement Neural vide.")

    if "win_probability" in m8.columns:
        m8["win_probability"] = pd.to_numeric(m8["win_probability"], errors="coerce")
        if "m8_win_rank" not in m8.columns:
            m8["m8_win_rank"] = (
                m8["win_probability"]
                .rank(method="first", ascending=False)
                .astype("Int64")
            )
    else:
        m8["win_probability"] = np.nan
        m8["m8_win_rank"] = pd.Series([pd.NA] * len(m8), dtype="Int64")

    if "market_probability" in m8.columns:
        m8["market_probability_num"] = pd.to_numeric(
            m8["market_probability"], errors="coerce"
        )
    elif "market_probability_num" in m8.columns:
        m8["market_probability_num"] = pd.to_numeric(
            m8["market_probability_num"], errors="coerce"
        )
    else:
        odds = pd.to_numeric(m8.get("odds"), errors="coerce").clip(lower=1.01)
        inv = 1.0 / odds
        total = inv.sum(skipna=True)
        if np.isfinite(total) and total > 0:
            m8["market_probability_num"] = inv.fillna(0.0) / total
        else:
            m8["market_probability_num"] = 0.0

    m8_place_top = m8.sort_values(
        ["m8_place_rank", "horse_number"]
    ).reset_index(drop=True)
    neural_top = neural.sort_values(
        ["neural_rank", "horse_number"]
    ).reset_index(drop=True)

    m8_top1 = m8_place_top.iloc[0]
    neural_top1 = neural_top.iloc[0]

    m8_top3 = _horse_set(m8_place_top, "m8_place_rank", 3)
    neural_top3 = _horse_set(neural_top, "neural_rank", 3)
    m8_top5 = _horse_set(m8_place_top, "m8_place_rank", 5)
    neural_top5 = _horse_set(neural_top, "neural_rank", 5)
    m8_top7 = _horse_set(m8_place_top, "m8_place_rank", 7)
    neural_top7 = _horse_set(neural_top, "neural_rank", 7)

    p1 = _num(m8_top1.get("place_probability"), 0.0)
    p2 = (
        _num(m8_place_top.iloc[1].get("place_probability"), 0.0)
        if len(m8_place_top) > 1
        else 0.0
    )

    place_probs = pd.to_numeric(
        m8_place_top["place_probability"], errors="coerce"
    ).fillna(0.0)
    total_place_mass = float(place_probs.clip(lower=0.0).sum())
    top7_mass = float(place_probs.head(7).clip(lower=0.0).sum())
    top7_concentration = (
        top7_mass / total_place_mass if total_place_mass > 0 else 0.0
    )

    win_sorted = m8.sort_values(
        ["m8_win_rank", "horse_number"],
        na_position="last",
    ).reset_index(drop=True)
    win_top = win_sorted.iloc[0]
    win_prob = _num(win_top.get("win_probability"), 0.0)
    market_prob = _num(win_top.get("market_probability_num"), 0.0)
    win_edge = win_prob - market_prob

    actual_field_size = int(field_size or len(m8))
    discipline_text = str(discipline or "").strip().upper()

    return {
        "field_size": actual_field_size,
        "discipline": discipline_text,
        "m8_top1_number": _int(m8_top1.get("horse_number")),
        "m8_top1_name": str(m8_top1.get("horse_name") or ""),
        "neural_top1_number": _int(neural_top1.get("horse_number")),
        "top1_agree": (
            _int(m8_top1.get("horse_number"))
            == _int(neural_top1.get("horse_number"))
        ),
        "top3_overlap": len(m8_top3 & neural_top3),
        "top5_overlap": len(m8_top5 & neural_top5),
        "top7_overlap": len(m8_top7 & neural_top7),
        "m8_place_p1": p1,
        "m8_place_p2": p2,
        "place_gap": p1 - p2,
        "top7_concentration": top7_concentration,
        "win_candidate_number": _int(win_top.get("horse_number")),
        "win_candidate_name": str(win_top.get("horse_name") or ""),
        "win_probability": win_prob,
        "win_market_probability": market_prob,
        "win_edge": win_edge,
    }


def _place_decision(s: dict[str, Any]) -> BetDecision:
    p1 = s["m8_place_p1"]
    same1 = s["top1_agree"]
    ov3 = s["top3_overlap"]

    if p1 >= PLACE_STRONG_MIN_PROB and same1 and ov3 == 3:
        return BetDecision(
            game="Simple placé",
            decision="BET",
            confidence=86,
            horse_number=s["m8_top1_number"],
            horse_name=s["m8_top1_name"],
            reason="P(placé) M8 ≥ 0,70 + même n°1 M8/Neural + Top 3 identique.",
            evidence="78,8 % placés sur 100 courses forward pour cette règle.",
            stake_level="NORMALE",
        )

    if p1 >= PLACE_STRONG_MIN_PROB and same1:
        return BetDecision(
            game="Simple placé",
            decision="PRUDENT",
            confidence=74,
            horse_number=s["m8_top1_number"],
            horse_name=s["m8_top1_name"],
            reason="P(placé) M8 ≥ 0,70 et même n°1, mais Top 3 non identique.",
            evidence="Signal positif, moins strict que la règle forte.",
            stake_level="RÉDUITE",
        )

    if p1 >= PLACE_WATCH_MIN_PROB and same1 and ov3 >= 2:
        return BetDecision(
            game="Simple placé",
            decision="PRUDENT",
            confidence=66,
            horse_number=s["m8_top1_number"],
            horse_name=s["m8_top1_name"],
            reason="Accord partiel M8/Neural avec P(placé) M8 ≥ 0,55.",
            evidence="Zone intermédiaire : pas assez forte pour un BET standard.",
            stake_level="RÉDUITE",
        )

    return BetDecision(
        game="Simple placé",
        decision="NO BET",
        confidence=max(25, min(59, int(round(p1 * 100)))),
        horse_number=s["m8_top1_number"],
        horse_name=s["m8_top1_name"],
        reason="Confiance pré-course insuffisante pour la règle placé V1.",
        evidence="Le système préfère laisser passer la course.",
        stake_level="AUCUNE",
    )


def _win_decision(s: dict[str, Any]) -> BetDecision:
    conditions = (
        s["top1_agree"]
        and s["top3_overlap"] == 3
        and s["win_probability"] >= WIN_MIN_PROB
        and s["win_edge"] >= WIN_MIN_EDGE
    )

    if conditions:
        return BetDecision(
            game="Simple gagnant",
            decision="PRUDENT",
            confidence=64,
            horse_number=s["win_candidate_number"],
            horse_name=s["win_candidate_name"],
            reason=(
                "Accord M8/Neural fort + probabilité gagnant suffisante "
                "+ edge ≥ 2 pp."
            ),
            evidence=(
                "V1 ne donne pas encore de BET fort en gagnant : "
                "le rang 1 forward reste trop instable."
            ),
            stake_level="RÉDUITE",
        )

    return BetDecision(
        game="Simple gagnant",
        decision="NO BET",
        confidence=45,
        horse_number=s["win_candidate_number"],
        horse_name=s["win_candidate_name"],
        reason="Le filtre gagnant conservateur n'est pas validé sur cette course.",
        evidence="Priorité actuelle au Simple Placé et aux jeux combinés.",
        stake_level="AUCUNE",
    )


def _trio_decision(s: dict[str, Any]) -> BetDecision:
    ov3 = s["top3_overlap"]
    ov5 = s["top5_overlap"]

    if ov3 == 3 and s["m8_place_p1"] >= 0.70:
        return BetDecision(
            game="Trio",
            decision="BET",
            confidence=79,
            reason="Top 3 M8/Neural identique et leader M8 très solide.",
            evidence=(
                "Avec accord complet, le podium réel était dans le Top 7 M8 "
                "dans 61,2 % des courses forward observées."
            ),
            stake_level="NORMALE",
        )

    if ov3 >= 2 and ov5 >= 4:
        return BetDecision(
            game="Trio",
            decision="PRUDENT",
            confidence=68,
            reason="Bon recouvrement M8/Neural dans le Top 3 et Top 5.",
            evidence="Signal combiné intéressant mais accord incomplet.",
            stake_level="RÉDUITE",
        )

    return BetDecision(
        game="Trio",
        decision="NO BET",
        confidence=50,
        reason="Désaccord trop important entre M8 et Neural.",
        evidence="La stabilité du haut de classement est insuffisante.",
        stake_level="AUCUNE",
    )


def _multi_decision(s: dict[str, Any]) -> BetDecision:
    ov3 = s["top3_overlap"]
    ov5 = s["top5_overlap"]
    fs = s["field_size"]

    if ov3 == 3 and ov5 >= 4 and s["m8_place_p1"] >= 0.65 and fs <= 16:
        return BetDecision(
            game="Multi",
            decision="BET",
            confidence=77,
            reason="Top 3 stable, Top 5 très proche et peloton exploitable.",
            evidence="Règle V1 de stabilité ; à suivre prospectivement séparément.",
            stake_level="NORMALE",
        )

    if ov3 >= 2 and ov5 >= 3 and fs <= 16:
        return BetDecision(
            game="Multi",
            decision="PRUDENT",
            confidence=65,
            reason="Structure assez stable pour un Multi élargi.",
            evidence="Signal provisoire, moins validé que le Simple Placé.",
            stake_level="RÉDUITE",
        )

    return BetDecision(
        game="Multi",
        decision="NO BET",
        confidence=48,
        reason="Classement trop dispersé ou peloton trop difficile pour le V1.",
        evidence="Le système évite de forcer un Multi.",
        stake_level="AUCUNE",
    )


def _quinte_decision(s: dict[str, Any]) -> BetDecision:
    ov3 = s["top3_overlap"]
    ov5 = s["top5_overlap"]
    c7 = s["top7_concentration"]
    fs = s["field_size"]

    if (
        fs >= 10
        and ov3 == 3
        and ov5 >= 4
        and s["m8_place_p1"] >= 0.70
        and c7 >= 0.72
    ):
        return BetDecision(
            game="Quinté",
            decision="PRUDENT",
            confidence=69,
            reason=(
                "Haut de classement stable et forte concentration M8 "
                "dans le Top 7."
            ),
            evidence=(
                "Le V1 refuse encore un BET fort Quinté tant que ce filtre "
                "n'a pas assez de suivi prospectif."
            ),
            stake_level="RÉDUITE",
        )

    return BetDecision(
        game="Quinté",
        decision="NO BET",
        confidence=46,
        reason="La structure Top 7 n'est pas assez concentrée/stable.",
        evidence="Quinté volontairement très sélectif dans V1.",
        stake_level="AUCUNE",
    )


def evaluate_bet_no_bet(
    m8_predictions: pd.DataFrame,
    neural_predictions: pd.DataFrame,
    *,
    field_size: int | None = None,
    discipline: str | None = None,
) -> dict[str, Any]:
    signals = build_signals(
        m8_predictions,
        neural_predictions,
        field_size=field_size,
        discipline=discipline,
    )

    decisions = [
        _place_decision(signals),
        _win_decision(signals),
        _trio_decision(signals),
        _multi_decision(signals),
        _quinte_decision(signals),
    ]

    return {
        "version": "BET_NO_BET_V1",
        "signals": signals,
        "decisions": [d.to_dict() for d in decisions],
        "reference_stats": dict(REFERENCE_STATS),
    }


def decisions_dataframe(result: dict[str, Any]) -> pd.DataFrame:
    decisions = result.get("decisions") or []
    if not decisions:
        return pd.DataFrame()
    return pd.DataFrame(decisions)

"""
Streamlit App — Prédicteur de suite logique
Charge un fichier Excel contenant une suite de chiffres, détecte des patterns,
et prédit les 5 prochaines valeurs.
"""

import io
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Prédicteur de suite logique",
    page_icon="🔢",
    layout="wide",
)

N_PREDICT = 5  # nombre de valeurs à prédire


# ==================================================================
# 1. LECTURE DU FICHIER EXCEL
# ==================================================================
def load_sequence_from_excel(file, sheet_name=0, column=None) -> List[float]:
    """Lit un fichier Excel et renvoie une liste de nombres."""
    df = pd.read_excel(file, sheet_name=sheet_name, header=None)

    # Si l'utilisateur a précisé une colonne
    if column is not None and column in df.columns:
        series = df[column]
    else:
        # On prend la première colonne contenant des nombres
        numeric_df = df.apply(pd.to_numeric, errors="coerce")
        # Choisir la colonne avec le plus de valeurs numériques
        col_scores = numeric_df.notna().sum()
        best_col = col_scores.idxmax()
        series = numeric_df[best_col]

    values = pd.to_numeric(series, errors="coerce").dropna().tolist()
    return [float(v) for v in values]


# ==================================================================
# 2. DÉTECTEURS DE PATTERNS
# ==================================================================
def detect_arithmetic(seq: List[float]) -> Optional[Dict]:
    """Suite arithmétique : u(n+1) = u(n) + r"""
    if len(seq) < 3:
        return None
    diffs = np.diff(seq)
    if np.allclose(diffs, diffs[0], atol=1e-6):
        r = float(diffs[0])
        preds = [seq[-1] + r * (i + 1) for i in range(N_PREDICT)]
        return {
            "name": "Arithmétique",
            "formula": f"u(n+1) = u(n) + {r:g}",
            "predictions": preds,
            "confidence": 1.0,
        }
    return None


def detect_geometric(seq: List[float]) -> Optional[Dict]:
    """Suite géométrique : u(n+1) = u(n) * q"""
    if len(seq) < 3 or any(v == 0 for v in seq[:-1]):
        return None
    ratios = np.array(seq[1:]) / np.array(seq[:-1])
    if np.allclose(ratios, ratios[0], atol=1e-6):
        q = float(ratios[0])
        preds = [seq[-1] * (q ** (i + 1)) for i in range(N_PREDICT)]
        return {
            "name": "Géométrique",
            "formula": f"u(n+1) = u(n) × {q:g}",
            "predictions": preds,
            "confidence": 1.0,
        }
    return None


def detect_fibonacci_like(seq: List[float]) -> Optional[Dict]:
    """Suite de type Fibonacci : u(n) = u(n-1) + u(n-2)"""
    if len(seq) < 4:
        return None
    ok = all(
        abs(seq[i] - (seq[i - 1] + seq[i - 2])) < 1e-6
        for i in range(2, len(seq))
    )
    if ok:
        preds = []
        a, b = seq[-2], seq[-1]
        for _ in range(N_PREDICT):
            nxt = a + b
            preds.append(nxt)
            a, b = b, nxt
        return {
            "name": "Fibonacci-like",
            "formula": "u(n) = u(n-1) + u(n-2)",
            "predictions": preds,
            "confidence": 1.0,
        }
    return None


def detect_polynomial(seq: List[float], max_degree: int = 4) -> Optional[Dict]:
    """
    Régression polynomiale : cherche le degré minimal qui fitte parfaitement (ou presque).
    """
    if len(seq) < 3:
        return None

    x = np.arange(len(seq)).reshape(-1, 1)
    y = np.array(seq)

    best = None
    for deg in range(1, min(max_degree, len(seq) - 1) + 1):
        poly = PolynomialFeatures(degree=deg)
        X_poly = poly.fit_transform(x)
        model = LinearRegression().fit(X_poly, y)
        y_pred = model.predict(X_poly)
        mse = float(np.mean((y - y_pred) ** 2))
        rel_err = mse / (np.var(y) + 1e-9)

        if best is None or rel_err < best["rel_err"]:
            # Prédictions
            x_future = np.arange(len(seq), len(seq) + N_PREDICT).reshape(-1, 1)
            X_future_poly = poly.fit_transform(x_future)
            preds = model.predict(X_future_poly).tolist()

            coeffs = model.coef_
            intercept = model.intercept_
            formula = f"P(n) polynôme degré {deg}"
            best = {
                "name": f"Polynomial (degré {deg})",
                "formula": formula,
                "predictions": preds,
                "confidence": max(0.0, 1.0 - rel_err),
                "rel_err": rel_err,
                "degree": deg,
            }
            # Si on a trouvé un fit quasi-parfait, on s'arrête
            if rel_err < 1e-8:
                break

    if best is not None:
        best.pop("rel_err", None)
        return best
    return None


def detect_diff_recurrence(seq: List[float]) -> Optional[Dict]:
    """
    Détecte une régularité dans les différences d'ordre supérieur
    (utile pour des suites du type 1, 4, 9, 16, 25 → diffs de diffs constantes).
    """
    if len(seq) < 4:
        return None
    arr = np.array(seq, dtype=float)
    diffs = arr.copy()
    for order in range(1, min(5, len(seq) - 1)):
        diffs = np.diff(diffs)
        if len(diffs) >= 2 and np.allclose(diffs, diffs[0], atol=1e-6):
            # Reconstruire vers l'avant en gardant les différences d'ordre `order` constantes
            preds = []
            # État courant = derniers termes de chaque niveau de différences
            levels = [arr.tolist()]
            tmp = arr.copy()
            for _ in range(order):
                tmp = np.diff(tmp)
                levels.append(tmp.tolist())
            const = float(diffs[0])
            for _ in range(N_PREDICT):
                # Étend le dernier niveau avec la constante
                levels[-1].append(const)
                # Remonte
                for lvl in range(len(levels) - 2, -1, -1):
                    levels[lvl].append(levels[lvl][-1] + levels[lvl + 1][-1])
                preds.append(levels[0][-1])
            return {
                "name": f"Différences d'ordre {order} constantes",
                "formula": f"Δ^{order} = {const:g}",
                "predictions": preds,
                "confidence": 0.95,
            }
    return None


def detect_linear_recurrence(seq: List[float], max_order: int = 3) -> Optional[Dict]:
    """
    Détecte une récurrence linéaire à coefficients constants :
        u(n) = a1*u(n-1) + a2*u(n-2) + ... + ak*u(n-k) + b
    """
    n = len(seq)
    if n < 6:
        return None

    best = None
    for k in range(1, min(max_order, n // 2) + 1):
        # Construit le système : y = X @ [a1, ..., ak, b]
        X = []
        y = []
        for i in range(k, n):
            X.append(seq[i - k:i][::-1] + [1.0])
            y.append(seq[i])
        X = np.array(X)
        y = np.array(y)
        if X.shape[0] < X.shape[1]:
            continue
        # Moindres carrés
        try:
            coefs, *_ = np.linalg.lstsq(X, y, rcond=None)
        except np.linalg.LinAlgError:
            continue
        y_pred = X @ coefs
        mse = float(np.mean((y - y_pred) ** 2))
        rel_err = mse / (np.var(y) + 1e-9)

        if rel_err < 1e-6 and (best is None or rel_err < best["rel_err"]):
            a = coefs[:-1]
            b = float(coefs[-1])
            preds = []
            history = list(seq)
            for _ in range(N_PREDICT):
                nxt = float(np.dot(a, history[-k:][::-1]) + b)
                preds.append(nxt)
                history.append(nxt)
            formula = " + ".join(
                [f"{a[j]:.4g}·u(n-{j+1})" for j in range(k)]
            )
            if abs(b) > 1e-9:
                formula += f" + {b:.4g}"
            best = {
                "name": f"Récurrence linéaire (ordre {k})",
                "formula": f"u(n) = {formula}",
                "predictions": preds,
                "confidence": max(0.0, 1.0 - rel_err),
                "rel_err": rel_err,
            }

    if best is not None:
        best.pop("rel_err", None)
        return best
    return None


def ml_fallback(seq: List[float]) -> Dict:
    """
    Fallback : régression polynomiale de meilleur degré (basé sur validation simple).
    Toujours retourne un résultat.
    """
    n = len(seq)
    x = np.arange(n).reshape(-1, 1)
    y = np.array(seq)

    best_deg = 1
    best_mse = float("inf")
    # Validation : on garde les 20% derniers points pour évaluer
    split = max(2, int(n * 0.8))
    x_train, y_train = x[:split], y[:split]
    x_val, y_val = x[split:], y[split:]

    for deg in range(1, min(5, n - 1) + 1):
        poly = PolynomialFeatures(degree=deg)
        X_train = poly.fit_transform(x_train)
        model = LinearRegression().fit(X_train, y_train)
        if len(x_val) > 0:
            X_val = poly.fit_transform(x_val)
            pred_val = model.predict(X_val)
            mse = float(np.mean((y_val - pred_val) ** 2))
        else:
            mse = float(np.mean((y_train - model.predict(X_train)) ** 2))
        if mse < best_mse:
            best_mse = mse
            best_deg = deg

    poly = PolynomialFeatures(degree=best_deg)
    X_full = poly.fit_transform(x)
    model = LinearRegression().fit(X_full, y)
    x_future = np.arange(n, n + N_PREDICT).reshape(-1, 1)
    preds = model.predict(poly.fit_transform(x_future)).tolist()

    rel_err = best_mse / (np.var(y) + 1e-9)
    return {
        "name": f"ML — Régression polynomiale (degré {best_deg})",
        "formula": f"Modèle polynomial degré {best_deg} (fallback)",
        "predictions": preds,
        "confidence": max(0.0, 1.0 - rel_err),
    }


# ==================================================================
# 3. ORCHESTRATEUR
# ==================================================================
def analyze_sequence(seq: List[float]) -> Tuple[Dict, List[Dict]]:
    """
    Applique tous les détecteurs, retourne (meilleur_pattern, tous_les_patterns).
    Ordre de priorité : arithmétique > géométrique > fibonacci > diff. constantes
    > récurrence linéaire > polynomial > ML fallback.
    """
    candidates = []
    detectors = [
        detect_arithmetic,
        detect_geometric,
        detect_fibonacci_like,
        detect_diff_recurrence,
        detect_linear_recurrence,
        detect_polynomial,
    ]
    for det in detectors:
        try:
            res = det(seq)
            if res is not None:
                candidates.append(res)
        except Exception as e:
            candidates.append({
                "name": det.__name__,
                "error": str(e),
                "confidence": 0.0,
                "predictions": [],
                "formula": "",
            })

    # Toujours ajouter le fallback ML
    candidates.append(ml_fallback(seq))

    # Meilleur = plus haute confiance
    valid = [c for c in candidates if c.get("predictions")]
    best = max(valid, key=lambda c: c.get("confidence", 0.0))
    return best, candidates


# ==================================================================
# 4. INTERFACE STREAMLIT
# ==================================================================
def main():
    st.title("🔢 Prédicteur de suite logique")
    st.markdown(
        "Charge un fichier **Excel** contenant une suite de chiffres. "
        "L'app détecte le pattern (arithmétique, géométrique, Fibonacci, "
        "polynomial, récurrence linéaire, ML) et prédit les **5 prochains chiffres**."
    )

    # ---- Sidebar ----
    with st.sidebar:
        st.header("⚙️ Options")
        show_all = st.checkbox("Afficher tous les patterns testés", value=True)
        st.markdown("---")
        st.markdown(
            "**Format attendu du fichier Excel :**\n\n"
            "Une colonne contenant des nombres (avec ou sans en-tête). "
            "L'app détecte automatiquement la colonne numérique."
        )

    # ---- Upload ----
    uploaded = st.file_uploader(
        "📂 Dépose ton fichier Excel (.xlsx / .xls)",
        type=["xlsx", "xls"],
    )

    # Option : saisie manuelle pour tester
    with st.expander("Ou saisir une suite manuellement (test rapide)"):
        manual = st.text_input(
            "Ex : 2, 4, 6, 8, 10",
            value="",
            help="Entre des nombres séparés par des virgules ou des espaces.",
        )

    sequence = None

    if uploaded is not None:
        try:
            sequence = load_sequence_from_excel(uploaded)
            st.success(f"✅ Fichier chargé — {len(sequence)} valeurs détectées.")
        except Exception as e:
            st.error(f"❌ Erreur de lecture du fichier : {e}")

    elif manual.strip():
        try:
            raw = manual.replace(",", " ").split()
            sequence = [float(x) for x in raw if x.strip()]
            st.info(f"ℹ️ Suite manuelle : {len(sequence)} valeurs.")
        except ValueError:
            st.error("❌ Format invalide. Utilise des nombres séparés par des virgules ou espaces.")

    if sequence is None:
        st.stop()

    if len(sequence) < 3:
        st.warning("⚠️ Il faut au moins 3 valeurs pour détecter un pattern.")
        st.stop()

    # ---- Affichage de la suite ----
    st.subheader("📊 Suite d'entrée")
    col1, col2 = st.columns([2, 1])
    with col1:
        df_in = pd.DataFrame({"n": range(len(sequence)), "u(n)": sequence})
        st.dataframe(df_in, use_container_width=True, height=200)
    with col2:
        st.metric("Nombre de valeurs", len(sequence))
        st.metric("Min", f"{min(sequence):g}")
        st.metric("Max", f"{max(sequence):g}")

    # ---- Analyse ----
    with st.spinner("🔍 Recherche du meilleur pattern..."):
        best, all_patterns = analyze_sequence(sequence)

    # ---- Résultat principal ----
    st.subheader("🎯 Prédiction des 5 prochains chiffres")
    st.markdown(f"**Pattern retenu :** `{best['name']}`")
    st.markdown(f"**Formule :** `{best['formula']}`")
    st.markdown(f"**Confiance :** `{best.get('confidence', 0):.2%}`")

    preds = best["predictions"]
    # Arrondi propre si les entrées sont entières
    all_int = all(float(v).is_integer() for v in sequence)
    if all_int:
        display_preds = [int(round(p)) for p in preds]
    else:
        display_preds = [round(p, 4) for p in preds]

    st.success("**Suite prédite :** " + " → ".join(str(v) for v in display_preds))

    pred_df = pd.DataFrame({
        "Rang": [f"u({len(sequence) + i})" for i in range(N_PREDICT)],
        "Valeur prédite": display_preds,
    })
    st.table(pred_df)

    # ---- Graphique ----
    st.subheader("📈 Visualisation")
    fig, ax = plt.subplots(figsize=(10, 4))
    x_in = list(range(len(sequence)))
    x_pred = list(range(len(sequence), len(sequence) + N_PREDICT))
    ax.plot(x_in, sequence, "o-", label="Suite fournie", color="#1f77b4")
    ax.plot(x_pred, preds, "s--", label="Prédiction (5 valeurs)", color="#d62728")
    ax.axvline(len(sequence) - 0.5, color="gray", ls=":", alpha=0.5)
    ax.set_xlabel("n")
    ax.set_ylabel("u(n)")
    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig)

    # ---- Tous les patterns testés ----
    if show_all:
        st.subheader("🧪 Tous les patterns testés")
        rows = []
        for p in all_patterns:
            if "error" in p:
                rows.append({
                    "Pattern": p["name"],
                    "Formule": f"❌ {p['error']}",
                    "Confiance": "-",
                    "Prédictions": "-",
                })
            else:
                preds_p = p.get("predictions", [])
                if all_int and preds_p:
                    preds_p = [int(round(v)) for v in preds_p]
                else:
                    preds_p = [round(v, 3) for v in preds_p]
                rows.append({
                    "Pattern": p["name"],
                    "Formule": p.get("formula", ""),
                    "Confiance": f"{p.get('confidence', 0):.2%}",
                    "Prédictions": ", ".join(str(v) for v in preds_p),
                })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

    # ---- Export ----
    st.subheader("💾 Exporter les résultats")
    export_df = pd.concat([
        pd.DataFrame({"n": range(len(sequence)), "valeur": sequence, "type": "input"}),
        pd.DataFrame({
            "n": range(len(sequence), len(sequence) + N_PREDICT),
            "valeur": display_preds,
            "type": "prediction",
        }),
    ], ignore_index=True)

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        export_df.to_excel(writer, index=False, sheet_name="Résultat")
    buf.seek(0)

    st.download_button(
        label="📥 Télécharger le résultat en Excel",
        data=buf,
        file_name="prediction_suite.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


if __name__ == "__main__":
    main()

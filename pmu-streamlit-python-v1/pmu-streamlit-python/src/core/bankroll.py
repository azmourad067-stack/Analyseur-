def calculer_mise_optimale(cote, proba_ia, bankroll, kelly_fraction=0.25):
    proba = proba_ia / 100 if proba_ia > 1 else proba_ia
    edge = cote * proba - 1
    if edge <= 0:
        return {"mise": 0, "edge": edge, "roi_attendu": 0, "recommandation": "SKIP", "raison": "Pas d'avantage statistique"}
    kelly = (edge / (cote - 1)) * kelly_fraction
    mise = bankroll * kelly
    if mise < 2:
        return {"mise": 0, "edge": edge, "roi_attendu": edge*100, "recommandation": "SKIP", "raison": "Mise calculée trop faible"}
    mise = min(mise, bankroll*.05)
    rec = "TRÈS FORTE" if edge > 1 else ("FORTE" if edge > .5 else "STANDARD")
    return {"mise": round(mise,2), "edge": round(edge,3), "roi_attendu": round(edge*100,2), "gain_potentiel": round(mise*(cote-1),2), "kelly_fraction": kelly_fraction, "recommandation": rec, "raison": f"Edge de {edge*100:.1f}%"}


def simuler_evolution_bankroll(bankroll_initial, resultats):
    capital=bankroll_initial; evolution=[capital]
    for r in resultats:
        capital += r["mise"]*(r["cote"]-1) if r["gagne"] else -r["mise"]
        evolution.append(capital)
    gain=capital-bankroll_initial
    return {"capital_final":round(capital,2),"gain_total":round(gain,2),"roi_reel":round(gain/bankroll_initial*100,2),"evolution":evolution}

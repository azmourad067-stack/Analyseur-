from src.core.bankroll import calculer_mise_optimale
from src.core.intelligence import calculer_prediction

def test_bankroll_skip():
    assert calculer_mise_optimale(2, 30, 1000)["mise"] == 0

def test_prediction_range():
    p={"musique":"1a2a3a","age":5,"gains":50000,"driver":"BAZIRE J.M.","ferrage":"D4","oeilleres":"SANS_OEILLERES","nb_courses":30,"nb_victoires":8,"nb_places":10,"cote_ref":4}
    score=calculer_prediction(p,{"discipline":"ATTELE","prixCourse":30000})
    assert 0 <= score <= 100

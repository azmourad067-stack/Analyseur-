from datetime import date, datetime
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE = "https://online.turfinfo.api.pmu.fr/rest/client/61"
HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "fr-FR,fr;q=0.9,en;q=0.8",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
    "Referer": "https://www.pmu.fr/turf/",
    "User-Agent": "Mozilla/5.0 (compatible; PMU-Streamlit/1.0)",
}


def _session():
    s = requests.Session()
    retry = Retry(total=3, backoff_factor=1, status_forcelist=(429, 500, 502, 503, 504), allowed_methods=("GET",))
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.headers.update(HEADERS)
    return s


def fetch_api(url: str):
    r = _session().get(url, timeout=30)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.json()


def _date_ddmmyyyy(day):
    if isinstance(day, str):
        day = datetime.fromisoformat(day).date()
    return day.strftime("%d%m%Y")


def fetch_course_participants(day, reunion, course):
    d = _date_ddmmyyyy(day)
    return fetch_api(f"{BASE}/programme/{d}/R{reunion}/C{course}/participants?specialisation=INTERNET")


def fetch_course_rapports(day, reunion, course):
    d = _date_ddmmyyyy(day)
    return fetch_api(f"{BASE}/programme/{d}/R{reunion}/C{course}/rapports?specialisation=INTERNET")


def fetch_day(day, config=None):
    d = _date_ddmmyyyy(day)
    data = fetch_api(f"{BASE}/programme/{d}?meteo=true&specialisation=INTERNET")
    if not data or not data.get("programme", {}).get("reunions"):
        return data
    for reunion in data["programme"]["reunions"]:
        for course in reunion.get("courses", []) or []:
            details = fetch_course_participants(day, reunion.get("numOfficiel"), course.get("numOrdre"))
            if details and details.get("participants"):
                course["participants"] = details["participants"]
            if course.get("statut") in {"ARRIVEE_DEFINITIVE_COMPLETE", "ARRIVEE"}:
                reports = fetch_course_rapports(day, reunion.get("numOfficiel"), course.get("numOrdre"))
                if reports:
                    course["rapportsDefinitifs"] = reports
    return data

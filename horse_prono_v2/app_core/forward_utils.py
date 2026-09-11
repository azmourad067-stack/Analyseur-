from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd


PARIS_TZ = ZoneInfo("Europe/Paris")


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _find_reunions(obj: Any) -> list[dict]:
    if isinstance(obj, dict):
        reunions = obj.get("reunions")
        if isinstance(reunions, list):
            return [x for x in reunions if isinstance(x, dict)]

        for value in obj.values():
            found = _find_reunions(value)
            if found:
                return found

    elif isinstance(obj, list):
        for value in obj:
            found = _find_reunions(value)
            if found:
                return found

    return []


def course_objects(programme: Any) -> dict[tuple[int, int], dict]:
    if not isinstance(programme, dict):
        return {}

    root = programme.get("programme")
    if not isinstance(root, dict):
        root = programme

    mapping: dict[tuple[int, int], dict] = {}

    for reunion_obj in _find_reunions(root):
        reunion = _as_int(
            reunion_obj.get("numOfficiel")
            or reunion_obj.get("numReunion")
            or reunion_obj.get("numReunionProgramme")
            or reunion_obj.get("numero")
        )
        if reunion is None:
            continue

        courses = reunion_obj.get("courses")
        if not isinstance(courses, list):
            continue

        for course_obj in courses:
            if not isinstance(course_obj, dict):
                continue

            course = _as_int(
                course_obj.get("numOrdre")
                or course_obj.get("numCourse")
                or course_obj.get("numOfficiel")
                or course_obj.get("numero")
            )
            if course is None:
                continue

            mapping[(reunion, course)] = course_obj

    return mapping


def _from_epoch(value: float) -> datetime | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None

    if numeric <= 0:
        return None

    if numeric >= 10**12:
        return datetime.fromtimestamp(
            numeric / 1000.0,
            tz=timezone.utc,
        )

    if numeric >= 10**9:
        return datetime.fromtimestamp(
            numeric,
            tz=timezone.utc,
        )

    return None


def parse_start_time(value: Any, race_date: date) -> datetime | None:
    if value is None:
        return None

    if isinstance(value, (int, float)):
        return _from_epoch(value)

    text = str(value).strip()
    if not text:
        return None

    if text.isdigit():
        parsed = _from_epoch(float(text))
        if parsed is not None:
            return parsed

    for fmt in ("%H:%M:%S", "%H:%M"):
        try:
            local_time = datetime.strptime(text, fmt).time()
            local_dt = datetime.combine(
                race_date,
                local_time,
                tzinfo=PARIS_TZ,
            )
            return local_dt.astimezone(timezone.utc)
        except ValueError:
            pass

    try:
        ts = pd.to_datetime(text, errors="raise", utc=False)
    except Exception:
        return None

    if isinstance(ts, pd.Timestamp):
        if ts.tzinfo is None:
            ts = ts.tz_localize(PARIS_TZ)
        return ts.tz_convert("UTC").to_pydatetime()

    return None


def course_start_time(
    course_obj: dict[str, Any] | None,
    race_date: date,
) -> datetime | None:
    if not course_obj:
        return None

    candidates = [
        course_obj.get("heureDepart"),
        course_obj.get("dateHeureDepart"),
        course_obj.get("heureDepartCourse"),
        course_obj.get("dateDepart"),
        course_obj.get("startTime"),
        course_obj.get("start_time"),
    ]

    for candidate in candidates:
        parsed = parse_start_time(candidate, race_date)
        if parsed is not None:
            return parsed

    return None


def course_status(course_obj: dict[str, Any] | None) -> str:
    if not course_obj:
        return ""

    value = (
        course_obj.get("statut")
        or course_obj.get("statutCourse")
        or course_obj.get("status")
        or course_obj.get("etat")
        or ""
    )

    if isinstance(value, dict):
        value = (
            value.get("libelle")
            or value.get("code")
            or value.get("value")
            or ""
        )

    return str(value).strip().upper()


def _iter_dicts(obj: Any):
    if isinstance(obj, dict):
        yield obj
        for value in obj.values():
            yield from _iter_dicts(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from _iter_dicts(value)


def non_runner_numbers(participant_payload: Any) -> set[int]:
    result: set[int] = set()

    for obj in _iter_dicts(participant_payload):
        number = _as_int(
            obj.get("numPmu")
            or obj.get("numero")
            or obj.get("numParticipant")
        )

        if number is None:
            continue

        explicit = (
            obj.get("nonPartant")
            or obj.get("isNonRunner")
            or obj.get("non_runner")
        )

        status = (
            obj.get("statut")
            or obj.get("statutParticipant")
            or obj.get("status")
            or ""
        )

        if isinstance(status, dict):
            status = (
                status.get("libelle")
                or status.get("code")
                or status.get("value")
                or ""
            )

        status_text = (
            str(status)
            .strip()
            .upper()
            .replace("-", "_")
        )

        is_np = bool(explicit)

        if any(
            token in status_text
            for token in (
                "NON_PARTANT",
                "NON PARTANT",
                "NONPARTANT",
                "SCRATCH",
                "WITHDRAWN",
            )
        ):
            is_np = True

        if is_np:
            result.add(number)

    return result

from __future__ import annotations

import os
import shutil
import zipfile
from pathlib import Path

import pandas as pd
from supabase import create_client


# ============================================================
# HORSEPRONO - EXPORT COMPLET HISTORIQUE SUPABASE
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "supabase_history"
)

ZIP_PATH = (
    OUTPUT_DIR
    / "horseprono_supabase_history.zip"
)

PAGE_SIZE = 1000


# ============================================================
# SUPABASE
# ============================================================

def get_client():

    url = os.getenv(
        "SUPABASE_URL"
    )

    key = (
        os.getenv(
            "SUPABASE_SERVICE_KEY"
        )
        or os.getenv(
            "SUPABASE_KEY"
        )
    )

    if not url or not key:

        raise RuntimeError(
            "SUPABASE_URL / "
            "SUPABASE_SERVICE_KEY absents."
        )

    return create_client(
        url,
        key,
    )


# ============================================================
# CHARGEMENT PAGINE D'UNE TABLE
# ============================================================

def load_table(
    client,
    table_name: str,
    order_column: str = "id",
):

    rows = []
    offset = 0

    print()
    print(
        f"Chargement de {table_name}..."
    )

    while True:

        response = (
            client
            .table(
                table_name
            )
            .select(
                "*"
            )
            .order(
                order_column
            )
            .range(
                offset,
                offset + PAGE_SIZE - 1,
            )
            .execute()
        )

        batch = (
            getattr(
                response,
                "data",
                None,
            )
            or []
        )

        if not batch:
            break

        rows.extend(
            batch
        )

        print(
            f"{table_name} : "
            f"{len(rows)} lignes chargées"
        )

        if len(batch) < PAGE_SIZE:
            break

        offset += PAGE_SIZE

    return pd.DataFrame(
        rows
    )


# ============================================================
# EXPORT CSV
# ============================================================

def export_csv(
    dataframe: pd.DataFrame,
    filename: str,
):

    path = (
        OUTPUT_DIR
        / filename
    )

    dataframe.to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"✅ {filename} : "
        f"{len(dataframe)} lignes"
    )

    return path


# ============================================================
# HISTORIQUE COMPLET FUSIONNE
# ============================================================

def build_full_history(
    races: pd.DataFrame,
    participants: pd.DataFrame,
):

    if races.empty:
        raise RuntimeError(
            "La table races est vide."
        )

    if participants.empty:
        raise RuntimeError(
            "La table participants est vide."
        )

    race_columns = {
        "id":
            "race_id",

        "external_id":
            "race_external_id",

        "created_at":
            "race_created_at",

        "updated_at":
            "race_updated_at",
    }

    participant_columns = {
        "id":
            "participant_id",

        "external_id":
            "participant_external_id",

        "created_at":
            "participant_created_at",
    }

    races_renamed = (
        races.rename(
            columns=race_columns
        )
    )

    participants_renamed = (
        participants.rename(
            columns=participant_columns
        )
    )

    history = (
        participants_renamed.merge(
            races_renamed,
            on="race_id",
            how="left",
            suffixes=(
                "_participant",
                "_race",
            ),
        )
    )

    preferred_columns = [
        "race_id",
        "race_external_id",
        "race_date",
        "meeting_number",
        "race_number",
        "hippodrome",
        "discipline",
        "distance_m",
        "terrain",
        "field_size",
        "status",

        "participant_id",
        "participant_external_id",
        "horse_name",
        "horse_number",
        "jockey_name",
        "trainer_name",
        "odds",
        "weight_kg",
        "draw",
        "age",
        "sex",
        "recent_form",
        "finish_position",
        "is_non_runner",

        "race_created_at",
        "race_updated_at",
        "participant_created_at",
    ]

    existing_preferred = [
        column
        for column
        in preferred_columns
        if column
        in history.columns
    ]

    remaining = [
        column
        for column
        in history.columns
        if column
        not in existing_preferred
    ]

    history = history[
        existing_preferred
        + remaining
    ]

    sort_columns = [
        column
        for column
        in [
            "race_date",
            "meeting_number",
            "race_number",
            "horse_number",
        ]
        if column
        in history.columns
    ]

    if sort_columns:

        history = (
            history.sort_values(
                sort_columns,
                na_position="last",
            )
        )

    return history


# ============================================================
# MANIFESTE
# ============================================================

def create_manifest(
    races: pd.DataFrame,
    participants: pd.DataFrame,
    history: pd.DataFrame,
):

    path = (
        OUTPUT_DIR
        / "README_EXPORT.txt"
    )

    first_date = ""
    last_date = ""

    if (
        "race_date"
        in races.columns
        and not races.empty
    ):

        dates = pd.to_datetime(
            races[
                "race_date"
            ],
            errors="coerce",
        )

        if dates.notna().any():

            first_date = (
                dates.min()
                .date()
                .isoformat()
            )

            last_date = (
                dates.max()
                .date()
                .isoformat()
            )

    content = f"""
HORSEPRONO - EXPORT SUPABASE

Courses :
{len(races)}

Participants :
{len(participants)}

Lignes historique_complet :
{len(history)}

Première date :
{first_date}

Dernière date :
{last_date}

FICHIERS

races.csv
    Une ligne par course.

participants.csv
    Une ligne par cheval / participant.

historique_complet.csv
    Une ligne par cheval avec les informations
    de sa course fusionnées.

Encodage CSV :
UTF-8 avec BOM, compatible Excel.
""".strip()

    path.write_text(
        content,
        encoding="utf-8",
    )

    return path


# ============================================================
# ZIP
# ============================================================

def create_zip(
    files: list[Path],
):

    if ZIP_PATH.exists():

        ZIP_PATH.unlink()

    with zipfile.ZipFile(
        ZIP_PATH,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:

        for file_path in files:

            archive.write(
                file_path,
                arcname=file_path.name,
            )

    size_mb = (
        ZIP_PATH.stat().st_size
        / 1024
        / 1024
    )

    print()
    print(
        f"✅ ZIP créé : "
        f"{ZIP_PATH}"
    )

    print(
        f"Taille : "
        f"{size_mb:.2f} Mo"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print(
        "=" * 72
    )

    print(
        "HORSEPRONO - EXPORT COMPLET SUPABASE"
    )

    print(
        "=" * 72
    )

    if OUTPUT_DIR.exists():

        shutil.rmtree(
            OUTPUT_DIR
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    client = get_client()

    # --------------------------------------------------------
    # TABLE RACES
    # --------------------------------------------------------

    races = load_table(
        client,
        "races",
    )

    # --------------------------------------------------------
    # TABLE PARTICIPANTS
    # --------------------------------------------------------

    participants = load_table(
        client,
        "participants",
    )

    print()
    print(
        "=" * 72
    )

    print(
        "CREATION HISTORIQUE FUSIONNE"
    )

    print(
        "=" * 72
    )

    history = build_full_history(
        races,
        participants,
    )

    # --------------------------------------------------------
    # EXPORTS
    # --------------------------------------------------------

    races_path = export_csv(
        races,
        "races.csv",
    )

    participants_path = export_csv(
        participants,
        "participants.csv",
    )

    history_path = export_csv(
        history,
        "historique_complet.csv",
    )

    manifest_path = create_manifest(
        races,
        participants,
        history,
    )

    # --------------------------------------------------------
    # ZIP
    # --------------------------------------------------------

    create_zip(
        [
            races_path,
            participants_path,
            history_path,
            manifest_path,
        ]
    )

    print()
    print(
        "=" * 72
    )

    print(
        "EXPORT TERMINE"
    )

    print(
        "=" * 72
    )

    print(
        f"Courses : "
        f"{len(races)}"
    )

    print(
        f"Participants : "
        f"{len(participants)}"
    )

    print(
        f"Historique complet : "
        f"{len(history)}"
    )

    print()
    print(
        ZIP_PATH
    )


if __name__ == "__main__":

    main()

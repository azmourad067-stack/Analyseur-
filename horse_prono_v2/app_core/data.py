from __future__ import annotations

from io import BytesIO

import numpy as np
import pandas as pd

from .config import HISTORY_COLUMNS


def _num(
    series: pd.Series,
) -> pd.Series:

    return pd.to_numeric(
        series,
        errors="coerce",
    )


def normalize_history(
    df: pd.DataFrame,
) -> pd.DataFrame:

    out = df.copy()

    aliases = {
        "date":
            "race_date",

        "date_course":
            "race_date",

        "cheval":
            "horse_name",

        "n°":
            "horse_number",

        "numero":
            "horse_number",

        "cote":
            "odds",

        "jockey_name":
            "jockey",

        "entraineur":
            "trainer",

        "position":
            "finish_position",

        "classement":
            "finish_position",

        "corde":
            "draw",

        "sexe":
            "sex",

        "age_cheval":
            "age",
    }

    out.columns = [
        str(column).strip()
        for column in out.columns
    ]

    for src, dst in aliases.items():

        if (
            src in out.columns
            and dst not in out.columns
        ):
            out[
                dst
            ] = out[
                src
            ]

    # --------------------------------------------------------
    # Colonnes métier obligatoires
    # --------------------------------------------------------

    for column in HISTORY_COLUMNS:

        if column not in out.columns:

            out[
                column
            ] = None

    # --------------------------------------------------------
    # Date
    # --------------------------------------------------------

    out[
        "race_date"
    ] = pd.to_datetime(
        out[
            "race_date"
        ],
        errors="coerce",
    )

    # --------------------------------------------------------
    # Entiers / numériques
    # --------------------------------------------------------

    for column in [
        "distance",
        "field_size",
        "horse_number",
        "draw",
        "age",
        "finish_position",
    ]:

        out[
            column
        ] = _num(
            out[
                column
            ]
        )

    for column in [
        "odds",
        "weight",
        "career_runs",
        "career_wins",
        "career_places",
    ]:

        if column not in out.columns:

            out[
                column
            ] = np.nan

        out[
            column
        ] = _num(
            out[
                column
            ]
        )

    # --------------------------------------------------------
    # Statistiques carrière facultatives
    # --------------------------------------------------------

    if (
        "career_win_rate"
        not in out.columns
    ):

        out[
            "career_win_rate"
        ] = np.where(
            out[
                "career_runs"
            ] > 0,

            out[
                "career_wins"
            ]
            / out[
                "career_runs"
            ],

            np.nan,
        )

    if (
        "career_place_rate"
        not in out.columns
    ):

        out[
            "career_place_rate"
        ] = np.where(
            out[
                "career_runs"
            ] > 0,

            out[
                "career_places"
            ]
            / out[
                "career_runs"
            ],

            np.nan,
        )

    # --------------------------------------------------------
    # Identifiants / textes
    # --------------------------------------------------------

    out[
        "race_id"
    ] = out[
        "race_id"
    ].astype(
        str
    )

    out[
        "horse_name"
    ] = (
        out[
            "horse_name"
        ]
        .fillna(
            "Inconnu"
        )
        .astype(
            str
        )
    )

    out[
        "jockey"
    ] = (
        out[
            "jockey"
        ]
        .fillna(
            ""
        )
        .astype(
            str
        )
    )

    out[
        "trainer"
    ] = (
        out[
            "trainer"
        ]
        .fillna(
            ""
        )
        .astype(
            str
        )
    )

    out[
        "discipline"
    ] = (
        out[
            "discipline"
        ]
        .fillna(
            "INCONNU"
        )
        .astype(
            str
        )
    )

    # --------------------------------------------------------
    # Sexe
    # --------------------------------------------------------

    out[
        "sex"
    ] = (
        out[
            "sex"
        ]
        .fillna(
            ""
        )
        .astype(
            str
        )
        .str.strip()
        .str.upper()
    )

    # --------------------------------------------------------
    # Taille du peloton
    # --------------------------------------------------------

    derived_field_size = (
        out
        .groupby(
            "race_id"
        )[
            "horse_number"
        ]
        .transform(
            "count"
        )
    )

    if (
        out[
            "field_size"
        ]
        .isna()
        .all()
    ):

        out[
            "field_size"
        ] = derived_field_size

    else:

        out[
            "field_size"
        ] = (
            out[
                "field_size"
            ]
            .fillna(
                derived_field_size
            )
        )

    # --------------------------------------------------------
    # Musique -> score simple
    # --------------------------------------------------------

    def form_score(
        value,
    ):

        if pd.isna(
            value
        ):
            return np.nan

        nums = [
            int(x)
            for x in __import__(
                "re"
            ).findall(
                r"\d+",
                str(value),
            )[:5]
        ]

        if not nums:
            return np.nan

        scores = [
            max(
                0.0,
                1.0
                - (
                    n - 1
                )
                / 9.0,
            )
            for n in nums
        ]

        return float(
            np.mean(
                scores
            )
        )

    out[
        "recent_form_score"
    ] = out[
        "recent_form"
    ].map(
        form_score
    )

    return out


def normalize_race_input(
    df: pd.DataFrame,
) -> pd.DataFrame:

    out = normalize_history(
        df
    )

    if (
        out[
            "race_id"
        ]
        .eq(
            "nan"
        )
        .all()
        or
        out[
            "race_id"
        ]
        .eq(
            ""
        )
        .all()
    ):

        out[
            "race_id"
        ] = "manual-race"

    out[
        "race_id"
    ] = out[
        "race_id"
    ].replace(
        {
            "nan":
                "manual-race",

            "None":
                "manual-race",
        }
    )

    out[
        "race_date"
    ] = out[
        "race_date"
    ].fillna(
        pd.Timestamp
        .today()
        .normalize()
    )

    return out


def read_uploaded_csv(
    uploaded,
) -> pd.DataFrame:

    raw = (
        uploaded.getvalue()
        if hasattr(
            uploaded,
            "getvalue",
        )
        else uploaded.read()
    )

    return normalize_history(
        pd.read_csv(
            BytesIO(
                raw
            )
        )
    )


def example_race() -> pd.DataFrame:

    rows = [

        [
            "demo-1",
            pd.Timestamp.today(),
            "PLAT",
            "ParisLongchamp",
            2100,
            "Bon",
            12,
            1,
            "Cheval Alpha",
            "Jockey A",
            "Entraîneur A",
            4.5,
            2,
            57,
            4,
            "MALES",
            "1a 2a 4a",
            None,
        ],

        [
            "demo-1",
            pd.Timestamp.today(),
            "PLAT",
            "ParisLongchamp",
            2100,
            "Bon",
            12,
            2,
            "Cheval Bravo",
            "Jockey B",
            "Entraîneur B",
            6.2,
            7,
            58,
            5,
            "HONGRES",
            "2a 5a 1a",
            None,
        ],

        [
            "demo-1",
            pd.Timestamp.today(),
            "PLAT",
            "ParisLongchamp",
            2100,
            "Bon",
            12,
            3,
            "Cheval Charlie",
            "Jockey C",
            "Entraîneur C",
            9.8,
            4,
            56,
            4,
            "FEMELLES",
            "3a 3a 6a",
            None,
        ],

        [
            "demo-1",
            pd.Timestamp.today(),
            "PLAT",
            "ParisLongchamp",
            2100,
            "Bon",
            12,
            4,
            "Cheval Delta",
            "Jockey D",
            "Entraîneur D",
            12.5,
            9,
            59,
            6,
            "HONGRES",
            "5a 1a 2a",
            None,
        ],

        [
            "demo-1",
            pd.Timestamp.today(),
            "PLAT",
            "ParisLongchamp",
            2100,
            "Bon",
            12,
            5,
            "Cheval Echo",
            "Jockey E",
            "Entraîneur E",
            18.0,
            1,
            55,
            3,
            "FEMELLES",
            "7a 4a 3a",
            None,
        ],

        [
            "demo-1",
            pd.Timestamp.today(),
            "PLAT",
            "ParisLongchamp",
            2100,
            "Bon",
            12,
            6,
            "Cheval Foxtrot",
            "Jockey F",
            "Entraîneur F",
            25.0,
            6,
            57,
            7,
            "HONGRES",
            "6a 8a 4a",
            None,
        ],
    ]

    return pd.DataFrame(
        rows,
        columns=HISTORY_COLUMNS,
    )

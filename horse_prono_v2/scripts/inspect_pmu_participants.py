from __future__ import annotations

import os
import sys
from datetime import date

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(__file__)
    ),
)

from app_core.pmu import get_participants


RACES = [
    ("PLAT", date(2026, 9, 8), 3, 1),
    ("ATTELE_AUTOSTART", date(2026, 9, 8), 2, 1),
    ("ATTELE_VOLTE", date(2026, 9, 8), 4, 1),
    ("TROT_MONTE", date(2026, 9, 8), 4, 5),
]


def find_participants(obj):
    if isinstance(obj, dict):

        if isinstance(
            obj.get("participants"),
            list,
        ):
            return obj["participants"]

        for value in obj.values():

            result = find_participants(
                value
            )

            if result is not None:
                return result

    elif isinstance(obj, list):

        for value in obj:

            result = find_participants(
                value
            )

            if result is not None:
                return result

    return None


def interesting_fields(participant):
    tokens = (
        "poids",
        "handicap",
        "age",
        "sexe",
        "corde",
    )

    return {
        key: value
        for key, value
        in participant.items()
        if any(
            token in key.lower()
            for token in tokens
        )
    }


def main():

    for (
        discipline,
        race_date,
        reunion,
        course,
    ) in RACES:

        print()
        print("=" * 70)

        print(
            discipline,
            race_date,
            f"R{reunion}C{course}",
        )

        print("=" * 70)

        payload = get_participants(
            race_date,
            reunion,
            course,
        )

        participants = (
            find_participants(
                payload
            )
            or []
        )

        print(
            f"Participants trouvés : "
            f"{len(participants)}"
        )

        for i, participant in enumerate(
            participants[:3],
            start=1,
        ):

            print()
            print(
                f"Participant #{i}"
            )

            print(
                "Nom :",
                participant.get("nom"),
            )

            print(
                "N° :",
                participant.get("numPmu"),
            )

            fields = (
                interesting_fields(
                    participant
                )
            )

            for key, value in fields.items():

                print(
                    f"  {key} = "
                    f"{value!r} "
                    f"({type(value).__name__})"
                )


if __name__ == "__main__":
    main()

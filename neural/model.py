from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


# ============================================================
# HORSEPRONO NEURAL V1
# Réseau neuronal tabulaire avec embeddings
# ============================================================


def get_embedding_dim(cardinality: int) -> int:
    """
    Taille raisonnable d'embedding selon le nombre
    de catégories différentes.

    Exemples :
        sexe          -> petit embedding
        discipline    -> petit embedding
        hippodrome    -> moyen embedding
        jockey        -> gros embedding
        entraîneur    -> gros embedding
    """

    if cardinality <= 4:
        return 2

    if cardinality <= 20:
        return 4

    if cardinality <= 100:
        return 8

    if cardinality <= 500:
        return 16

    if cardinality <= 2000:
        return 24

    return 32


class HorsePronoNeuralV1(nn.Module):
    """
    Réseau neuronal tabulaire HorseProno Neural V1.

    Entrées :
        numeric_x :
            variables numériques normalisées

        categorical_x :
            indices des catégories :
            sexe
            hippodrome
            discipline
            terrain
            jockey
            entraîneur

    Sortie :
        un logit par cheval

    Le sigmoid sera appliqué seulement
    lors du calcul des probabilités.
    """

    def __init__(
        self,
        num_numeric_features: int,
        categorical_cardinalities: Dict[str, int],
        hidden_1: int = 128,
        hidden_2: int = 64,
        hidden_3: int = 32,
        dropout_1: float = 0.25,
        dropout_2: float = 0.20,
    ):

        super().__init__()

        self.category_names = list(
            categorical_cardinalities.keys()
        )

        # ----------------------------------------------------
        # Embeddings
        # ----------------------------------------------------

        self.embeddings = nn.ModuleDict()

        total_embedding_dim = 0

        for (
            column_name,
            cardinality,
        ) in categorical_cardinalities.items():

            embedding_dim = get_embedding_dim(
                cardinality
            )

            # +1 réservé à UNKNOWN
            self.embeddings[column_name] = (
                nn.Embedding(
                    num_embeddings=cardinality + 1,
                    embedding_dim=embedding_dim,
                    padding_idx=0,
                )
            )

            total_embedding_dim += embedding_dim

        # ----------------------------------------------------
        # Taille totale entrée réseau
        # ----------------------------------------------------

        input_dim = (
            num_numeric_features
            + total_embedding_dim
        )

        # ----------------------------------------------------
        # Réseau principal
        # ----------------------------------------------------

        self.network = nn.Sequential(

            nn.Linear(
                input_dim,
                hidden_1,
            ),

            nn.BatchNorm1d(
                hidden_1
            ),

            nn.ReLU(),

            nn.Dropout(
                dropout_1
            ),

            # -----------------------------------------------

            nn.Linear(
                hidden_1,
                hidden_2,
            ),

            nn.BatchNorm1d(
                hidden_2
            ),

            nn.ReLU(),

            nn.Dropout(
                dropout_2
            ),

            # -----------------------------------------------

            nn.Linear(
                hidden_2,
                hidden_3,
            ),

            nn.ReLU(),

            # -----------------------------------------------

            nn.Linear(
                hidden_3,
                1,
            ),
        )

        self._initialize_weights()

    # ========================================================
    # INITIALISATION
    # ========================================================

    def _initialize_weights(self):

        for module in self.modules():

            if isinstance(
                module,
                nn.Linear,
            ):

                nn.init.kaiming_normal_(
                    module.weight,
                    nonlinearity="relu",
                )

                if module.bias is not None:

                    nn.init.zeros_(
                        module.bias
                    )

    # ========================================================
    # FORWARD
    # ========================================================

    def forward(
        self,
        numeric_x: torch.Tensor,
        categorical_x: torch.Tensor,
    ) -> torch.Tensor:

        embedding_outputs = []

        for index, column_name in enumerate(
            self.category_names
        ):

            category_indices = (
                categorical_x[:, index]
            )

            embedded = (
                self.embeddings[
                    column_name
                ](
                    category_indices
                )
            )

            embedding_outputs.append(
                embedded
            )

        # ----------------------------------------------------
        # Numériques + embeddings
        # ----------------------------------------------------

        if embedding_outputs:

            categorical_embeddings = (
                torch.cat(
                    embedding_outputs,
                    dim=1,
                )
            )

            x = torch.cat(
                [
                    numeric_x,
                    categorical_embeddings,
                ],
                dim=1,
            )

        else:

            x = numeric_x

        # ----------------------------------------------------
        # Logit P(Top3)
        # ----------------------------------------------------

        logits = self.network(x)

        return logits.squeeze(1)


# ============================================================
# OUTILS
# ============================================================

def count_parameters(
    model: nn.Module,
) -> int:

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def print_model_summary(
    model: HorsePronoNeuralV1,
):

    print("\n")
    print("=" * 70)

    print(
        "HORSEPRONO NEURAL V1 - ARCHITECTURE"
    )

    print("=" * 70)

    print(model)

    print(
        "\nParamètres entraînables : "
        f"{count_parameters(model):,}"
    )

    print("=" * 70)

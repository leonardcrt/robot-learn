"""Réseaux PyTorch partagés par le behavior cloning et PPO.

Architecture retenue : `NavEncoder` (encodeur « DeepSets », Zaheer et al., 2017)
---------------------------------------------------------------------------------

    cible (3) ───────────── MLP 3→64→64 ─────────────┐
                                                     ├─ concat (128) → Linear 128→128 → ReLU → features
    obstacles (K×5) → φ partagé 4→64→64 → max-pool ──┘       (masque : emplacements vides ignorés)

Pourquoi ce choix plutôt qu'un MLP sur le vecteur aplati :

1. **Les obstacles forment un ensemble, pas une liste.** L'observation les trie par
   distance ; quand deux obstacles échangent leur rang, un MLP aplati voit ses
   entrées permuter brutalement et sa sortie saute, alors que la scène physique n'a
   pas changé. Un encodeur par obstacle *partagé* suivi d'un pooling symétrique est
   invariant par permutation par construction : ces discontinuités disparaissent.
2. **Partage de poids.** Le même φ traite chaque obstacle : il apprend « ce qu'est un
   obstacle dangereux » une seule fois au lieu de K fois (une par position dans le
   vecteur), ce qui économise des paramètres et des données.
3. **Max-pooling plutôt que moyenne.** Pour éviter une collision, c'est l'obstacle le
   plus critique qui compte ; le max permet à chaque neurone de répondre à « l'obstacle
   le plus X » sans être dilué par des obstacles lointains ou par des emplacements vides.
   Le masque ignore les emplacements vides, donc le nombre d'obstacles visibles peut varier.
4. **Tête de sortie linéaire (puis écrêtage à [-1, 1]) plutôt que tanh.** L'expert roule
   souvent à vitesse maximale (a = +1) ; avec tanh, atteindre exactement ±1 demande une
   pré-activation infinie et le gradient s'écrase dans la zone saturée.
5. **Taille modeste (≈ 30 k paramètres).** Le problème est de faible dimension ; un
   réseau plus gros sur-apprend les démonstrations et ralentit PPO sur CPU.

`FlatMLPEncoder` sert de référence pour l'ablation (même nombre de couches, entrée aplatie).
"""

from __future__ import annotations

import torch
from torch import nn

GOAL_DIM = 3
OBSTACLE_DIM = 5


def mlp(sizes: list[int], activation=nn.ReLU, last_activation: bool = True) -> nn.Sequential:
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2 or last_activation:
            layers.append(activation())
    return nn.Sequential(*layers)


class NavEncoder(nn.Module):
    """Encodeur invariant par permutation des obstacles (DeepSets + max-pooling masqué)."""

    def __init__(self, k_nearest: int = 5, hidden: int = 64, features_dim: int = 128):
        super().__init__()
        self.k = k_nearest
        self.features_dim = features_dim
        self.goal_net = mlp([GOAL_DIM, hidden, hidden])
        self.obstacle_net = mlp([OBSTACLE_DIM - 1, hidden, hidden])  # le masque n'est pas une entrée
        self.trunk = mlp([2 * hidden, features_dim])

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        goal = obs[:, :GOAL_DIM]
        obstacles = obs[:, GOAL_DIM:].reshape(-1, self.k, OBSTACLE_DIM)
        mask = obstacles[..., 4:5] > 0.5  # (B, K, 1)

        h = self.obstacle_net(obstacles[..., :4])  # (B, K, hidden)
        h = h.masked_fill(~mask, float("-inf")).amax(dim=1)
        h = torch.where(torch.isinf(h), torch.zeros_like(h), h)  # aucun obstacle visible -> 0

        return self.trunk(torch.cat([self.goal_net(goal), h], dim=1))


class FlatMLPEncoder(nn.Module):
    """Référence pour l'ablation : MLP sur le vecteur d'observation aplati."""

    def __init__(self, obs_dim: int, hidden: int = 128, features_dim: int = 128):
        super().__init__()
        self.features_dim = features_dim
        self.net = mlp([obs_dim, hidden, features_dim])

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


def build_encoder(kind: str, obs_dim: int, k_nearest: int, **kwargs) -> nn.Module:
    if kind == "deepsets":
        return NavEncoder(k_nearest=k_nearest, **kwargs)
    if kind == "mlp":
        return FlatMLPEncoder(obs_dim=obs_dim, **kwargs)
    raise ValueError(f"encodeur inconnu : {kind}")


class Actor(nn.Module):
    """Politique déterministe : encodeur -> tête MLP -> action dans [-1, 1]^2."""

    def __init__(self, encoder: nn.Module, head_hidden: tuple[int, ...] = (64,), action_dim: int = 2):
        super().__init__()
        self.encoder = encoder
        self.head = mlp([encoder.features_dim, *head_hidden, action_dim], last_activation=False)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Sortie non bornée (utilisée pour la perte) ; `act` écrête dans [-1, 1]."""
        return self.head(self.encoder(obs))

    @torch.no_grad()
    def act(self, obs: torch.Tensor) -> torch.Tensor:
        return self.forward(obs).clamp(-1.0, 1.0)


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())

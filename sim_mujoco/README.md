# Extension MuJoCo

Le projet principal utilise un simulateur **cinématique** : le robot obéit instantanément, ne dérape jamais, et une
collision est un simple test de distance. Cette extension refait la même tâche avec la **physique de corps rigides** de
[MuJoCo](https://mujoco.org/) (masse, inertie, forces bornées, contacts). Deux questions :

1. Que perdent l'expert, le BC, DAgger et PPO quand on change la dynamique **sans les réentraîner** ?
2. Un PPO entraîné **dans** MuJoCo, avec le même réseau, retrouve-t-il le niveau ?

Rien d'existant ne change, sauf trois ajouts minimaux et rétrocompatibles : l'extra `[mujoco]` (et le paquet
`sim_mujoco`) dans `pyproject.toml`, un paramètre optionnel `env_factory` dans `eval/metrics.py::evaluate_policy`, et un
paragraphe dans le `README.md`.

## Le modèle physique

- **Robot** : cylindre de rayon 0,2 m et de masse 5 kg, avec trois articulations (x, y, lacet) : il reste dans le plan.
  **Obstacles** : cylindres statiques, mêmes positions et mêmes rayons que dans le simulateur cinématique. Le XML MuJoCo
  est régénéré à chaque épisode (`robot_model.py`).
- **Commande** : la politique renvoie la même commande (v, ω) qu'avant. À chaque sous-pas de physique (0,01 s, dix par pas
  de contrôle de 0,1 s), un contrôleur de vitesse à force limitée applique une force le long du cap (proportionnelle à
  l'écart de vitesse, bornée), une force latérale qui s'oppose au glissement (amortissement fort, équivalent d'un
  roulement sans glissement) et un couple de rotation borné (`mujoco_env.py`).
- **Collision** : lue dans les contacts MuJoCo, testée après chaque sous-pas.
- **Identique au simulateur cinématique** : l'observation (`env/observation.py`), la récompense, les conditions de fin,
  les cartes et les graines. Les checkpoints existants se branchent donc sans modification.

| Paramètre | Valeur | Effet |
|---|---|---|
| gain de vitesse avant $k_v$ | 33 N·s/m | constante de temps 0,15 s |
| force maximale | 20 N | accélération maximale 4 m/s² |
| gain de rotation $k_\omega$ | 1 N·m·s/rad | constante de temps 0,1 s |
| couple maximal | 1 N·m | accélération angulaire maximale 10 rad/s² |
| amortissement latéral | 200 N·s/m | dérapage de 0,05 m/s dans un virage à 1 m/s et 2 rad/s |

Ces valeurs sont des **choix de conception**, pas des mesures sur un robot réel : les écarts mesurés ci-dessous en
dépendent.

## Fichiers

| Fichier | Rôle |
|---|---|
| `ant_demo.py` | prise en main : PPO sur `Ant-v5` (500 000 pas) et une vidéo |
| `robot_model.py` | une carte devient un XML MuJoCo ; paramètres du robot |
| `mujoco_env.py` | l'environnement Gymnasium `MuJoCoNavEnv` (sous-classe de `NavigationEnv`) |
| `run_expert.py` | l'expert VFH dans les deux simulateurs, sur les mêmes cartes |
| `evaluate_mujoco.py` | benchmark dans MuJoCo (500 cartes de test), options de sensibilité `--k-v` et `--k-w` |
| `train_ppo.py` | PPO dans MuJoCo, même réseau et mêmes hyperparamètres que `rl/train.py` |
| `render_compare.py` | GIF : la même politique dans les deux simulateurs |

Sorties : `checkpoints/mujoco/ppo_mujoco.pt` et `results/mujoco/`. Tests : `tests/test_mujoco.py` (ignorés si MuJoCo
n'est pas installé).

## Tout refaire

```bash
pip install -e ".[mujoco]"                          # 1. installer MuJoCo
python -m sim_mujoco.evaluate_mujoco --episodes 500 # 2. évaluer les checkpoints existants, sans réentraînement
python -m sim_mujoco.train_ppo --subproc            # 3. entraîner PPO dans MuJoCo (environ une heure sur CPU)
python -m sim_mujoco.evaluate_mujoco --episodes 500 # 4. réévaluer : la ligne ppo_mujoco s'ajoute au tableau
python -m sim_mujoco.render_compare                 # 5. GIF de comparaison
```

Sous Windows, ces commandes se lancent depuis la racine du dépôt, avec le Python du venv
(`.venv\Scripts\python.exe -m ...`). Autres scripts : `python -m sim_mujoco.ant_demo`, `python -m sim_mujoco.run_expert`,
et un test de sensibilité avec un robot plus lent :
`python -m sim_mujoco.evaluate_mujoco --episodes 200 --k-v 6.7 --k-w 0.25 --out results/mujoco/sensibilite_lent`.

## Résultats

Cartes de test jamais vues à l'entraînement, identiques dans les deux simulateurs (500 cartes).

| Politique | Succès cinématique | Succès MuJoCo | Collisions cinématique | Collisions MuJoCo | Durée cinématique | Durée MuJoCo |
|---|---:|---:|---:|---:|---:|---:|
| Expert (VFH) | 96,6 % | 97,0 % | 0,0 % | 0,0 % | 13,1 s | 13,2 s |
| BC (DeepSets) | 92,8 % | 93,6 % | 3,0 % | 2,8 % | 13,7 s | 13,7 s |
| DAgger | 96,0 % | 95,4 % | 0,2 % | 0,4 % | 14,0 s | 13,9 s |
| PPO | 97,2 % | 97,4 % | 2,6 % | 2,6 % | 8,6 s | 8,8 s |
| PPO (entraîné dans MuJoCo) | — | 95,0 % | — | 4,8 % | — | 9,0 s |

Sensibilité à la dynamique (les 200 premières cartes de test, sauf la colonne nominale : succès / collisions) :

| Politique | Nominal : constantes de temps 0,15 s et 0,1 s | Moyen : 0,38 s et 0,2 s | Lent : 0,75 s et 0,4 s |
|---|---:|---:|---:|
| Expert (VFH) | 97,0 % / 0,0 % | 98,5 % / 0,0 % | 97,0 % / 0,0 % |
| BC (DeepSets) | 93,6 % / 2,8 % | 92,0 % / 5,0 % | 90,5 % / 6,0 % |
| DAgger | 95,4 % / 0,4 % | 94,0 % / 1,5 % | 91,0 % / 5,0 % |
| PPO | 97,4 % / 2,6 % | 95,0 % / 5,0 % | 92,5 % / 7,5 % |
| PPO (entraîné dans MuJoCo) | 95,0 % / 4,8 % | 95,0 % / 5,0 % | 92,5 % / 7,5 % |

**Ce que montrent ces chiffres.**

1. À la dynamique choisie, les politiques existantes passent presque intactes dans MuJoCo : moins d'un point d'écart sur
   le succès, dans les intervalles de confiance. L'hypothèse de départ (le retard du robot ferait monter les collisions)
   n'est pas confirmée à ces valeurs.
2. Avec un robot beaucoup plus lent (plus de trois pas de contrôle pour répondre), les politiques apprises se dégradent
   progressivement (collisions de 2,8 % à 6,0 % pour le BC, de 0,4 % à 5,0 % pour DAgger, de 2,6 % à 7,5 % pour PPO),
   alors que l'expert reste à 0 % de collision. Explication possible, non testée : sa marge de sécurité explicite.
3. Le PPO réentraîné dans MuJoCo (95,0 % de succès, 4,8 % de collisions) ne fait pas mieux que le PPO d'origine évalué
   dans MuJoCo (97,4 %, 2,6 %) : à cette dynamique il n'y avait presque rien à rattraper, et avec une seule graine
   d'entraînement on ne peut pas distinguer un vrai écart du hasard. Le test qui trancherait, entraîner PPO avec le robot
   lent, n'a pas été fait.

Prudence : avec 200 cartes, l'intervalle de confiance fait environ ±3 à ±4 points.

## Limites

Les paramètres du robot sont des choix, pas des mesures. La physique reste plane et les obstacles sont statiques et
parfaits. L'intégration continue n'installe pas MuJoCo : les tests de l'extension y sont ignorés.

# Fashion Trend Intelligence

> Projet 2 — Formation AI Engineer · OpenClassrooms

Segmentation sémantique de vêtements sur des photos d'influenceurs via l'API Hugging Face, avec évaluation quantitative des prédictions (IoU, Dice, Précision) comparées à des masques de vérité terrain.

---

## Contexte

Dans le cadre de la formation **AI Engineer d'OpenClassrooms**, ce projet simule une mission pour une startup d'analyse des tendances mode. L'objectif : automatiser la détection et la segmentation des pièces vestimentaires portées par des influenceurs, afin d'identifier les tendances émergentes à partir d'un corpus d'images.

Le pipeline couvre l'appel à un modèle de segmentation en production (Hugging Face Inference API), la visualisation des résultats et la validation rigoureuse des performances sur 50 images annotées.

---

## Ce que fait le projet

1. **Segmentation par lot** — envoie 50 photos d'influenceurs (2024) à l'API Hugging Face et récupère un masque de segmentation par image
2. **Visualisation comparée** — affiche côte à côte l'image originale, le masque prédit et le masque de référence
3. **Évaluation des métriques** — calcule IoU, Dice (F1) et Précision pour chacune des 18 classes vestimentaires
4. **Synthèse tabulaire** — produit un DataFrame Pandas (900 lignes × 12 colonnes) permettant d'analyser les performances classe par classe et image par image

### Classes détectées (18)

| ID | Classe | ID | Classe |
|----|--------|----|--------|
| 0 | Arrière-plan | 9 | Chaussure gauche |
| 1 | Chapeau | 10 | Chaussure droite |
| 2 | Cheveux | 11 | Visage |
| 3 | Lunettes | 12 | Jambe gauche |
| 4 | Haut | 13 | Jambe droite |
| 5 | Jupe | 14 | Bras gauche |
| 6 | Pantalon | 15 | Bras droit |
| 7 | Robe | 16 | Sac |
| 8 | Ceinture | 17 | Écharpe |

---

## Technologies

| Catégorie | Outil |
|-----------|-------|
| Langage | Python 3.13+ |
| Gestion des dépendances | Poetry |
| Modèle de segmentation | [`sayeed99/segformer_b3_clothes`](https://huggingface.co/sayeed99/segformer_b3_clothes) via Hugging Face Inference API |
| Traitement d'images | Pillow 12, NumPy 2 |
| Analyse de données | Pandas 2 |
| Visualisation | Matplotlib 3 |
| Environnement | Jupyter Notebook |
| Variables d'environnement | python-dotenv |

---

## Exemple de résultat

Chaque image produit une grille en 4 colonnes :

```
┌──────────────┬─────────────────────┬──────────────────┬──────────────────┐
│  Photo       │  Masque prédit      │  Masque vérité   │  IoU par classe  │
│  originale   │  (segformer_b3)     │  terrain         │  (bar chart)     │
└──────────────┴─────────────────────┴──────────────────┴──────────────────┘
```

> Exemple : `![Segmentation image_9](notebooks/top_influenceurs_2024/IMG/image_9.png)`

**Métriques observées (image_9, exemple) :**

| Classe | IoU |
|--------|-----|
| Jupe | 0.878 |
| Cheveux | 0.795 |
| Haut | 0.712 |
| Ceinture | 0.641 |
| Jambe gauche | 0.180 |
| **mIoU moyen** | **~0.55** |

---

## Structure du projet

```
Fashion Trend Intelligence/
├── notebooks/
│   ├── fashion_trend_intelligence.ipynb   ← notebook principal
│   └── top_influenceurs_2024/
│       ├── IMG/                           ← 50 photos d'influenceurs
│       └── Mask/                          ← 50 masques de vérité terrain
├── huggingface_api_cloth_seg.ipynb        ← tutoriel API HF
├── Exemple_Viz_segmentation_dataset.ipynb ← visualisation du dataset
├── pyproject.toml
├── poetry.lock
├── .env.example
└── .gitignore
```

---

## Lancer le projet

### 1. Prérequis

- Python 3.13+
- Un compte [Hugging Face](https://huggingface.co) avec un token d'accès

### 2. Installation

**Option A — Poetry (recommandé)**

```bash
# Cloner le dépôt
git clone <url-du-repo>
cd "Fashion Trend Intelligence"

# Installer les dépendances
poetry install
```

**Option B — pip + requirements.txt**

```bash
git clone <url-du-repo>
cd "Fashion Trend Intelligence"

python -m venv .venv
source .venv/bin/activate  # Windows : .venv\Scripts\activate

pip install -r requirements.txt
```

### 3. Configuration

```bash
# Copier le fichier d'exemple
cp .env.example .env

# Éditer .env et renseigner votre token Hugging Face
# HUGGINGFACE_HUB_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxx
```

> Obtenez votre token sur https://huggingface.co/settings/tokens (rôle "read" suffisant).

### 4. Exécution

```bash
# Lancer Jupyter dans l'environnement Poetry
poetry run jupyter notebook

# Ouvrir : notebooks/fashion_trend_intelligence.ipynb
# Exécuter toutes les cellules (Kernel > Restart & Run All)
```

Le notebook exécute le pipeline complet :
- appels API pour les 50 images (≈ 1 min avec le délai anti-rate-limit)
- génération des visualisations comparées
- calcul et affichage des métriques

---

## Auteur

**Jonathan Fernandez** — Formation AI Engineer, OpenClassrooms

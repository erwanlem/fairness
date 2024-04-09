# Fairness final

Travail sur les données des accidents corporels de la circulation routière de 2005 à 2022.

Travail sur la classification des véhicules comme impliqués dans un accident mortel ou non mortel.
L'objectif de ce projet est l'identification des attributs sensibles, et de corriger les biais qu'ils peuvent provoquer.

Le répertoire de ce projet est organisé comme suit:
- Le **rapport** au format *pdf* `rapport/rapport.pdf`
- `main.ipynb` contient le code principal
- `mainGaussien.ipynb` contient le code avec le modèle *GaussianNB*
- `lime.ipynb` contient le travail effectué avec *lime*
- Le fichier `utils.py` contient l'ensemble des fonctions auxiliaires utilisées dans `main.ipynb`

Vous pouvez retrouver la base de donnée dans le répertoire `dataset/`.


Participants:
- Yannis CHUPIN L3 Info
- Erwan LEMATTRE LDD3 Mag. Info

## Organisation

### Résultats
- `model.ipynb` : fichier entrainement du modèle, le modèle est enregistré dans le répertoire models pour utilisation depuis d'autres fichiers.
- `analyze.ipynb` : fichier analyse du dataset et recherche de l'attribut sensible.
- `preprocessing.ipynb` : contient les méthodes de preprocessing
- `inprocessing.ipynb` : contient la méthode inprocessing
- `postprocessing.ipynb` : contient les méthodes postprocessing

### Utilitaires
- `utils.py` : divers outils. Permet de réduire la quantité de code dans les fichiers principaux
- `dataset_prepare.py` : fonctions pour préparer le dataset rapidement

## Présentation

Support pour l'oral final : https://upsud-my.sharepoint.com/:p:/g/personal/yannis_chupin_universite-paris-saclay_fr/Ea5M5moxLA5Br4gq7a7e4b4B1FSPjRAmwQC-zdOhjXNGkg?e=AFMnig 

## Documentation

Source du dataset: https://www.data.gouv.fr/fr/datasets/bases-de-donnees-annuelles-des-accidents-corporels-de-la-circulation-routiere-annees-de-2005-a-2022/


- Lime : https://christophm.github.io/interpretable-ml-book/lime.html#lime
- shapley values : https://christophm.github.io/interpretable-ml-book/shapley.html

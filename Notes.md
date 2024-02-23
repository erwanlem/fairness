# Notes sur les attributs

## Attributs retirés :

- l'adresse au sens manuscrit : ne nous sert pas, la longitude/latitude est suffisant, couplé au nim du dpt et de la commune.

## Attributs ajoutés :

- à partir du dataset : déduire si l'accident a été mortel, ie si une des personnes impliquées est morte dans l'accident. 1 indique qu'il y aeu un mort, 0 le reste .

- cat 50-70 et 70-100

## Choses à faire :

- les one hot sur les différentes cat (lum, col, vma, trajet, situ, atm, catv, choc, surf, plan (bin), prof, circ, catr, trajet (obs, obsm, infra (à voir), ) Yannis 
- ajouter 1 catégorie vma Yannis
- fusionner 0 et -1 nbv
- regarder voie ultérieurement
- ajouter le volet des métriques de fairness - Erwan
- netoyage du code : 
    - faire un .py
    - ajouter du commentaitre
    - ajouter du md
- rapport d'étape 
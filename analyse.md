# Premières annalyses de la correction de biais
## 1. Le cas du sexe conducteur

### 1.1 Pré-processing

### 1.3 Post processing

#### 1.3.1 RejectOption Classification 

En mettant privileged à 1 (Hommes) et unprivileged à 0 (Femmes) : les seuils recomandés sont :
 - Sans fairness : 11.89 %
 - Avec fairness : 12.89 %

> La différence est maigre, on peut estimer que le biais du sexe du conducteur existe mais a un impact minime. 

En inversant : privileged à 0 (Femmes) et unprivileged à 1 (Hommes) : les seuils recomandés sont identiques :
 - Sans fairness : 12.89 %
 - Avec fairness : 12.89 %

> On en déduit que le post process ne juge pas utile de changer le seuil.
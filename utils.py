import plotly.express as px
import numpy as np
import random
from shapkit.monte_carlo_shapley import MonteCarloShapley, MonteCarloShapleyBatch
import pandas as pd
from threading import Thread, Lock
import multiprocessing
from sklearn.ensemble import RandomForestClassifier
from collections import OrderedDict
from aif360.metrics import ClassificationMetric
from sklearn import preprocessing





def pred_thres(pred, thres=0.5):
    p = np.array([])
    for i in pred:
        if i[1] > thres:
            p = np.append(p, 1)
        else:
            p = np.append(p, 0)
    return p


class custom_RFC(RandomForestClassifier):

    def __init__(self, n_estimators = 100, *, criterion = "gini", max_depth = None, min_samples_split = 2, min_samples_leaf = 1, min_weight_fraction_leaf = 0, max_features = "sqrt", max_leaf_nodes = None, min_impurity_decrease = 0, bootstrap: bool = True, oob_score: bool = False, n_jobs = None, random_state = None, verbose = 0, warm_start: bool = False, class_weight = None, ccp_alpha: float = 0, max_samples: float | None | int = None) -> None:
        super().__init__(n_estimators, criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features, max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=min_impurity_decrease, bootstrap=bootstrap, oob_score=oob_score, n_jobs=n_jobs, random_state=random_state, verbose=verbose, warm_start=warm_start, class_weight=class_weight, ccp_alpha=ccp_alpha, max_samples=max_samples)

    def predict(self, X, threshold=0.2):
        preds = pred_thres(self.predict_proba(X), 0.2)
        return preds









########################################
# Signification des valeurs du dataset #
########################################
attributes_values = {
   'catv' : {0:"Autre", 1:"Bicyclette", 2:"Cyclomoteur", 3:"Voiture", 4:"Utilitaire", 5:"Moto"},
   'sexe_conducteur' : {1:"Homme", 0:"Femme"},
   'catr' : {1:"Autoroute", 2:"Route nationale", 3:"Route départementale", 4:"Voie communale", 5: "Autre"},
   'col' : {-1:"Non renseigné", 
            1:"Deux véhicules - Fontale", 
            2:"Deux véhicules - par l'arrière", 
            3:"Deux véhicules - par le côté", 
            4:"Trois véhicules et plus en chaine", 
            5:"Trois véhicules et plus collisions multiples",
            6:"Autre collision", 
            7: "Pas de collision"},
    'obs' : {0:"Pas d'obstacle", 1:"Obstacle"},
    'agg' : {1 : "Hors-Aglomération", 2 : "En Aglomération"}, 
    'pieton' : {0 : "Aucun Piéton impliqué", 1 : "Piéton(s) impliqué(s)"},
    'atm' : {-1 : "Non renseigné",
            1 : "Normale",
            2 : "Pluie légère",
            3 : "Pluie forte",
            4 : "Neige - grêle",
            5 : "Brouillard - fumée",
            6 : "Vent fort - tempête", 
            7 : "Temps éblouissant", 
            8 : "Temps couvert",
            9 : "Autre" },
    'obsm' : {-1 : "Non renseigné", 
              0 : "Aucun",
              1 : "Piéton",
              2 : "Véhicule",
              4 : "Véhicule sur rail",
              5 : "Animal domestique",
              6 : "Animal sauvage",
              9 : "Autre"}
}



# Supprime les lettres des départements
def transforme_dpt(n):
    if n[-1] == 'D': # Rhone
        return '69'
    elif n[-1] == 'M': # Lyon métropole
        return '96'
    elif n[-1] == 'A': # Corse du sud
        return '97'
    elif n[-1] == 'B': # Haute Corse
        return '98'
    else:
        return str(n)
    

# Découpage en 4 catégories de vitesse
# Identifiants:
#   1 -> < 50
#   2 -> >=50 && <=70
#   3 -> > 70 && < 100
#   3 -> >=100
def split_vma(df):
    df = df.copy()
    v = df['vma']
    cat = []
    for i in v:
        if i < 50:
            cat.append(1)
        elif i>=50 and i<=70:
            cat.append(2)
        elif i >= 100:
            cat.append(4)
        else:
            cat.append(3)
    df['vma'] = cat
    return df 


def simplify_catr(df):
    df = df.copy()
    r = []
    for i in df['catr']:
        if i > 4:
            r.append(5)
        else:
            r.append(i)
    df['catr'] = r
    return df



# Réduction du nombre de carégories de véhicules
# Les nouveaux indices sont :
# 1 : Bicyclette
# 2 : cyclomoteur
# 3 : VL (voiture)
# 4 : Utilitaire
# 5 : Motocyclette
# 0 : autre
def simplify_catv(df):
    df = df.copy()
    cat = [1, 2, 7, 33, 10]
    corresp = {1:1, 2:2, 7:3, 10:4, 33:5}

    r = []
    for i in df['catv']:
        if i in cat:
            r.append(corresp[i])
        else:
            r.append(0)
    df['catv'] = r
    return df


# rend pour chaque ligne si l'accident associé est mortel
def extract_mortal(df):
    df = df.copy()
    # Véhicule impliqué dans un accident mortel
    with_death = df[df['grav'] == 2]
    acc_with_death = with_death['Num_Acc']

    d = []
    for i in df['Num_Acc']:
        if i in acc_with_death.to_numpy():
            d.append(1)
        else:
            d.append(0)
    return d

# Passage d'une caractéristique à un attribut
def to_attribute(df, id_valid, cat_a, cat_b):
    m = []
    for i in df['id_vehicule']:
        if i in id_valid['id_vehicule'].to_numpy():
            m.append(cat_a)
        else:
            m.append(cat_b)
    return m

# Retourne une colonne avec l'âge du conducteur pour chaque ligne
def get_driver_age(df):
    df = df.copy()
    # Age du conducteur du véhicule
    driver_age = df[(df['catu'] == 1)][['an_nais', 'id_vehicule', 'an']]

    age = driver_age['an'] - driver_age['an_nais']
    return age


# Metrics function depuis leur fichier 'common-utils' sur le git aif360
def compute_metrics(dataset_true, dataset_pred, 
                    unprivileged_groups, privileged_groups,
                    disp = True):
    """ Compute the key metrics """
    classified_metric_pred = ClassificationMetric(dataset_true,
                                                 dataset_pred, 
                                                 unprivileged_groups=unprivileged_groups,
                                                 privileged_groups=privileged_groups)
    metrics = OrderedDict()
    metrics["Balanced accuracy"] = 0.5*(classified_metric_pred.true_positive_rate()+
                                             classified_metric_pred.true_negative_rate())
    metrics["Statistical parity difference"] = classified_metric_pred.statistical_parity_difference()
    metrics["Disparate impact"] = classified_metric_pred.disparate_impact()
    metrics["Average odds difference"] = classified_metric_pred.average_odds_difference()
    metrics["Equal opportunity difference"] = classified_metric_pred.equal_opportunity_difference()
    metrics["Theil index"] = classified_metric_pred.theil_index()
    
    if disp:
        for k in metrics:
            print("%s = %.4f" % (k, metrics[k]))
    
    return metrics

def rapport_corr(x, y):
  '''
  Calcule le rapport de correlation entre une variable
  qualitative (x) et une variable quantitative (y)
  x : list of the qualitative variable observations
  y : list of the quantitative variable obesvations
  x and y must be of the same  length
  '''
  if len(x)!=len(y):
    print("The two list doesn't have the same length {len(x)}!={len(y)}")
    return -1
  mean_y = np.mean(y)
  se2 = 0
  sr2 = 0
  for cat in set(x):
    y_cat = [y[i] for i in range(len(x)) if x[i]==cat]
    se2 += len(y_cat)*(np.mean(y_cat)-mean_y)**2
    sr2 += len(y_cat)*np.var(y_cat)
  se2/=len(y)
  sr2/=len(y)
  return se2/(se2 + sr2)



def analyse_bi_quali_quanti(quali, quanti, df_):
  df = df_.copy()
  # Rapport de correlation
  rapp = rapport_corr(df[quali].values, df[quanti].values)
  print(f"Rapport de corr {quali} X {quanti}: {rapp} ")

  if quali in attributes_values.keys():
    df[quali] = df[quali].replace(attributes_values[quali])
  if quanti in attributes_values.keys():
    df[quanti] = df[quanti].replace(attributes_values[quanti])

  # boite a moustaches
  bam = px.box(df, x=quali, y=quanti)
  bam.show()
  # histogramme
  hist = px.histogram(df, x=quanti, color=quali, barmode='overlay', opacity=0.75)
  hist.show()

  hist_prop = px.histogram(df, x=quanti, color=quali, barmode='overlay', opacity=0.75, histnorm="probability")
  hist_prop.show()


def analyse_bi_quali_quali(quali1, quali2, df, numerical_features):
  if quali1==quali2:
    return

  # table de contingence
  # on groupe le dataframe sur les variables d'intéret
  df_group = df[[quali1, quali2, numerical_features[0]]].groupby([quali1,quali2]).count().reset_index()
  df_group.rename(columns={numerical_features[0]:"count"}, inplace=True)

  rows = list(set(df_group[quali2].values))
  cols = list(set(df_group[quali1].values))
  contingence_tab = [
      [ df_group[(df_group[quali2]==row) & (df_group[quali1]==col)]["count"].values[0] if col in df_group[df_group[quali2]==row][quali1].values else 0 for row in rows  ]
      for col in cols]
  
  if quali1 in attributes_values.keys():
    df_group[quali1] = df_group[quali1].replace(attributes_values[quali1])
  if quali2 in attributes_values.keys():
    df_group[quali2] = df_group[quali2].replace(attributes_values[quali2])

  contingence_img = px.imshow(contingence_tab,
                              text_auto=True,
                              labels=dict(x=quali2, y=quali1),
                              x=rows, y=cols)
  contingence_img.show()

  # diagrammes en barres
  quali1_count = df_group[[quali1, "count"]].groupby([quali1]).sum()
  df_group['count_quali1_prop'] = df_group[[quali1,"count"]].apply(lambda x: 100*x['count']/quali1_count.loc[x[quali1]], axis=1)
  bar1 = px.bar(df_group, x=quali1, y="count_quali1_prop", color=quali2, text_auto=True)
  bar1.show()

  quali2_count = df_group[[quali2, "count"]].groupby([quali2]).sum()
  df_group['count_quali2_prop'] = df_group[[quali2,"count"]].apply(lambda x: 100*x['count']/quali2_count.loc[x[quali2]], axis=1)
  bar2 = px.bar(df_group, x=quali2, y="count_quali2_prop", color=quali1, text_auto=True)
  bar2.show()


def reduce_trajet_values(df):
  df = df.copy()
  t = df['trajet']
  r = []
  for i in t:
    if i == 9 or i == -1:
      r.append(0)
    else:
      r.append(i)
  return r

def reduce_surf_values(df):
  df = df.copy()
  t = df['surf']
  r = []
  for i in t:
    if i == 9 or i == -1:
      r.append(0)
    else:
      r.append(i)
  return r 

def reduce_obs_values(df):
  df = df.copy()
  t = df['obs']
  r = []
  for i in t:
    if i == 0:
      r.append(0)
    else:
      r.append(1)
  return r



def predictproba_fn(data, clf, columns):
  df_test = pd.DataFrame(
            [data], columns=columns )
  preds = clf.predict_proba(df_test)
  return np.squeeze(preds, axis=0)

## Parallélisation du calcul des valeurs de Shapley
## n_iter doit être un multiple de 5
class parallel_shap:
  def __init__(self, data_test, X_train, clf, ids, n_iter=5):
    self.data_test = data_test
    self.X_train = X_train
    self.clf = clf
    self.ids = ids
    self.n_iter = n_iter
    self.results = []
    cols = data_test.columns.to_list()
    cols.remove('Y')
    self.fc = lambda x: predictproba_fn(x, clf, cols)[1]
    self.queue = multiprocessing.Queue()
    self.queue.put([])

  def shap_computation(self, data_test, X_train, clf, fc, id, result):
    query_instance = data_test.drop(columns="Y")[id:(id+1)]
    x_class = int(clf.predict(query_instance))
    query_instance=query_instance.squeeze()
    
    X_opposite_class = X_train[clf.predict(X_train) != x_class].copy()
    reference = X_opposite_class.sample()
    ref_class = int(clf.predict(reference))
    reference=reference.squeeze()

    true_shap = MonteCarloShapley(x=query_instance, fc=fc, ref=reference, n_iter=1000, callback=None)
    ret = self.queue.get()
    ret.append((true_shap, query_instance, reference))
    self.queue.put(ret)

  def start_shap(self):
    for i in range (0, self.n_iter//5):
      id = self.ids[random.randint(0, len(self.ids))]
      self.p1 = multiprocessing.Process(target=self.shap_computation, args=(self.data_test, self.X_train, self.clf, self.fc, id, self.results))

      id = self.ids[random.randint(0, len(self.ids))]
      self.p2 = multiprocessing.Process(target=self.shap_computation, args=(self.data_test, self.X_train, self.clf, self.fc, id, self.results))

      id = self.ids[random.randint(0, len(self.ids))]
      self.p3 = multiprocessing.Process(target=self.shap_computation, args=(self.data_test, self.X_train, self.clf, self.fc, id, self.results))

      id = self.ids[random.randint(0, len(self.ids))]
      self.p4 = multiprocessing.Process(target=self.shap_computation, args=(self.data_test, self.X_train, self.clf, self.fc, id, self.results))
      
      id = self.ids[random.randint(0, len(self.ids))]
      self.p5 = multiprocessing.Process(target=self.shap_computation, args=(self.data_test, self.X_train, self.clf, self.fc, id, self.results))

      self.p1.start()
      self.p2.start()
      self.p3.start()
      self.p4.start()
      self.p5.start()


      self.p1.join()
      self.p2.join()
      self.p3.join()
      self.p4.join()
      self.p5.join()
    self.results = self.queue.get()

  def stop_shap(self):
    self.p1.terminate()
    self.p2.terminate()
    self.p3.terminate()
    self.p4.terminate()
    self.p5.terminate()


from sklearn.metrics import confusion_matrix


def print_conf_matrix(model, y_test, preds):
  tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
  tn, fp, fn, tp
  import plotly.express as px

  fig = px.imshow([[tn, fp], [fn, tp]], text_auto=True, labels=dict(y="Truth", x="Pred"),
                  x=["False", "True"],
                  y=["False", "True"]
                )
  fig.show()


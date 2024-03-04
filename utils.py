import plotly.express as px
import numpy as np


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



def analyse_bi_quali_quanti(quali, quanti, df):
  # Rapport de correlation
  rapp = rapport_corr(df[quali].values, df[quanti].values)
  print(f"Rapport de corr {quali} X {quanti}: {rapp} ")
  # boite a moustaches
  bam = px.box(df, x=quali, y=quanti)
  bam.show()
  # histogramme
  hist = px.histogram(df, x=quanti, color=quali, barmode='overlay', opacity=0.75)
  hist.show()

  hist_prop = px.histogram(df, x=quanti, color=quali, barmode='overlay', opacity=0.75, histnorm="probability")
  hist_prop.show()


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

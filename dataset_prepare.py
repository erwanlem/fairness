import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from utils import *
from sklearn.model_selection import train_test_split
import imblearn
from aif360.datasets.standard_dataset import StandardDataset
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer



# valeurs catégorielles
categorical_features = ['trajet', 'catr', 'circ', 'nbv', 'prof',
                        'plan', 'surf', 'vma', 'lum', 'agg', 
                        'int', 'atm', 'col', 'catv', 'obs', 'obsm', 'choc', 'pieton',
                        'sexe_conducteur', 'infra', 'situ']
# valeurs numériques
numerical_features = ['dep','age', 'mois']


'''
    Chargement et préparation du dataset
    en une seule fonction
'''
def load_dataset():
    label = "mortal"

    # Lecture datasets
    df1 = pd.read_csv("dataset/usagers-2022.csv", sep=';')
    df2 = pd.read_csv("dataset/lieux-2022.csv", sep=';', low_memory=False)
    df3 = pd.read_csv("dataset/carcteristiques-2022.csv", sep=';')
    df4 = pd.read_csv("dataset/vehicules-2022.csv", sep=';')

    df4 = df4.drop(columns=['id_vehicule', 'num_veh'])


    df = df1.join(df2.set_index('Num_Acc'), on='Num_Acc')
    df = df.join(df3.set_index('Accident_Id'), on='Num_Acc')
    df = df.join(df4.set_index('Num_Acc'), on='Num_Acc', lsuffix='_')

    # Suppression colonnes inutiles
    df = df.drop(columns=['voie', 'v1', 'v2', 'pr', 'pr1', 'lartpc', 'larrout'
                        , 'num_veh', 'occutc', 'adr', 'senc','etatp','actp', 
                        'manv', 'jour', 'com', 'hrmn', 'motor', 'place', 'vosp', 'locp', 'lat', 'long'])

    df = df.drop_duplicates(subset=['id_usager']) # retire les doublons dans les usagers

    # Remplacement des valeurs NaN
    #df['an_nais'] = df['an_nais'].fillna(df['an_nais'].mode()[0])
    df = df.dropna()

    # Convertir en entier
    df['id_vehicule'] = df['id_vehicule'].apply(lambda l: l[0:3] + l[4:7])
    df['id_vehicule'] = df['id_vehicule'].astype(int)
    df['sexe'] = df['sexe'].astype(int)

    df_2 = df.copy()

    # On crée un attribut pour les accidents mortels
    df_2['mortal'] = extract_mortal(df)

    # Accident impliquant un piéton
    has_pedestrian = df[df['catu'] == 3]
    p = to_attribute(df, has_pedestrian, 1, 0)
    df_2['pieton'] = p

    # Sexe du conducteur
    driver = df[(df['catu'] == 1) & (df['sexe'] == 1)]
    dr = to_attribute(df, driver, 1, 0)
    df_2['sexe_conducteur'] = dr


    df_2 = df_2.drop_duplicates(subset=['id_vehicule'])

    # On réduit les carégories de véhicules
    df_2 = simplify_catv(df_2)

    # On enlève la catégorie peu repésentées qu'on ajoute dans une catégorie autre (identifiant 5)
    df_2 = simplify_catr(df_2)

    # Découpage en 4 catégories de vitesse
    df_2 = split_vma(df_2)

            
    # Département en entiers
    df_2['dep'] = df_2['dep'].apply(transforme_dpt)
    df_2['dep'] = pd.to_numeric(df_2['dep'], errors='coerce', downcast='integer')

    # nbv en entier
    df_2['nbv'] = pd.to_numeric(df_2['nbv'], errors='coerce', downcast='integer')
    df_2['nbv'].fillna(2, inplace=True) # only one entry 
    df_2['nbv'] = pd.to_numeric(df_2['nbv'], errors='coerce', downcast='integer')

    # Lat et long en float :
    #df_2['lat'] = pd.to_numeric(df_2['lat'], errors='coerce')
    #df_2['long'] = pd.to_numeric(df_2['long'], errors='coerce')

    # Age du conducteur du véhicule
    df_2['age'] = get_driver_age(df_2)
    #df_2['age'] = df_2['age'].fillna(df_2['age'].mode()[0])
    df_2['age'] = pd.to_numeric(df_2['age'], errors='coerce', downcast='integer')

    # Réduire les valeurs de trajet
    df_2['trajet'] = reduce_trajet_values(df_2)

    # Réduire les valeurs de surf
    df_2['surf'] = reduce_surf_values(df_2)

    # obs prend pour valeur 0 si obstacle 1 si pas d'obstacle
    df_2['obs'] = reduce_obs_values(df_2)

    df3 = pd.DataFrame()
    df3['bidule'] = df_2[(df_2['age'].isna())][['an_nais']]
    df3['bidule2'] = df_2[(df_2['age'].isna())][['an']]

    df_2.loc[df_2['age'].isna(), 'age'] = (df3['bidule2'] - df3['bidule'])

    df_2[(df_2['age'].isna())]['age']
    df_2['age'] = df_2['age'].astype(int)

    # On enlève les attributs qui ne sont plus utiles
    df_2 = df_2.drop(columns=['grav', 'sexe','catu', 'id_usager', 'id_vehicule',
                              'secu1','secu2','secu3', 'an', 'an_nais'])


    return df_2

def test_train_sets(df, train_ratio=0.33):
    df_2 = df.copy(deep=True)

    # On découpe le jeu de données tout en conservant ensemble les véhicules impliqué dans un même accident
    # Les données d'entrainement et de test n'ont donc pas de rapport direct

    unique_accidents = df_2['Num_Acc'].unique() # Num_Acc uniques

    df_3 = df_2.drop(columns=['mortal'])

    # Création des train et test set à partir des numéros d'accident
    X_train, X_test = train_test_split(unique_accidents, test_size=train_ratio, random_state=42)

    # On peut ensuite récupérer les véhicules correspondants aux accidents
    train_df = df_2[df_2['Num_Acc'].isin(X_train)]
    test_df = df_2[df_2['Num_Acc'].isin(X_test)]
    y_train = train_df['mortal']
    y_test = test_df['mortal']
    X_train = train_df.drop(columns=['mortal', 'Num_Acc'])
    X_test = test_df.drop(columns=['mortal', 'Num_Acc'])
    df_2 = df_2.drop(columns='Num_Acc')

    return X_train, X_test, y_train, y_test



def prepare_standard_dataset(X_train, y_train, X_test, y_test, label, vt=False, oversample=True, transform=True):
    col = X_train.columns
    scaler = StandardScaler()

    if oversample:
        resample = imblearn.over_sampling.SMOTE(random_state=42)
        X_train, y_train = resample.fit_resample(X_train, y_train)

    train_dataset = X_train.copy(deep=True)
    train_dataset[label] = y_train

    test_dataset = X_test.copy(deep=True)
    test_dataset[label] = y_test

    if transform:
        X_train = scaler.fit_transform(X_train, y_train)

    df_standard_test = StandardDataset(test_dataset, label
                              , favorable_classes=[1]
                              , protected_attribute_names=['sexe_conducteur']
                              , privileged_classes= [[1]])

    dataset_orig_train = StandardDataset(train_dataset, label
                                , favorable_classes=[1]
                                , protected_attribute_names=['sexe_conducteur']
                                , privileged_classes= [[1]])
    
    if vt:
        dataset_orig_valid, dataset_orig_test = df_standard_test.split([0.5], shuffle=True)
        return dataset_orig_train, dataset_orig_valid, dataset_orig_test
    else:
        return dataset_orig_train, df_standard_test

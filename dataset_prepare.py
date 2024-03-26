import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from utils import *


def load_dataset():
    label = "mortal"

    # Lecture datasets
    df1 = pd.read_csv("dataset/usagers-2022.csv", sep=';')
    df2 = pd.read_csv("dataset/lieux-2022.csv", sep=';')
    df3 = pd.read_csv("dataset/carcteristiques-2022.csv", sep=';')
    df4 = pd.read_csv("dataset/vehicules-2022.csv", sep=';')

    df4 = df4.drop(columns=['id_vehicule', 'num_veh'])


    df = df1.join(df2.set_index('Num_Acc'), on='Num_Acc')
    df = df.join(df3.set_index('Accident_Id'), on='Num_Acc')
    df = df.join(df4.set_index('Num_Acc'), on='Num_Acc', lsuffix='_')

    # Suppression colonnes inutiles
    df = df.drop(columns=['voie', 'v1', 'v2', 'pr', 'pr1', 'lartpc', 'larrout'
                        , 'num_veh', 'occutc', 'adr', 'senc','etatp','actp', 
                        'manv', 'jour', 'com', 'hrmn', 'motor', 'place', 'vosp', 'locp'])

    df = df.drop_duplicates(subset=['id_usager']) # retire les doublons dans les usagers

    # Remplacement des valeurs NaN
    df['an_nais'] = df['an_nais'].fillna(df['an_nais'].mode()[0])

    # Convertir en entier
    df['id_vehicule'] = df['id_vehicule'].apply(lambda l: l[0:3] + l[4:7])
    df['id_vehicule'] = df['id_vehicule'].astype(int)
    df['sexe'] = df['sexe'].astype(int)

    # On préserve le df d'origine
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

    df_2['dep'] = df_2['dep'].apply(transforme_dpt)
    df_2['dep'] = pd.to_numeric(df_2['dep'], errors='coerce', downcast='integer')

    df_2['nbv'] = pd.to_numeric(df_2['nbv'], errors='coerce', downcast='integer')
    df_2['nbv'].fillna(2, inplace=True) # only one entry 
    df_2['nbv'] = pd.to_numeric(df_2['nbv'], errors='coerce', downcast='integer')

    df_2['lat'] = pd.to_numeric(df_2['lat'], errors='coerce')
    df_2['long'] = pd.to_numeric(df_2['long'], errors='coerce')

    df_2['age'] = get_driver_age(df_2)
    df_2['age'] = df_2['age'].fillna(df_2['age'].mode()[0])
    df_2['age'] = pd.to_numeric(df_2['age'], errors='coerce', downcast='integer')

    df_2['trajet'] = reduce_trajet_values(df_2)
    df_2['surf'] = reduce_surf_values(df_2)
    df_2['obs'] = reduce_obs_values(df_2)

    df_2 = df_2.drop(columns=['an_nais','grav', 'sexe','catu', 'id_usager', 'id_vehicule',
                            'secu1','secu2','secu3', 'an', 'lat', 'long'])
    
    return df_2

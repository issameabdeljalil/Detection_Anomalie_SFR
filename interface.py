"""
Challenge Nexialog MoSEF
Interface Streamlit
Permet de simuler une nouvelle heure en insérant des anomalies dans des noeuds
choisis (Boucles, PEAG, OLT). Détecte les noeuds anormaux et les affiche sous formes
de tableau. Visualisation 3d pour voir les anomalies sur la distribution empirique 
des tests
"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import gaussian_kde
from nodes_checker import NodesChecker
from utils import import_json_to_dict

## chargement des données au lancement de l'application ###################################################################

# sotckage dans la session_state
if "donnees_chargees" not in st.session_state:
    # importation de l'heure simulée
    st.session_state.lignes_1fev = pd.read_csv('data/results/lignes_1fev.csv', index_col=0).head(10) # TEMPORAIRE
    # importation des vecteurs de distribution par test
    st.session_state.dict_distribution_test = import_json_to_dict("data/results/dict_test.json")
    # colonnes dans lesquelles injecter et repérer des anomalies
    st.session_state.variables_test = [
        'avg_score_scoring',
        'avg_latence_scoring',
        'avg_dns_time',
        'std_score_scoring',
        'std_latence_scoring',
        'std_dns_time'
    ]
    # nouvelles colonnes de p_values à ajouter
    st.session_state.p_values_col = [
        'p_val_avg_dns_time',
        'p_val_avg_score_scoring',
        'p_val_avg_latence_scoring',
        'p_val_std_dns_time',
        'p_val_std_score_scoring',
        'p_val_std_latence_scoring'
    ]

    st.session_state.donnees_chargees = True
    st.session_state.button_launched = False # boutton de détection d'anomalies

############################################################################################################################

# Affichage streamlit

## config page
st.set_page_config(page_title="Challenge Nexialog", layout="wide")

## logos
col_logo1, col_logo2, col_logo3 = st.columns(3)
with col_logo1:
    st.image("images/SFR_logo.png", width=150)
with col_logo2:
    st.image("images/paris_1_logo.jpg", width=400)
with col_logo3:
    st.image("images/nexialog_logo.png", width=150)

## titre
st.title("Challenge Nexialog : Détection d'anomalies sur le réseau")

## Insertion des anomalies
st.subheader("Insertion des anomalies")

# Boucle
col_names, col_vars, col_val, col_btn = st.columns([2, 2, 1, 1])

with col_names:
    boucle_names = st.multiselect("Boucles", options=sorted(list(st.session_state.lignes_1fev['boucle'].unique())))
with col_vars:
    var_test = st.multiselect("Variables de test", options=st.session_state.variables_test, key='insertvar1')
with col_val:
    valeur_insertion = st.number_input("Valeur à insérer", value=0.0, step=1.0, key='insertion1')
with col_btn:
    if st.button("Insérer", key="btn_inserer1"):
        st.write("Insertion de la valeur ", round(valeur_insertion, 2), "dans les variables : ", var_test, 
                 "passant par les Boucles : ", boucle_names)
        # rajout de d'une valeur aux colonnes de test choisies pour les Boucles choisies
        st.session_state.lignes_1fev = NodesChecker.add_anomalies(st.session_state.lignes_1fev, 
            'boucle', 
            boucle_names,
            var_test,
            valeur_insertion)

# Peag
col_names, col_vars, col_val, col_btn = st.columns([2, 2, 1, 1])

with col_names:
    peag_names = st.multiselect("PEAG", options=sorted(list(st.session_state.lignes_1fev['peag_nro'].unique())))
with col_vars:
    var_test = st.multiselect("Variables de test", options=st.session_state.variables_test, key='insertvar2')
with col_val:
    valeur_insertion = st.number_input("Valeur à insérer", value=0.0, step=1.0, key='insertion2')
with col_btn:
    if st.button("Insérer", key="btn_inserer2"):
        st.write("Insertion de la valeur ", round(valeur_insertion, 2), "dans les variables : ", var_test, 
                 "passant par les PEAG : ", peag_names)
        # rajout de d'une valeur aux colonnes de test choisies pour les PEAG choisis
        st.session_state.lignes_1fev = NodesChecker.add_anomalies(st.session_state.lignes_1fev, 
            'peag_nro', 
            peag_names,
            var_test,
            valeur_insertion)

# OLT
col_names, col_vars, col_val, col_btn = st.columns([2, 2, 1, 1])

with col_names:
    olt_names = st.multiselect("OLT", options=sorted(list(st.session_state.lignes_1fev['olt_name'].unique())))
with col_vars:
    var_test = st.multiselect("Variables de test", options=st.session_state.variables_test, key='insertvar3')
with col_val:
    valeur_insertion = st.number_input("Valeur à insérer", value=0.0, step=1.0, key='insertion3')
with col_btn:
    if st.button("Insérer", key="btn_inserer3"):
        st.write("Insertion de la valeur ", round(valeur_insertion, 2), "dans les variables : ", var_test, 
                 "passant par les OLT : ", olt_names)
        # rajout de d'une valeur aux colonnes de test choisies pour les OLT choisis
        st.session_state.lignes_1fev = NodesChecker.add_anomalies(st.session_state.lignes_1fev, 
            'olt_name', 
            olt_names,
            var_test,
            valeur_insertion)

## bouton de lancement de la detection d'anomalies
if st.button("Lancer la détection d'anomalies par noeud") or st.session_state.button_launched:
    
    st.session_state.button_launched = True
    # instance de classe : recherche les noeuds anormaux
    nc = NodesChecker()
    # calcul des p_values à partir des distributions empiriques
    st.session_state.lignes_1fev = NodesChecker.add_p_values(st.session_state.lignes_1fev, 
                                                             st.session_state.dict_distribution_test)

    ## Tableaux des noeuds avec anomalies detectées
    # 1er tableau : boucles
    st.header("Boucles anormales")
    # p_values issues du test de Fisher pour les boucles
    df_p_values_boucle = nc.get_df_fisher_p_values(st.session_state.lignes_1fev,
                                                   node_type = 'boucle',
                                                   p_values = st.session_state.p_values_col)
    st.dataframe(df_p_values_boucle)

    # 2eme tableau : peag
    st.header("PEAG anormaux")
    # p_values issues du test de Fisher pour les peag
    df_p_values_peag = nc.get_df_fisher_p_values(st.session_state.lignes_1fev,
                                                 node_type = 'peag_nro',
                                                 p_values = st.session_state.p_values_col)
    st.dataframe(df_p_values_peag)

    # 3eme tableau : OLT
    st.header("OLT anormaux")
    # p_values issues du test de Fisher pour les olt
    df_p_values_olt = nc.get_df_fisher_p_values(st.session_state.lignes_1fev, 
                                                node_type = 'olt_name', 
                                                p_values = st.session_state.p_values_col)
    st.dataframe(df_p_values_olt)

    ## Graphiques 3D pour visualiser les anomalies
    st.header("Graphique 3D")
    
    col_test_names, col_test_vars = st.columns([2, 2])

    with col_test_names:
        test_name = st.selectbox("Test", options=sorted(list(st.session_state.lignes_1fev['name'].unique())))
    with col_test_vars:
        variables_test = st.multiselect("Variables de test", options=st.session_state.variables_test)

    if len(variables_test) != 2:
        st.error("Veuillez sélectionner exactement 2 variables de test à représenter en 3D")
        st.stop()

    # ligne choisie par l'utilisateur
    # on représentera la position de la "potentielle" anomalie sur la distribution 3d
    row_to_plot = st.session_state.lignes_1fev[st.session_state.lignes_1fev['name'] == test_name]

    #  axes x et y
    # distribution empirique
    x = np.array(st.session_state.dict_distribution_test[variables_test[0]][test_name])
    y = np.array(st.session_state.dict_distribution_test[variables_test[1]][test_name])

    # borne de la grille
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()

    # grille 2d ici : 100x100 points
    X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])

    # esimation densité empirique avec gaussian_kde
    values = np.vstack([x, y])
    kernel = gaussian_kde(values)
    Z = np.reshape(kernel(positions).T, X.shape)

    # Fig (3d)
    fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z, colorscale='Plasma')])
    fig.update_layout(
        title=f'Distribution Empirique 3D de {test_name}',
        autosize=True,
        scene=dict(
            xaxis_title=variables_test[0],
            yaxis_title=variables_test[1],
            zaxis_title='Densité'
        )
    )
    # utiliser lignes1fev ici
    fig.add_trace(go.Scatter3d(
        x=row_to_plot[variables_test[0]],
        y=row_to_plot[variables_test[1]],
        z=[0.5],                                                       # TEMPORAIRE
        mode='markers+text',
        text=["Anomalie"],
        marker=dict(size=8, color='red')
    ))
    # affiche des figures 3d
    st.plotly_chart(fig)

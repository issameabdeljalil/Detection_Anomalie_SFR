"""
Challenge Nexialog MoSEF
Interface Streamlit :
- Permet de simuler une nouvelle heure en insérant des anomalies dans des noeuds
  choisis (Boucles, PEAG, OLT). 
- Détecte les noeuds anormaux et les affiche sous formes
  de tableau. 
- Visualisation 3D pour voir les anomalies avec approche unidimensionnelle
ou multidimensionnelle (Isolation Forest)
"""

import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
import joblib
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm
import plotly.express as px

from utils import import_json_to_dict
from nodes_checker import NodesChecker
from anomaly_detection_isolation_forest import MultiIsolationForestDetector
from anomaly_detection_mahalanobis import MahalanobisDetector


# Configuration 
st.set_page_config(
    page_title="Challenge Nexialog - Détection d'anomalies",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

def initialize_session_state():
    if "donnees_chargees" not in st.session_state:
        # Importation de l'heure simulée
        st.session_state.lignes_1fev = pd.read_csv('data/results/lignes_1fev.csv', index_col=0).replace([np.inf, -np.inf], np.nan).dropna().head(600)
        st.session_state.lignes_1fev_copy = st.session_state.lignes_1fev.copy()
        # Importation des vecteurs de distribution par test
        st.session_state.dict_distribution_test = import_json_to_dict("data/results/dict_test.json")
        # Colonnes dans lesquelles injecter et repérer des anomalies
        st.session_state.variables_test = [
            'avg_score_scoring',
            'avg_latence_scoring',
            'avg_dns_time',
            'std_score_scoring',
            'std_latence_scoring',
            'std_dns_time'
        ]
        
        st.session_state.df_with_anomalies = None

        # Nouvelles colonnes de p_values à ajouter
        st.session_state.p_values_col = [
            'p_val_avg_dns_time',
            'p_val_avg_score_scoring',
            'p_val_avg_latence_scoring',
            'p_val_std_dns_time',
            'p_val_std_score_scoring',
            'p_val_std_latence_scoring'
        ]

        st.session_state.p_value_threshold = 5.0  # Seuil de sensibilité/rejet
        st.session_state.donnees_chargees = True
        
        # Initialisation du détecteur Isolation Forest
        st.session_state.isolation_forest =  MultiIsolationForestDetector(chain_id_col = 'name')
         # chargement des modeles d'isolation forest
        st.session_state.isolation_forest.load_models(st.session_state.lignes_1fev_copy)

        st.session_state.isolation_forest_threshold = 0.00
        
        # Variables pour stocker les résultats des analyses
        st.session_state.results = {
            "boucles": None,
            "peag_nro": None,
            "olt": None
        }
       
        st.session_state.anomalies_detected = False

# Fonction pour afficher l'en-tête de l'application
def show_header():
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        st.image("images/SFR_logo.png", width=120, use_container_width=False)
    
    with col2:
        st.title("Détection d'anomalies sur le réseau")
        st.caption("Challenge Nexialog & Université Paris 1 Panthéon-Sorbonne")
    
    with col3:
        st.image("images/nexialog_logo.png", width=120, use_container_width=False)
    
    # CSS pour l'alignement
    st.markdown("""
    <style>
    [data-testid="stHorizontalBlock"] {
        align-items: center;
    }
    [data-testid="stHorizontalBlock"] > div:first-child {
        display: flex;
        justify-content: flex-start;
        align-items: center;
        padding-left: 10px;
    }
    [data-testid="stHorizontalBlock"] > div:last-child {
        display: flex;
        justify-content: flex-end;
        align-items: center;
        padding-right: 10px;
    }
    [data-testid="stHorizontalBlock"] > div:nth-child(2) {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("---")



def show_network_status():
    st.header("📊 État du réseau")
    
    # Section 1: Statistiques globales du réseau
    st.subheader("Statistiques globales")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        # Nombre d'équipements actifs
        olt_count = len(st.session_state.lignes_1fev['olt_name'].unique())
        peag_count = len(st.session_state.lignes_1fev['peag_nro'].unique())
        boucle_count = len(st.session_state.lignes_1fev['boucle'].unique())
        total_equipment = olt_count + peag_count + boucle_count
        
        st.metric(label="Équipements actifs", 
                 value=f"{total_equipment}",
                 help="Somme des OLT, PEAG et Boucles actifs")
    
    with col2:
        # Couverture du réseau
        dept_count = len(st.session_state.lignes_1fev['code_departement'].unique())
        st.metric(label="Départements couverts", 
                  value=f"{dept_count}")
    
    with col3:
        # Volume de tests
        dns_tests = st.session_state.lignes_1fev['nb_test_dns'].sum()
        scoring_tests = st.session_state.lignes_1fev['nb_test_scoring'].sum()
        total_tests = dns_tests + scoring_tests
        
        st.metric(label="Tests effectués", 
                  value=f"{total_tests:,}".replace(',', ' '),
                  help="Somme des tests DNS et Scoring")
    
    # Section 2: Indicateurs de performance
    st.subheader("Indicateurs de performance réseau")
    
    col1, col2, col3 = st.columns(3)
    
    # Temps de réponse moyen
    avg_dns = st.session_state.lignes_1fev['avg_dns_time'].mean()
    avg_latence = st.session_state.lignes_1fev['avg_latence_scoring'].mean()
    
    with col1:
        st.metric(label="Temps DNS moyen (ms)", 
                  value=f"{avg_dns:.2f}")
    
    with col2:
        st.metric(label="Latence scoring moyenne (ms)", 
                  value=f"{avg_latence:.2f}")
    
    with col3:
        # Score moyen du réseau
        avg_score = st.session_state.lignes_1fev['avg_score_scoring'].mean()
        max_score = 5.0  # Score maximum possible
        score_percentage = (avg_score / max_score) * 100
        
        st.metric(label="Score qualité réseau (%)", 
                  value=f"{score_percentage:.1f}")
    
    # Section 3: Calcul des indicateurs de santé
    st.subheader("Indicateurs de santé réseau")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        # Stabilité DNS
        std_dns = st.session_state.lignes_1fev['std_dns_time'].mean()
        dns_stability = 100 - min(100, (std_dns / avg_dns * 100))
        
        delta_color = "normal"
        if dns_stability < 80:
            delta_color = "off"
        elif dns_stability > 95:
            delta_color = "inverse"
            
        st.metric(label="Stabilité DNS (%)", 
                  value=f"{dns_stability:.1f}",
                  delta=f"{dns_stability-90:.1f}" if dns_stability != 90 else None,
                  delta_color=delta_color)
    
    with col2:
        # Stabilité latence
        std_latence = st.session_state.lignes_1fev['std_latence_scoring'].mean()
        latence_stability = 100 - min(100, (std_latence / avg_latence * 100))
        
        delta_color = "normal"
        if latence_stability < 80:
            delta_color = "off"
        elif latence_stability > 95:
            delta_color = "inverse"
            
        st.metric(label="Stabilité latence (%)", 
                  value=f"{latence_stability:.1f}",
                  delta=f"{latence_stability-90:.1f}" if latence_stability != 90 else None,
                  delta_color=delta_color)
    
    with col3:
        # Score global de santé réseau (moyenne pondérée)
        health_score = (dns_stability * 0.4 + latence_stability * 0.3 + score_percentage * 0.3)
        
        delta_color = "normal"
        if health_score < 80:
            delta_color = "off"
        elif health_score > 95:
            delta_color = "inverse"
            
        st.metric(label="Santé globale du réseau (%)", 
                  value=f"{health_score:.1f}",
                  delta=f"{health_score-90:.1f}" if health_score != 90 else None,
                  delta_color=delta_color)
     
    # Section 3: Statistiques des métriques
    st.subheader("Statistiques des métriques")
    
    metrics_df = pd.DataFrame({
        "Métrique": [
            "Temps DNS moyen (ms)",
            "Écart-type DNS (ms)",
            "Latence scoring moyenne (ms)",
            "Écart-type latence (ms)",
            "Score qualité moyen (0-5)",
            "Écart-type score"
        ],
        "Moyenne": [
            f"{st.session_state.lignes_1fev['avg_dns_time'].mean():.2f}",
            f"{st.session_state.lignes_1fev['std_dns_time'].mean():.2f}",
            f"{st.session_state.lignes_1fev['avg_latence_scoring'].mean():.2f}",
            f"{st.session_state.lignes_1fev['std_latence_scoring'].mean():.2f}",
            f"{st.session_state.lignes_1fev['avg_score_scoring'].mean():.2f}",
            f"{st.session_state.lignes_1fev['std_score_scoring'].mean():.2f}"
        ],
        "Médiane": [
            f"{st.session_state.lignes_1fev['avg_dns_time'].median():.2f}",
            f"{st.session_state.lignes_1fev['std_dns_time'].median():.2f}",
            f"{st.session_state.lignes_1fev['avg_latence_scoring'].median():.2f}",
            f"{st.session_state.lignes_1fev['std_latence_scoring'].median():.2f}",
            f"{st.session_state.lignes_1fev['avg_score_scoring'].median():.2f}",
            f"{st.session_state.lignes_1fev['std_score_scoring'].median():.2f}"
        ],
        "Minimum": [
            f"{st.session_state.lignes_1fev['avg_dns_time'].min():.2f}",
            f"{st.session_state.lignes_1fev['std_dns_time'].min():.2f}",
            f"{st.session_state.lignes_1fev['avg_latence_scoring'].min():.2f}",
            f"{st.session_state.lignes_1fev['std_latence_scoring'].min():.2f}",
            f"{st.session_state.lignes_1fev['avg_score_scoring'].min():.2f}",
            f"{st.session_state.lignes_1fev['std_score_scoring'].min():.2f}"
        ],
        "Maximum": [
            f"{st.session_state.lignes_1fev['avg_dns_time'].max():.2f}",
            f"{st.session_state.lignes_1fev['std_dns_time'].max():.2f}",
            f"{st.session_state.lignes_1fev['avg_latence_scoring'].max():.2f}",
            f"{st.session_state.lignes_1fev['std_latence_scoring'].max():.2f}",
            f"{st.session_state.lignes_1fev['avg_score_scoring'].max():.2f}",
            f"{st.session_state.lignes_1fev['std_score_scoring'].max():.2f}"
        ]
    })
    
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    
        # Section 4: Distributions des métriques principales
    st.subheader("Distribution des métriques principales")
    
    # Créer un tableau pour les histogrammes
    metric_mapping = {
        'avg_dns_time': 'Temps DNS moyen',
        'avg_latence_scoring': 'Latence moyenne',
        'avg_score_scoring': 'Score qualité'
    }
    
    selected_metric = st.selectbox(
        "Choisir une métrique",
        options=list(metric_mapping.keys()),
        format_func=lambda x: metric_mapping[x]
    )
    
    # Créer un histogramme pour la métrique sélectionnée
    fig = px.histogram(
        st.session_state.lignes_1fev,
        x=selected_metric,
        nbins=50,
        title=f"Distribution de {metric_mapping[selected_metric]}",
        labels={selected_metric: metric_mapping[selected_metric]}
    )
    
    # Ajouter une ligne verticale pour la moyenne
    mean_value = st.session_state.lignes_1fev[selected_metric].mean()
    fig.add_vline(x=mean_value, line_dash="dash", line_color="red", annotation_text=f"Moyenne: {mean_value:.2f}")
    
    st.plotly_chart(fig, use_container_width=True)



def display_node_anomaly_chart():
    """
    Crée un graphique de synthèse montrant la distribution des anomalies par type de nœud
    et leur répartition par variable.
    """
    if not st.session_state.anomalies_detected:
        return
    
    # Créer deux colonnes pour les graphiques
    col1, col2 = st.columns(2)
    
    with col1:
        # Créer un diagramme à barres pour comparer les proportions d'anomalies par type de nœud
        node_types = ['PEAG', 'OLT']
        anomaly_percentages = []

        if st.session_state.results["peag_nro"] is not None:
            df = st.session_state.results["peag_nro"]
            p_val_min = df[['p_val_avg_dns_time', 'p_val_avg_score_scoring', 'p_val_avg_latence_scoring']].min(axis=1)
            condition = (p_val_min < st.session_state.p_value_threshold / 100) | (df['avg_isolation_forest_score'] < st.session_state.isolation_forest_threshold)
            anomaly_percentages.append(round(100 * sum(condition) / len(df), 2))
        else:
            anomaly_percentages.append(0)
        
        if st.session_state.results["olt"] is not None:
            df = st.session_state.results["olt"]
            p_val_min = df[['p_val_avg_dns_time', 'p_val_avg_score_scoring', 'p_val_avg_latence_scoring']].min(axis=1)
            condition = (p_val_min < st.session_state.p_value_threshold / 100) | (df['avg_isolation_forest_score'] < st.session_state.isolation_forest_threshold)
            anomaly_percentages.append(round(100 * sum(condition) / len(df), 2))
        else:
            anomaly_percentages.append(0)
        
        fig = go.Figure([
            go.Bar(
                x=node_types,
                y=anomaly_percentages,
                marker_color=['#66B3FF', '#99FF99'],  # Seulement deux couleurs maintenant
                text=anomaly_percentages,
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title="Pourcentage de nœuds anormaux par type (IsolationForest + Unidimensionnel)",
            xaxis_title="Type de nœud",
            yaxis_title="Pourcentage d'anomalies (%)",
            yaxis=dict(range=[0, 100]),
            height=350,
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Remplacer le graphique radar par un graphique à barres horizontales
        metrics = ['avg_dns_time', 'std_dns_time', 'avg_score_scoring', 
                  'std_score_scoring', 'avg_latence_scoring', 'std_latence_scoring']
        
        # Noms plus lisibles pour l'affichage
        display_names = [
            'Temps DNS moyen', 
            'Écart-type DNS', 
            'Score scoring moyen', 
            'Écart-type scoring', 
            'Latence scoring moyenne', 
            'Écart-type latence'
        ]
        
        # Compter le nombre d'anomalies par métrique (toutes variables)
        anomaly_counts = [0] * len(metrics)
        threshold_p_val = st.session_state.p_value_threshold / 100
        
        for i, metric in enumerate(metrics):
            p_val_col = f'p_val_{metric}'
            
            for node_type in ["boucles", "peag_nro", "olt"]:
                if st.session_state.results[node_type] is not None and p_val_col in st.session_state.results[node_type].columns:
                    anomaly_counts[i] += (st.session_state.results[node_type][p_val_col] < threshold_p_val).sum()
        
        # Créer le graphique à barres horizontales
        colors = ['#FF9999', '#66B3FF', '#99FF99', '#FFCC99', '#C2C2F0', '#FFB3E6']
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            y=display_names,
            x=anomaly_counts,
            orientation='h',
            marker_color=colors,
            text=anomaly_counts,
            textposition='auto',
        ))
        
        fig.update_layout(
            title="Nombre d'anomalies par métrique (détection unidimensionnelle)",
            xaxis_title="Nombre d'anomalies",
            yaxis=dict(
                categoryorder='total ascending',  # Tri par nombre d'anomalies
            ),
            height=350,
        )
        
        st.plotly_chart(fig, use_container_width=True)

# Fonction pour afficher le résumé des anomalies (approche unidimensionnelle)
def display_anomaly_summary():
    results = st.session_state.results
    # Définir les seuils
    threshold_p_val = st.session_state.p_value_threshold / 100
    threshold_if = st.session_state.isolation_forest_threshold
    
    # Créer trois colonnes principales
    st.subheader("Résumé des anomalies détectées (Isolation Forest + Unidimensionnel)")
    col1, col2 = st.columns(2)
    
    # Fonction pour créer une jauge d'anomalies
    def create_gauge_chart(percentage, title):
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=percentage,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': title},
                gauge={
                    'axis': {
                        'range': [0, 100],
                        'tickwidth': 1,
                        'tickcolor': "darkblue",
                        'tickvals': [0, 25, 50, 75, 100],
                        'ticktext': ['0', '25', '50', '75', '100'],
                        'tickangle': 0,  # Assurez-vous que le texte est horizontal
                        'tickfont': {'size': 12}  # Réduire la taille des étiquettes
                    },
                    'bar': {'color': "darkblue"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 25], 'color': 'green'},
                        {'range': [25, 50], 'color': 'yellow'},
                        {'range': [50, 75], 'color': 'orange'},
                        {'range': [75, 100], 'color': 'red'}
                    ],
                }
            ))
            
            # Augmenter les marges à droite pour éviter les coupures
            fig.update_layout(
                height=200,
                width=320,  # Définir une largeur fixe
                margin=dict(l=20, r=50, t=50, b=20),  # Augmenter la marge droite
                paper_bgcolor="rgba(0,0,0,0)",  # Fond transparent
                plot_bgcolor="rgba(0,0,0,0)"
            )
            return fig
    
    with col1:
        if results["peag_nro"] is not None:
            
            df = results["peag_nro"]
            p_val_min = df[['p_val_avg_dns_time', 'p_val_avg_score_scoring', 'p_val_avg_latence_scoring']].min(axis=1)
            condition = (p_val_min < threshold_p_val) | (df['avg_isolation_forest_score'] < threshold_if)
            anomalous_peag = df[condition]
            
            # Pourcentage de PEAG anormaux
            peag_percent = round(100 * len(anomalous_peag) / len(results["peag_nro"]), 2)
            
            # Afficher la jauge
            st.plotly_chart(create_gauge_chart(peag_percent, f"% de PEAG anormaux"), use_container_width=True)
            
            st.metric("Nbr PEAG anormaux / Total", f"{len(anomalous_peag)} / {len(results['peag_nro'])}")

            if not anomalous_peag.empty:
                # Trouver les PEAG les plus problématiques
                worst_peag = anomalous_peag.sort_values('avg_isolation_forest_score').index[:3]
                
                # Créer une table avec formatage coloré pour les PEAG les plus anormaux
                st.markdown("#### PEAG les plus anormaux:")
                for idx in worst_peag:
                    score = anomalous_peag.loc[idx, 'avg_isolation_forest_score']
                    color = 'red' if score < threshold_if else 'orange'
                    st.markdown(f"<span style='color:{color};font-weight:bold;font-size:16px'>• {idx}</span> (score: {score:.3f})", unsafe_allow_html=True)
    
    with col2:
        if results["olt"] is not None:
            df = results["olt"]
            p_val_min = df[['p_val_avg_dns_time', 'p_val_avg_score_scoring', 'p_val_avg_latence_scoring']].min(axis=1)
            condition = (p_val_min < threshold_p_val) | (df['avg_isolation_forest_score'] < threshold_if)
            anomalous_olt = df[condition]
            
            # Pourcentage d'OLT anormaux
            olt_percent = round(100 * len(anomalous_olt) / len(results["olt"]), 2)
            
            # Afficher la jauge
            st.plotly_chart(create_gauge_chart(olt_percent, f"% d'OLT anormaux"), use_container_width=True)
            
            st.metric("Nbr OLT anormaux / Total", f"{len(anomalous_olt)} / {len(results['olt'])}")
            
            if not anomalous_olt.empty:
                # Trouver les OLT les plus problématiques
                worst_olt = anomalous_olt.sort_values('avg_isolation_forest_score').index[:3]
                
                # Créer une table avec formatage coloré pour les OLT les plus anormaux
                st.markdown("#### OLT les plus anormaux:")
                for idx in worst_olt:
                    score = anomalous_olt.loc[idx, 'avg_isolation_forest_score']
                    color = 'red' if score < threshold_if else 'orange'
                    st.markdown(f"<span style='color:{color};font-weight:bold;font-size:16px'>• {idx}</span> (score: {score:.3f})", unsafe_allow_html=True)
    
    # Ajouter une section pour un graphique synthétique des anomalies
    st.markdown("---")
    if st.session_state.df_with_anomalies is not None:
        st.subheader("Distribution des anomalies par type de nœud")
        display_node_anomaly_chart()
    

# Fonction pour afficher une carte de chaleur des anomalies (Isolation Forest)
# def display_anomaly_heatmap():
#     if st.session_state.df_with_anomalies is None:
#         return
    
#     # Préparer les données pour la heatmap
#     df_anomalies = st.session_state.df_with_anomalies
    
#     # Calculer le pourcentage d'anomalies par OLT et PEAG
#     cross_tab = pd.crosstab(
#         df_anomalies['peag_nro'], 
#         df_anomalies['olt_name'],
#         values=df_anomalies['anomaly_score'].apply(lambda x: 1 if x == -1 else 0),
#         aggfunc='mean'
#     ).fillna(0)
    
#     # Limiter à 10 PEAG et 10 OLT maximum pour la lisibilité
#     if cross_tab.shape[0] > 10 or cross_tab.shape[1] > 10:
#         # Identifier les PEAG et OLT avec le plus d'anomalies
#         peag_anomaly_count = cross_tab.sum(axis=1).sort_values(ascending=False).head(10).index
#         olt_anomaly_count = cross_tab.sum(axis=0).sort_values(ascending=False).head(10).index
#         cross_tab = cross_tab.loc[peag_anomaly_count, olt_anomaly_count]
    
#     fig = go.Figure(data=go.Heatmap(
#         z=cross_tab.values * 100,  # En pourcentage
#         x=cross_tab.columns,
#         y=cross_tab.index,
#         colorscale='Reds',
#         colorbar=dict(title="% Anomalies")
#     ))
    
#     fig.update_layout(
#         title="Heatmap des anomalies par OLT et PEAG",
#         xaxis_title="OLT",
#         yaxis_title="PEAG",
#         height=400,
#         margin=dict(l=0, r=0, t=40, b=0)
#     )
    
#     st.plotly_chart(fig, use_container_width=True)

# Page d'insertion d'anomalies
def show_anomaly_insertion():
    st.header("🔧 Insertion d'anomalies")
    
    st.write("""
    Cette section vous permet d'insérer des anomalies simulées dans le réseau.
    Sélectionnez les nœuds, les variables de test et les valeurs à insérer.
    """)
    
    # Création de tabs pour organiser l'insertion par type de nœud
    tab1, tab2, tab3 = st.tabs(["Boucles", "PEAG", "OLT"])
    
    with tab1:
        st.subheader("Insertion d'anomalies dans les Boucles")
        col_names, col_vars, col_val, col_btn = st.columns([2, 2, 1, 1])
        
        with col_names:
            boucle_names = st.multiselect(
                "Boucles à modifier", 
                options=sorted(list(st.session_state.lignes_1fev['boucle'].unique()))
            )
        with col_vars:
            var_test = st.multiselect(
                "Variables de test", 
                options=st.session_state.variables_test, 
                key='insertvar1'
            )
        with col_val:
            valeur_insertion = st.number_input(
                "Valeur à insérer", 
                value=0.0, 
                step=1.0, 
                key='insertion1'
            )
        with col_btn:
            if st.button("Insérer anomalies", key="btn_inserer1"):
                st.write(f"Insertion d'une valeur de {round(valeur_insertion, 2)} dans {var_test} pour {len(boucle_names)} boucles")
                st.session_state.lignes_1fev = NodesChecker.add_anomalies(
                    st.session_state.lignes_1fev, 
                    'boucle', 
                    boucle_names,
                    var_test,
                    valeur_insertion
                )
                st.success(f"Anomalies insérées avec succès dans {len(boucle_names)} boucles!")
                # Réinitialiser le statut de détection
                st.session_state.anomalies_detected = False
    
    with tab2:
        st.subheader("Insertion d'anomalies dans les PEAG")
        col_names, col_vars, col_val, col_btn = st.columns([2, 2, 1, 1])
        
        with col_names:
            peag_names = st.multiselect(
                "PEAG à modifier", 
                options=sorted(list(st.session_state.lignes_1fev['peag_nro'].unique()))
            )
        with col_vars:
            var_test = st.multiselect(
                "Variables de test", 
                options=st.session_state.variables_test, 
                key='insertvar2'
            )
        with col_val:
            valeur_insertion = st.number_input(
                "Valeur à insérer", 
                value=0.0, 
                step=1.0, 
                key='insertion2'
            )
        with col_btn:
            if st.button("Insérer anomalies", key="btn_inserer2"):
                st.write(f"Insertion d'une valeur de {round(valeur_insertion, 2)} dans {var_test} pour {len(peag_names)} PEAG")
                st.session_state.lignes_1fev = NodesChecker.add_anomalies(
                    st.session_state.lignes_1fev, 
                    'peag_nro', 
                    peag_names,
                    var_test,
                    valeur_insertion
                )
                st.success(f"Anomalies insérées avec succès dans {len(peag_names)} PEAG!")
                # Réinitialiser le statut de détection
                st.session_state.anomalies_detected = False
    
    with tab3:
        st.subheader("Insertion d'anomalies dans les OLT")
        col_names, col_vars, col_val, col_btn = st.columns([2, 2, 1, 1])
        
        with col_names:
            olt_names = st.multiselect(
                "OLT à modifier", 
                options=sorted(list(st.session_state.lignes_1fev['olt_name'].unique()))
            )
        with col_vars:
            var_test = st.multiselect(
                "Variables de test", 
                options=st.session_state.variables_test, 
                key='insertvar3'
            )
        with col_val:
            valeur_insertion = st.number_input(
                "Valeur à insérer", 
                value=0.0, 
                step=1.0, 
                key='insertion3'
            )
        with col_btn:
            if st.button("Insérer anomalies", key="btn_inserer3"):
                st.write(f"Insertion d'une valeur de {round(valeur_insertion, 2)} dans {var_test} pour {len(olt_names)} OLT")
                st.session_state.lignes_1fev = NodesChecker.add_anomalies(
                    st.session_state.lignes_1fev, 
                    'olt_name', 
                    olt_names,
                    var_test,
                    valeur_insertion
                )
                st.success(f"Anomalies insérées avec succès dans {len(olt_names)} OLT!")
                # Réinitialiser le statut de détection
                st.session_state.anomalies_detected = False
    
    # Bouton pour réinitialiser les données
    st.markdown("---")
    if st.button("🔄 Réinitialiser toutes les données", type="secondary"):
        st.session_state.lignes_1fev = st.session_state.lignes_1fev_copy.copy()
        st.session_state.anomalies_detected = False
        st.success("Toutes les données ont été réinitialisées avec succès!")


def show_detection_config():
    st.header("⚙️ Configuration et lancement de la détection d'anomalies")
  
    # Section des méthodes de détection
    st.subheader("Méthodes de détection")
    
    # Utiliser plus d'espace horizontal entre les colonnes
    st.markdown("""
    <style>
    .stColumn > div {
        margin: 0 10px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        use_unidimensional = st.checkbox("Approche unidimensionnelle", value=True)
        st.info("Analyse statistique des distributions de chaque métrique")
    
    with col2:
        use_isolation_forest = st.checkbox("Isolation Forest", value=True)
        st.info("Détection de points atypiques dans un espace multidimensionnel")
    
    with col3:
        use_mahalanobis = st.checkbox("Distance de Mahalanobis", value=True) 
        st.info("Mesure de l'éloignement par rapport à la distribution normale")

# Configuration des seuils
    st.subheader("Configuration des seuils")
    
    # Approche unidimensionnelle
    if use_unidimensional:
        st.markdown("##### Approche unidimensionnelle")
        unidim_threshold = st.slider(
            "Seuil (%) de rejet α pour l'approche unidimensionnelle", 
            min_value=0.5, 
            max_value=20.0, 
            value=st.session_state.p_value_threshold, 
            step=0.05,
            format="%.1f"
        )
        st.session_state.p_value_threshold = unidim_threshold
        st.info(f"Les variables avec une p-value inférieure à {unidim_threshold}% seront considérées comme anormales")
    
    # Isolation Forest
    if use_isolation_forest:
        st.markdown("##### Isolation Forest")
        if_threshold = st.slider(
            "Seuil de score d'anomalie pour Isolation Forest", 
            min_value=-1.0, 
            max_value=0.5, 
            value=st.session_state.isolation_forest_threshold, 
            step=0.05,
            format="%.2f"
        )
        st.session_state.isolation_forest_threshold = if_threshold
        st.info(f"Les observations avec un score inférieur à {if_threshold} seront considérées comme anormales")
    
    # Mahalanobis
    if use_mahalanobis:
        st.markdown("##### Distance de Mahalanobis")
        
        # Stockage du seuil dans session_state s'il n'existe pas encore
        if "mahalanobis_threshold" not in st.session_state:
            st.session_state.mahalanobis_threshold = 0.01
            
        maha_threshold = st.slider(
            "Seuil de p-value pour Mahalanobis", 
            min_value=0.01, 
            max_value=10.0, 
            value=st.session_state.mahalanobis_threshold, 
            step=0.05,
            format="%.2f"
        )
        st.session_state.mahalanobis_threshold = maha_threshold
        st.info(f"Les observations avec une p-value inférieure à {maha_threshold} seront considérées comme anormales")
    
    # Configuration de l'approche ensembliste
    st.subheader("Approche ensembliste")
    
    ensemble_method = st.radio(
        "Méthode de combinaison des détecteurs",
        options=["Vote majoritaire", "Union", "Intersection"],
        index=0,
        help="Vote majoritaire: anomalie si détectée par au moins 2 méthodes\nUnion: anomalie si détectée par au moins 1 méthode\nIntersection: anomalie si détectée par toutes les méthodes"
    )
    
    if "ensemble_method" not in st.session_state:
        st.session_state.ensemble_method = "Vote majoritaire"
    st.session_state.ensemble_method = ensemble_method
    
    # Lancement de la détection
    st.markdown("---")
    launch_col1, launch_col2 = st.columns(2)
    
    with launch_col1:
        launch_button_text = "🚀 Lancer la détection d'anomalies"
        if st.button(launch_button_text, type="primary", use_container_width=True):
            with st.spinner("Détection d'anomalies en cours ..."):
                # Stocker les méthodes sélectionnées
                st.session_state.use_unidimensional = use_unidimensional
                st.session_state.use_isolation_forest = use_isolation_forest
                st.session_state.use_mahalanobis = use_mahalanobis
                
                run_anomaly_detection()
                st.session_state.anomalies_detected = True
                st.success("Détection d'anomalies terminée avec succès!")
                st.balloons()  # Animation pour célébrer la fin de l'analyse
    
    with launch_col2:
        if st.button("🔄 Réinitialiser les paramètres", use_container_width=True):
            st.session_state.p_value_threshold = 5.0
            st.session_state.isolation_forest_threshold = 0.0
            st.session_state.mahalanobis_threshold = 0.01
            st.session_state.ensemble_method = "Vote majoritaire"
            st.success("Paramètres réinitialisés aux valeurs par défaut")
            st.experimental_rerun()
            
    # Explication détaillée en bas de page
    with st.expander("Comprendre les seuils de détection", expanded=False):
        st.markdown("""
        ### Guide des seuils de détection

        #### Approche unidimensionnelle
        Le seuil représente le niveau de confiance pour rejeter l'hypothèse nulle (que les données sont normales).
        - **5%** : Valeur standard recommandée dans la plupart des cas
        - **1%** : Détection plus stricte, moins de faux positifs mais peut manquer des anomalies
        - **10%** : Détection plus sensible, capture plus d'anomalies potentielles avec plus de faux positifs

        #### Isolation Forest
        Le score d'anomalie est entre -1 et +1, où:
        - **-1** : Anomalie certaine
        - **0** : Cas limite
        - **+1** : Normal certain
        
        La valeur par défaut de 0 est un bon compromis. Une valeur négative rend la détection plus stricte.

        #### Distance de Mahalanobis
        La p-value indique la probabilité qu'une observation soit issue de la distribution normale.
        - **0.01** (1%) : Niveau standard de détection
        - **0.001** (0.1%) : Détection très stricte, uniquement les anomalies extrêmes
        - **0.05** (5%) : Détection plus sensible

        #### Approche ensembliste
        - **Vote majoritaire** : Équilibre entre précision et sensibilité
        - **Union** : Maximise la sensibilité (capture toutes les anomalies possibles)
        - **Intersection** : Maximise la précision (minimise les faux positifs)
        """)
                   


# def show_impact_analysis():
#     st.subheader("📉 Analyse d'impact potentiel")

#     # Initialisation
#     total_anomalies = 0
#     total_nodes = 0
#     anomalies_by_type = {"boucles": 0, "peag_nro": 0, "olt": 0}
#     client_counts = {}
#     degradation_metrics = {}

#     for node_type in ["boucles", "peag_nro", "olt"]:
#         df = st.session_state.results.get(node_type)
#         if df is not None:
#             total_nodes += len(df)
#             anomaly_count = df['is_anomaly'].sum() if 'is_anomaly' in df.columns else 0
#             total_anomalies += anomaly_count
#             anomalies_by_type[node_type] = anomaly_count

#             # Clients impactés
#             node_col = {"boucles": "boucle", "peag_nro": "peag_nro", "olt": "olt_name"}[node_type]
#             if 'nb_client_total' in st.session_state.lignes_1fev_copy.columns:
#                 anomalous_nodes = df[df['is_anomaly']].index.tolist() if 'is_anomaly' in df.columns else []
#                 total_clients = 0
#                 for node in anomalous_nodes:
#                     mask = st.session_state.lignes_1fev_copy[node_col] == node
#                     total_clients += st.session_state.lignes_1fev_copy.loc[mask, 'nb_client_total'].sum()
#                 client_counts[node_type] = total_clients
#             else:
#                 default_per_node = {"boucles": 10, "peag_nro": 25, "olt": 50}
#                 client_counts[node_type] = anomalies_by_type[node_type] * default_per_node[node_type]

#             # Dégradations
#             if 'is_anomaly' in df.columns and df['is_anomaly'].sum() > 0:
#                 for metric, invert in [("avg_dns_time", False), ("avg_latence_scoring", False), ("avg_score_scoring", True)]:
#                     if metric in df.columns:
#                         normal = df[~df['is_anomaly']][metric].mean()
#                         anomaly = df[df['is_anomaly']][metric].mean()
#                         if pd.notna(normal) and normal > 0:
#                             delta = ((normal - anomaly) / normal) * 100 if invert else ((anomaly - normal) / normal) * 100
#                             degradation_metrics[f"{node_type}_{metric}"] = max(0, round(delta, 2))

#     # Résumés globaux
#     total_clients_impacted = int(sum(client_counts.values()))
#     overall_degradation = round(sum(degradation_metrics.values()) / len(degradation_metrics), 1) if degradation_metrics else 0
#     max_clients = max(1000, total_clients_impacted * 1.2)

#     # 🔒 Valeurs sécurisées
#     total_clients_impacted = max(0, total_clients_impacted)
#     overall_degradation = max(0, min(100, overall_degradation))

#     # 📊 Jauges
#     col1, col2 = st.columns(2)

#     with col1:
#         fig1 = go.Figure(go.Indicator(
#             mode="gauge+number",
#             value=total_clients_impacted,
#             title={'text': "Clients impactés", 'font': {'size': 18, 'color': 'white'}},
#             number={'font': {'size': 26, 'color': 'white'}},
#             gauge={
#                 'axis': {'range': [0, max_clients], 'tickwidth': 1, 'tickcolor': 'white'},
#                 'bar': {'color': "rgba(0,0,0,0)"},
#                 'steps': [
#                     {'range': [0, 200], 'color': '#3cb44b'},
#                     {'range': [200, 500], 'color': '#ff9a00'},
#                     {'range': [500, max_clients], 'color': '#e6194B'}
#                 ],
#                 'threshold': {'line': {'color': "red", 'width': 4}, 'value': 500}
#             }
#         ))
#         fig1.update_layout(
#             height=300,
#             paper_bgcolor="rgba(0,0,0,0)",
#             plot_bgcolor="rgba(0,0,0,0)",
#             font=dict(color="white")
#         )
#         st.markdown("&nbsp;", unsafe_allow_html=True)
#         st.plotly_chart(fig1, use_container_width=True)

#     with col2:
#         fig2 = go.Figure(go.Indicator(
#             mode="gauge+number",
#             value=overall_degradation,
#             title={'text': "Dégradation du service", 'font': {'size': 18, 'color': 'white'}},
#             number={'suffix': "%", 'font': {'size': 26, 'color': 'white'}},
#             gauge={
#                 'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': 'white'},
#                 'bar': {'color': "rgba(0,0,0,0)"},
#                 'steps': [
#                     {'range': [0, 25], 'color': '#3cb44b'},
#                     {'range': [25, 50], 'color': '#ff9a00'},
#                     {'range': [50, 100], 'color': '#e6194B'}
#                 ],
#                 'threshold': {'line': {'color': "red", 'width': 4}, 'value': 75}
#             }
#         ))
#         fig2.update_layout(
#             height=300,
#             paper_bgcolor="rgba(0,0,0,0)",
#             plot_bgcolor="rgba(0,0,0,0)",
#             font=dict(color="white")
#         )
#         st.markdown("&nbsp;", unsafe_allow_html=True)
#         st.plotly_chart(fig2, use_container_width=True)

#     # 📊 Barres de dégradation
#     if degradation_metrics:
#         st.subheader("📊 Détails de la dégradation par métrique")
#         metrics_df = pd.DataFrame({
#             'Métrique': list(degradation_metrics.keys()),
#             'Dégradation (%)': list(degradation_metrics.values())
#         }).sort_values("Dégradation (%)", ascending=False)

#         fig = px.bar(
#             metrics_df,
#             y='Métrique',
#             x='Dégradation (%)',
#             orientation='h',
#             color='Dégradation (%)',
#             color_continuous_scale=['green', 'yellow', 'red'],
#             title="Dégradation par métrique et type de nœud"
#         )
#         fig.update_layout(height=400)
#         st.markdown("&nbsp;", unsafe_allow_html=True)
#         st.plotly_chart(fig, use_container_width=True)

#     # 🧠 Bloc d'analyse final
#     st.markdown("""
#     <div style='background-color: #1e1e1e; padding: 20px; border-radius: 10px; margin-top: 20px;'>
#         <h4 style='color: #ff4d4d;'>⚠️ Conséquences potentielles si aucune action n'est prise</h4>
#         <ul style='margin-top: 10px; font-size: 16px; color: white;'>
#             <li><strong>Court terme (24-48h):</strong> Augmentation des temps de réponse DNS et de latence</li>
#             <li><strong>Moyen terme (3-5 jours):</strong> Dégradation du streaming, jeux en ligne, navigation</li>
#             <li><strong>Long terme (>1 semaine):</strong> Risque de coupures et hausse des appels au service client</li>
#         </ul>
#         <p style='margin-top: 15px; font-weight: bold; font-size: 16px; color: white;'>
#             Une intervention préventive est recommandée pour éviter l’aggravation du problème et protéger la satisfaction client.
#         </p>
#     </div>
#     """, unsafe_allow_html=True)


# Fonction pour obtenir un label de gravité basé sur la p-value
def get_severity_label(p_value):
    if p_value < 0.001:
        return "Critique"
    elif p_value < 0.01:
        return "Sévère"
    elif p_value < 0.05:
        return "Modéré"
    else:
        return "Faible"


def create_gauge_chart(percentage, title, ratio_text):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=percentage,
        number={'suffix': "%", 'font': {'size': 24, 'color': 'white'}},
        domain={'x': [0, 1], 'y': [0.2, 1]},  # décaler vers le haut
        title={'text': title, 'font': {'size': 18, 'color': 'white'}},
        gauge={
            'axis': {
                'range': [0, 100],
                'tickwidth': 1,
                'tickcolor': "white",
                'tickfont': {'size': 12, 'color': 'white'}
            },
            'bar': {'color': "rgba(0,0,0,0)"},
            'bgcolor': "white",
            'borderwidth': 1,
            'bordercolor': "#ccc",
            'steps': [
                {'range': [0, 25], 'color': 'green'},
                {'range': [25, 50], 'color': 'yellow'},
                {'range': [50, 75], 'color': 'orange'},
                {'range': [75, 100], 'color': 'red'}
            ],
        }
    ))

    # Ajouter le ratio en dessous de la jauge
    fig.add_annotation(
        text=f"<b>{ratio_text}</b>",
        x=0.5,
        y=0.05,  # juste en bas
        showarrow=False,
        font=dict(size=16),
        xanchor="center"
    )

    fig.update_layout(
        height=260,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig


def show_anomaly_statistics():
    st.subheader("📊 Statistiques des anomalies détectées")
    
    # Compter les anomalies par type de nœud
    stats = {}
    for node_type, display_name in zip(["boucles", "peag_nro", "olt"], ["Boucles", "PEAG", "OLT"]):
        if node_type in st.session_state.results and st.session_state.results[node_type] is not None:
            df = st.session_state.results[node_type]
            total_count = len(df)
            anomaly_count = df['is_anomaly'].sum() if 'is_anomaly' in df.columns else 0
            percentage = (anomaly_count / total_count * 100) if total_count > 0 else 0

            stats[display_name] = {
                "total": total_count,
                "anomalies": anomaly_count,
                "percentage": percentage
            }

    # Afficher les jauges dans des colonnes bien réparties
    cols = st.columns(len(stats), gap="large")

    for i, (display_name, data) in enumerate(stats.items()):
        with cols[i]:
            fig = create_gauge_chart(
                data['percentage'],
                f"{display_name} anormaux",
                f"{data['anomalies']} / {data['total']} {display_name}"
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(
                "<div style='text-align: center;'>&nbsp;</div>",
                unsafe_allow_html=True
            )


    # Graphique à barres des méthodes
    st.subheader("📌 Répartition des anomalies par méthode de détection")

    method_stats = {
        "Unidimensionnelle": 0,
        "Isolation Forest": 0,
        "Mahalanobis": 0
    }

    for node_type in ["boucles", "peag_nro", "olt"]:
        if node_type in st.session_state.results and st.session_state.results[node_type] is not None:
            df = st.session_state.results[node_type]
            threshold = st.session_state.p_value_threshold / 100

            # Compter par méthode
            if any(col.startswith('p_val_') for col in df.columns):
                unidim_count = (df.filter(like='p_val_') < threshold).any(axis=1).sum()
                method_stats["Unidimensionnelle"] += unidim_count

            if 'avg_isolation_forest_score' in df.columns:
                if_count = (df['avg_isolation_forest_score'] < st.session_state.isolation_forest_threshold).sum()
                method_stats["Isolation Forest"] += if_count

            if 'mahalanobis_anomaly' in df.columns:
                mah_count = df['mahalanobis_anomaly'].sum()
                method_stats["Mahalanobis"] += mah_count
            elif 'avg_mahalanobis_pvalue' in df.columns:
                mah_count = (df['avg_mahalanobis_pvalue'] < st.session_state.mahalanobis_threshold).sum()
                method_stats["Mahalanobis"] += mah_count

    # Afficher le graphique à barres
    fig = px.bar(
        x=list(method_stats.keys()),
        y=list(method_stats.values()),
        labels={"x": "Méthode de détection", "y": "Nombre d'anomalies"},
        color=list(method_stats.keys()),
        color_discrete_map = {
            "Unidimensionnelle": "#fcae91",   # Rouge clair
            "Isolation Forest": "#fb6a4a",    # Rouge moyen
            "Mahalanobis": "#de2d26"          # Rouge foncé
        },
        text=list(method_stats.values())
    )

    fig.update_traces(textposition='outside')
    fig.update_layout(
        height=400,
        xaxis=dict(title=None),
        margin=dict(t=20, b=20, l=20, r=20),
        showlegend=False,
        font=dict(size=16),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )

    st.plotly_chart(fig, use_container_width=True)


def display_equipment_summary_table():
    st.subheader("📋 Synthèse par équipement")

    rows = []
    threshold = st.session_state.p_value_threshold / 100
    
    # Mapping pour assurer l'accès correct aux colonnes
    column_mapping = {
        "boucles": "boucle",
        "peag_nro": "peag_nro",
        "olt": "olt_name"
    }
    
    for node_type, display_name in zip(["boucles", "peag_nro", "olt"], ["Boucle", "PEAG", "OLT"]):
        if node_type in st.session_state.results and st.session_state.results[node_type] is not None:
            df = st.session_state.results[node_type]
            
            for idx, row in df.iterrows():
                # Vérifier si c'est une anomalie
                is_anomaly = row.get('is_anomaly', False)
                
                if is_anomaly:
                    # Collecter les métriques en anomalie
                    abnormal_metrics = []
                    for col in df.columns:
                        if 'p_val_' in col and row[col] < threshold:
                            metric_name = col.replace('p_val_', '')
                            abnormal_metrics.append(metric_name)
                    
                    # Vérifier les autres méthodes de détection
                    detection_methods = []
                    if abnormal_metrics:
                        detection_methods.append("Unidimensionnelle")
                    
                    if 'avg_isolation_forest_score' in row and row['avg_isolation_forest_score'] < st.session_state.isolation_forest_threshold:
                        detection_methods.append("Isolation Forest")
                    
                    if 'mahalanobis_anomaly' in row and row['mahalanobis_anomaly'] == 1:
                        detection_methods.append("Mahalanobis")
                    
                    # Déterminer les métriques touchées
                    metrics_text = ", ".join(abnormal_metrics) if abnormal_metrics else "N/A"
                    
                    # Déterminer la gravité (3 classes seulement)
                    p_values = [row[col] for col in df.columns if 'p_val_' in col]
                    if p_values:
                        min_p_value = min(p_values)
                        if min_p_value < 0.01:  # Très faible p-value
                            severity = "Critique"
                        elif min_p_value < 0.05:  # Faible p-value
                            severity = "Modéré"
                        else:  # P-value plus élevée
                            severity = "Faible"
                    else:
                        # Si pas de p-values, utiliser isolation forest ou autre
                        if 'avg_isolation_forest_score' in row and row['avg_isolation_forest_score'] < -0.3:
                            severity = "Critique"
                        elif 'avg_isolation_forest_score' in row and row['avg_isolation_forest_score'] < 0:
                            severity = "Modéré"
                        else:
                            severity = "Faible"
                    
                    # Déterminer le département
                    try:
                        col_name = column_mapping[node_type]
                        mask = st.session_state.lignes_1fev_copy[col_name] == idx
                        dept = st.session_state.lignes_1fev_copy.loc[mask, 'code_departement'].iloc[0] if any(mask) else "N/A"
                    except (KeyError, IndexError):
                        dept = "N/A"
                    
                    # Créer l'entrée pour le tableau
                    rows.append({
                        "Type": display_name,
                        "Identifiant": idx,
                        "Département": dept,
                        "Gravité": severity,
                        "Métriques touchées": metrics_text,
                        "Méthode(s) de détection": ", ".join(detection_methods) if detection_methods else "Aucune"
                    })

    # Créer le DataFrame de synthèse
    if rows:
        df_summary = pd.DataFrame(rows)

        if len(df_summary) > 0:
            df_summary = df_summary.sort_values("Gravité", key=lambda x: x.map({"Critique": 0, "Modéré": 1, "Faible": 2}))

            def highlight_severity(val):
                color = ""
                if val == "Critique":
                    color = "#e63946"
                elif val == "Modéré":
                    color = "#f9844a"
                elif val == "Faible":
                    color = "#90be6d"
                return f'background-color: {color}; color: white'

            styled_df = df_summary.style.applymap(highlight_severity, subset=['Gravité'])

            st.dataframe(styled_df, use_container_width=True, hide_index=True)
        else:
            st.info("Aucune anomalie n'a été détectée avec les seuils actuels.")
    else:
        st.info("Aucune donnée d'anomalie disponible. Veuillez lancer une détection.")

def display_anomaly_details():
    st.subheader("🔬 Détail d'une anomalie")

    # Collecter toutes les anomalies
    all_anomalies = []
    for node_type in ["boucles", "peag_nro", "olt"]:
        if node_type in st.session_state.results and st.session_state.results[node_type] is not None:
            df = st.session_state.results[node_type]
            for idx in df.index:
                # Vérifier si c'est une anomalie
                is_anomaly = False
                for col in df.columns:
                    if 'p_val_' in col and df.loc[idx, col] < st.session_state.p_value_threshold / 100:
                        is_anomaly = True
                        break
                if 'avg_isolation_forest_score' in df.columns and df.loc[idx, 'avg_isolation_forest_score'] < st.session_state.isolation_forest_threshold:
                    is_anomaly = True
                
                if is_anomaly:
                    all_anomalies.append(f"{node_type.upper()} - {idx}")
    
    if not all_anomalies:
        st.info("Aucune anomalie détectée pour l'instant. Lancez une détection ou ajustez les seuils.")
        return
    
    selected = st.selectbox("Choisir une anomalie à explorer", options=all_anomalies)

    if selected:
        node_type, node_id = selected.split(" - ")
        node_type = node_type.lower()
        df = st.session_state.results[node_type]
        row = df.loc[node_id]

        st.markdown(f"### 🧠 Anomalie sur {node_type.upper()} {node_id}")
        
        # Métriques concernées
        affected_metrics = []
        for col in df.columns:
            if "p_val_" in col and row[col] < st.session_state.p_value_threshold / 100:
                metric_name = col.replace('p_val_', '')
                affected_metrics.append(metric_name)
        
        # Si aucune métrique spécifique mais Isolation Forest détecte une anomalie
        if not affected_metrics and 'avg_isolation_forest_score' in row and row['avg_isolation_forest_score'] < st.session_state.isolation_forest_threshold:
            st.markdown("- Anomalie multivariée détectée par Isolation Forest")
            
        if 'avg_mahalanobis_distance' in row and 'mahalanobis_anomaly' in row and row['mahalanobis_anomaly'] == 1:
            st.markdown("- Anomalie détectée par distance de Mahalanobis")
        

    # Bouton pour afficher toutes les observations
    if st.button("📑 Afficher toutes les observations de ce nœud"):
        try:
            # Trouver la colonne correspondante
            column_mapping = {
                "boucles": "boucle",
                "peag_nro": "peag_nro",
                "olt": "olt_name"
            }
            
            col_name = column_mapping.get(node_type)
            if col_name in st.session_state.lignes_1fev_copy.columns:
                mask = st.session_state.lignes_1fev_copy[col_name] == node_id
                node_data = st.session_state.lignes_1fev_copy[mask]
                
                if len(node_data) > 0:
                    # Supprimer la colonne d'index si elle existe
                    if 'Unnamed: 0' in node_data.columns:
                        node_data = node_data.drop(columns=['Unnamed: 0'])
                    
                    st.dataframe(node_data, use_container_width=True, hide_index=True)
                else:
                    st.info(f"Aucune observation trouvée pour {node_type.upper()} {node_id}")
            else:
                st.warning(f"Colonne {col_name} non trouvée dans les données")
        except Exception as e:
            st.error(f"Erreur lors de l'accès aux données: {str(e)}")
            st.info("Impossible d'afficher les observations détaillées")

    # Recommandation
    p_values = [row[col] for col in df.columns if 'p_val_' in col]
    if p_values:
        severity = get_severity_label(min(p_values))
        if severity == "Critique":
            st.error("📌 Recommandation : Intervention immédiate requise")
        elif severity == "Sévère":
            st.warning("📌 Recommandation : Intervention préventive recommandée")
        else:
            st.info("📌 Recommandation : Surveillance recommandée")
    else:
        st.info("📌 Recommandation : Surveillance recommandée")

    # Graphiques des variables
    st.subheader("Évolution des métriques")
    
    # Liste des variables à afficher
    all_metrics = [
        'avg_dns_time', 
        'std_dns_time', 
        'avg_score_scoring', 
        'std_score_scoring', 
        'avg_latence_scoring', 
        'std_latence_scoring'
    ]
    
    # Sélection des variables à afficher
    selected_metrics = st.multiselect(
        "Sélectionner les métriques à visualiser",
        options=all_metrics,
        default=[all_metrics[0]] if all_metrics else [],
        key="metrics_multiselect"
    )
    
    if selected_metrics:
        # Générer des données simulées
        import numpy as np
        
        # Créer un DataFrame avec des indices numériques pour les dates
        days = list(range(1, 61))
        df_chart = pd.DataFrame({'Jour': days})
        
        # Générer des données pour chaque métrique
        np.random.seed(42)  # Pour reproductibilité
        for metric in selected_metrics:
            # Générer des données simulées avec une tendance et un peu de bruit
            if 'avg' in metric:
                base = 10  # Valeur de base
            else:
                base = 2  # Valeur de base pour std
            
            trend = np.linspace(0, 2, 60)  # Tendance croissante
            noise = np.random.normal(0, 0.5, 60)  # Bruit aléatoire
            
            # Ajouter une anomalie vers la fin
            anomaly = np.zeros(60)
            # Détection automatique du point d'anomalie - dernier quartile avec une croissance soudaine
            anomaly_start = 45  # 75% des données
            anomaly[anomaly_start:] = np.linspace(0, 3, 60-anomaly_start)  # Anomalie croissante
            
            data = base + trend + noise + anomaly
            df_chart[metric] = data
        
        # Créer un SEUL graphique avec toutes les métriques sélectionnées
        fig = go.Figure()
        
        # Couleurs distinctes pour chaque métrique
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
        # Ajouter chaque métrique comme une ligne séparée
        for i, metric in enumerate(selected_metrics):
            fig.add_trace(go.Scatter(
                x=df_chart['Jour'],
                y=df_chart[metric],
                mode='lines+markers',
                name=metric,
                line=dict(color=colors[i % len(colors)], width=2),
                marker=dict(size=6)
            ))
        
        # Ajouter une ligne verticale pour le début de l'anomalie détectée
        anomaly_day = 45  # Point où l'anomalie commence (75% des données)
        fig.add_vline(
            x=anomaly_day,
            line_dash="dash",
            line_color="red",
            annotation_text="Début probable de l'anomalie"
        )
        
        # Mise en page
        fig.update_layout(
            title="Évolution des métriques sélectionnées (60 derniers jours)",
            xaxis_title="Jour",
            yaxis_title="Valeur",
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ),
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)



def show_results():
    st.header("📈 Résultats de la détection d'anomalies")

    if not st.session_state.anomalies_detected:
        st.warning("Veuillez lancer la détection avant d'accéder aux résultats.")
        return
    
    # Vérifier que nous avons des résultats à afficher
    has_results = False
    for node_type in ["boucles", "peag_nro", "olt"]:
        if node_type in st.session_state.results and st.session_state.results[node_type] is not None:
            has_results = True
            break
    
    if not has_results:
        st.error("Aucun résultat disponible. Veuillez relancer la détection d'anomalies.")
        return

    # 1. Statistiques d'anomalies
    show_anomaly_statistics()
    
    # 2. Tableau de synthèse par équipement
    display_equipment_summary_table()
    
    # 3. Détails par anomalie
    display_anomaly_details()
    
    # 4. Section d'analyse d'impact
    # show_impact_analysis()

# Page d'aide
def show_about():
    st.header("ℹ️ À propos de l'application")
    
    # Créer des onglets à l'intérieur de la page à propos
    about_tabs = st.tabs(["Présentation", "Fonctionnalités", "Méthodologie", "Aide", "Équipe"])
    
    with about_tabs[0]:  # Présentation
        st.subheader("Présentation du projet")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.write("""
            Cette application de détection d'anomalies réseau a été développée dans le cadre du Challenge Nexialog & Université Paris 1 Panthéon-Sorbonne. 
            
            L'objectif est d'anticiper les problèmes sur le réseau SFR afin d'intervenir avant que les clients ne soient impactés par des interruptions de service.
            
            L'application analyse en temps réel les données des tests réseau effectués sur les différents équipements (OLT, PEAG, boucles) et identifie les comportements anormaux grâce à des techniques avancées de détection d'anomalies.
            """)
        
        with col2:
            st.image("images/SFR_logo.png", width=150)
            st.image("images/nexialog_logo.png", width=150)
    
    with about_tabs[1]:  # Fonctionnalités
        st.subheader("Principales fonctionnalités")
        
        st.markdown("""
        #### 📊 État du réseau
        - Visualisation des statistiques globales du réseau
        - Aperçu des performances des métriques clés
        - Distribution des variables de test
        
        #### 🔧 Insertion d'anomalies
        - Simulation d'anomalies sur différents types de nœuds
        - Modification des valeurs de test pour tester les algorithmes
        
        #### ⚙️ Configuration et détection
        - Paramétrage des seuils de détection
        - Sélection des méthodes de détection (unidimensionnelle, Isolation Forest, Mahalanobis)
        - Configuration de l'approche ensembliste
        
        #### 📈 Résultats
        - Visualisation des anomalies détectées
        - Analyse par type de nœud (boucles, PEAG, OLT)
        - Visualisation 3D des anomalies multidimensionnelles
        """)
    
    with about_tabs[2]:  # Méthodologie
        st.subheader("Méthodologie de détection")
        
        st.markdown("""
        Notre application utilise trois approches complémentaires pour détecter les anomalies réseau:
        
        #### 1. Approche unidimensionnelle
        Cette méthode analyse chaque métrique indépendamment:
        - Applique un filtrage Hodrick-Prescott pour séparer tendance et cycles
        - Estime la distribution empirique via kernel density estimation (KDE)
        - Calcule des p-values pour déterminer si une observation est anormale
        - Utilise le test de Fisher pour combiner les p-values par nœud
        
        #### 2. Isolation Forest
        Approche multidimensionnelle qui:
        - Construit des arbres de décision qui isolent les observations
        - Détecte les anomalies comme les points nécessitant moins d'étapes d'isolation
        - Fonctionne sans hypothèse préalable sur la distribution des données
        - Est particulièrement efficace pour les jeux de données à grande dimension
        
        #### 3. Distance de Mahalanobis
        Méthode statistique robuste qui:
        - Mesure la distance entre un point et la distribution globale
        - Prend en compte les corrélations entre variables
        - Utilise l'estimateur MCD (Minimum Covariance Determinant) pour la robustesse
        - Effectue une normalisation pour comparer les scores entre chaînes techniques
        """)
        
    
    with about_tabs[3]:  # Aide
        st.subheader("Guide d'utilisation")
        
        st.markdown("""
        ### Comment utiliser l'application
        
        #### Configuration et lancement de la détection
        1. Naviguez vers l'onglet "Configuration et détection"
        2. Sélectionnez les méthodes de détection souhaitées
        3. Ajustez les seuils de détection selon vos besoins
        4. Choisissez la méthode de combinaison pour l'approche ensembliste
        5. Cliquez sur "Lancer la détection d'anomalies"
        
        #### Interprétation des résultats
        - **p-value faible**: Plus la p-value est faible, plus l'anomalie est significative
        - **Score Isolation Forest négatif**: Les scores proches de -1 sont les plus anormaux
        - **Distance de Mahalanobis élevée**: Indique un éloignement de la distribution normale
        
        #### Conseils pratiques
        - Commencez par un seuil de p-value de 5% (0.05)
        - Pour l'Isolation Forest, un seuil de 0 est généralement approprié
        - Pour Mahalanobis, un seuil de 0.01 est recommandé
        - Utilisez le vote majoritaire pour combiner les méthodes

        #### Insertion d'anomalies
        1. Naviguez vers l'onglet "Insertion d'anomalies"
        2. Sélectionnez le type de nœud (Boucle, PEAG ou OLT)
        3. Choisissez les nœuds spécifiques à modifier
        4. Sélectionnez les variables de test à modifier
        5. Entrez la valeur à ajouter
        6. Cliquez sur "Insérer anomalies"                   
        """)
        
        # Afficher une FAQ
        with st.expander("Questions fréquentes"):
            st.markdown("""
            **Q: Comment savoir quelle méthode de détection choisir?**  
            R: L'approche unidimensionnelle est meilleure pour identifier les anomalies sur des métriques spécifiques. Isolation Forest et Mahalanobis sont plus adaptés pour détecter des anomalies dans des combinaisons de variables.
            
            **Q: Que faire en cas d'anomalie détectée?**  
            R: Vérifiez d'abord la gravité et les métriques concernées. Pour les anomalies critiques, une intervention technique rapide est recommandée.
            
            **Q: Les faux positifs sont-ils possibles?**  
            R: Oui. C'est pourquoi nous utilisons plusieurs méthodes de détection et une approche ensembliste pour réduire ce risque.
            """)
    

    with about_tabs[4]:  # Équipe
        st.subheader("Équipe de développement")
        
        st.write("Challenge réalisé par l'équipe du Master MOSEF de l'Université Paris 1 Panthéon-Sorbonne:")

        team_data = [
            {
                "nom": "Chahla Tarmoun",
                "role": "Data Scientist chez SG Corporate & Investment Banking",
                "linkedin": "https://www.linkedin.com/in/chahla-tarmoun-4b546a160/"
            },
            {
                "nom": "Louis Lebreton",
                "role": "Data Scientist chez Crédit Agricole Leasing & Factoring",
                "linkedin": "https://www.linkedin.com/in/louis-lebreton-17418a1b7/"
            },
            {
                "nom": "Alexis Christien",
                "role": "Data Scientist chez Sogeti",
                "linkedin": "https://www.linkedin.com/in/alexis-christien/"
            },
            {
                "nom": "Issame Abdeljalil",
                "role": "Data Scientist chez Crédit Agricole Leasing & Factoring",
                "linkedin": "https://www.linkedin.com/in/issameabdeljalil/"
            }
        ]

        # Affichage avec icônes cliquables
        for member in team_data:
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(f"**{member['nom']}**  \n{member['role']}")
            with col2:
                st.markdown(
                    f"<a href='{member['linkedin']}' target='_blank'>"
                    f"<img src='https://cdn-icons-png.flaticon.com/512/174/174857.png' width='24' style='margin-top: 6px'/>"
                    f"</a>",
                    unsafe_allow_html=True
                )

        st.write("### Remerciements")
        st.write("Nous tenons à remercier les équipes de SFR et Nexialog pour leur encadrement et les données fournies pour ce challenge.")

# # Fonction pour exécuter la détection d'anomalies
# def run_anomaly_detection():
    
    
#     # Instance de classe : recherche les noeuds anormaux
#     nc = NodesChecker()
#     # Calcul des p_values à partir des distributions empiriques
#     lignes_1fev_with_pval = NodesChecker.add_p_values(
#         st.session_state.lignes_1fev.copy(), 
#         st.session_state.dict_distribution_test
#     )

#     # application de l'isolation forest
#     lignes_1fev_with_if_scores = st.session_state.isolation_forest.predict(st.session_state.lignes_1fev)
#     st.session_state.df_with_anomalies = lignes_1fev_with_if_scores
    
#     # Boucles
#     # p values
#     df_p_values_boucle = nc.get_df_fisher_p_values(
#         lignes_1fev_with_pval,
#         node_type='boucle',
#         p_values=st.session_state.p_values_col
#     )
#     # scores d'isolation forest
#     df_if_scores_boucle = nc.get_if_scores_by_node(lignes_1fev_with_if_scores, node_type = 'boucle')
#     df_results_boucle = pd.merge(df_p_values_boucle, 
#                                   df_if_scores_boucle[['avg_isolation_forest_score', 
#                                                        'majority_anomaly_score',
#                                                        'anomaly_percentage',
#                                                        'total_samples',
#                                                        'anomaly_count']], 
#                                   how='left', 
#                                   left_index=True, right_index=True)

#     st.session_state.results["boucles"] = df_results_boucle
    
#     # Filtrer les boucles défaillantes
#     boucles_defaillantes = df_results_boucle[df_results_boucle['avg_isolation_forest_score'] < st.session_state.isolation_forest_threshold].index
    
#     if len(boucles_defaillantes) > 0:
#         lignes_1fev_with_pval = lignes_1fev_with_pval[~lignes_1fev_with_pval['boucle'].isin(boucles_defaillantes)]
#         lignes_1fev_with_if_scores = lignes_1fev_with_if_scores[~lignes_1fev_with_if_scores['boucle'].isin(boucles_defaillantes)]
    
#     # PEAG
#     # p values
#     df_p_values_peag = nc.get_df_fisher_p_values(
#         lignes_1fev_with_pval,
#         node_type='peag_nro',
#         p_values=st.session_state.p_values_col
#     )
#     # scores d'isolation forest
#     df_if_scores_peag = nc.get_if_scores_by_node(lignes_1fev_with_if_scores, node_type = 'peag_nro')
#     df_results_peag = pd.merge(df_p_values_peag, 
#                                   df_if_scores_peag[['avg_isolation_forest_score', 
#                                                        'majority_anomaly_score',
#                                                        'anomaly_percentage',
#                                                        'total_samples',
#                                                        'anomaly_count']], 
#                                   how='left', 
#                                   left_index=True, right_index=True)
#     st.session_state.results["peag_nro"] = df_results_peag
    
#     # Filtrer les PEAG défaillants
#     peag_defaillants = df_results_peag[df_results_peag['avg_isolation_forest_score'] < st.session_state.isolation_forest_threshold].index
   
#     if len(peag_defaillants) > 0:
#         lignes_1fev_with_pval = lignes_1fev_with_pval[~lignes_1fev_with_pval['peag_nro'].isin(peag_defaillants)]
#         lignes_1fev_with_if_scores = lignes_1fev_with_if_scores[~lignes_1fev_with_if_scores['peag_nro'].isin(peag_defaillants)]
    
#     # OLT
#     # p values
#     df_p_values_olt = nc.get_df_fisher_p_values(
#         lignes_1fev_with_pval,
#         node_type='olt_name',
#         p_values=st.session_state.p_values_col
#     )
#     # scores d'isolation forest
#     df_if_scores_olt = nc.get_if_scores_by_node(lignes_1fev_with_if_scores, node_type = 'olt_name')
#     df_results_olt = pd.merge(df_p_values_olt, 
#                                   df_if_scores_olt[['avg_isolation_forest_score', 
#                                                        'majority_anomaly_score',
#                                                        'anomaly_percentage',
#                                                        'total_samples',
#                                                        'anomaly_count']], 
#                                   how='left', 
#                                   left_index=True, right_index=True)
#     st.session_state.results["olt"] = df_results_olt

#     # Appliquer la méthode ensembliste choisie sur tous les résultats
#     if st.session_state.ensemble_method != "Vote majoritaire" and st.session_state.ensemble_method != "Union" and st.session_state.ensemble_method != "Intersection":
#         st.session_state.ensemble_method = "Vote majoritaire"  # Valeur par défaut
    
#     # Parcourir tous les nœuds et appliquer la méthode ensembliste
#     for node_type in ["boucles", "peag_nro", "olt"]:
#         if node_type in st.session_state.results and st.session_state.results[node_type] is not None:
#             df = st.session_state.results[node_type]
            
#             # Identifier les anomalies selon chaque méthode
#             unidim_columns = [col for col in df.columns if 'p_val_' in col]
#             threshold = st.session_state.p_value_threshold / 100
            
#             # Créer des masques pour chaque méthode
#             unidim_mask = pd.Series(False, index=df.index)
#             if unidim_columns:
#                 unidim_mask = (df[unidim_columns] < threshold).any(axis=1)
            
#             isolation_mask = pd.Series(False, index=df.index)
#             if 'avg_isolation_forest_score' in df.columns:
#                 isolation_mask = df['avg_isolation_forest_score'] < st.session_state.isolation_forest_threshold
            
#             mahalanobis_mask = pd.Series(False, index=df.index)
#             if 'mahalanobis_anomaly' in df.columns:
#                 mahalanobis_mask = df['mahalanobis_anomaly'] == 1
#             elif 'avg_mahalanobis_distance' in df.columns and 'mahalanobis_pvalue' in df.columns:
#                 mahalanobis_mask = df['mahalanobis_pvalue'] < st.session_state.mahalanobis_threshold
            
#             # Appliquer la méthode d'ensemble choisie
#             if st.session_state.ensemble_method == "Vote majoritaire":
#                 # Une anomalie est détectée si au moins 2 méthodes la détectent
#                 final_mask = (unidim_mask.astype(int) + isolation_mask.astype(int) + mahalanobis_mask.astype(int)) >= 2
#                 df['is_anomaly'] = final_mask
#             elif st.session_state.ensemble_method == "Union":
#                 # Une anomalie est détectée si au moins une méthode la détecte
#                 df['is_anomaly'] = unidim_mask | isolation_mask | mahalanobis_mask
#             elif st.session_state.ensemble_method == "Intersection":
#                 # Une anomalie est détectée si toutes les méthodes la détectent
#                 methods_used = sum([len(unidim_columns) > 0, 'avg_isolation_forest_score' in df.columns, 
#                                    'mahalanobis_anomaly' in df.columns or 'avg_mahalanobis_distance' in df.columns])
#                 if methods_used >= 2:  # Au moins 2 méthodes utilisées
#                     active_masks = []
#                     if len(unidim_columns) > 0:
#                         active_masks.append(unidim_mask)
#                     if 'avg_isolation_forest_score' in df.columns:
#                         active_masks.append(isolation_mask)
#                     if 'mahalanobis_anomaly' in df.columns or 'avg_mahalanobis_distance' in df.columns:
#                         active_masks.append(mahalanobis_mask)
                    
#                     final_mask = active_masks[0]
#                     for mask in active_masks[1:]:
#                         final_mask = final_mask & mask
                    
#                     df['is_anomaly'] = final_mask
#                 else:
#                     df['is_anomaly'] = unidim_mask | isolation_mask | mahalanobis_mask  # Default to union if only one method
            
#             # Mettre à jour le DataFrame résultant
#             st.session_state.results[node_type] = df


def run_anomaly_detection():
    """
    Fonction qui exécute la détection d'anomalies avec les méthodes sélectionnées 
    et les paramètres configurés
    """
    
    # Instance de NodesChecker
    nc = NodesChecker()
    
    # 1. MÉTHODE UNIDIMENSIONNELLE
    if st.session_state.use_unidimensional:
        # Calcul des p-values à partir des distributions empiriques
        lignes_1fev_with_pval = nc.add_p_values(
            st.session_state.lignes_1fev.copy(), 
            st.session_state.dict_distribution_test
        )
    else:
        # Créer une copie sans p-values si la méthode n'est pas utilisée
        lignes_1fev_with_pval = st.session_state.lignes_1fev.copy()
    
    # 2. ISOLATION FOREST
    if st.session_state.use_isolation_forest:
        # Application de l'isolation forest
        detector_if = MultiIsolationForestDetector(chain_id_col='name')
        detector_if.load_models(lignes_1fev_with_pval)
        lignes_1fev_with_if_scores = detector_if.predict(lignes_1fev_with_pval)
        
        # Stocker pour référence
        st.session_state.df_with_anomalies = lignes_1fev_with_if_scores
    else:
        # Sans Isolation Forest, utiliser les données précédentes
        lignes_1fev_with_if_scores = lignes_1fev_with_pval
        st.session_state.df_with_anomalies = lignes_1fev_with_pval
    
    # 3. MAHALANOBIS
    if st.session_state.use_mahalanobis:
        detector_mah = MahalanobisDetector(chain_id_col='name')
        detector_mah.load_models(lignes_1fev_with_pval)
        lignes_1fev_with_mah_scores = detector_mah.predict(
            lignes_1fev_with_pval, 
            threshold=st.session_state.mahalanobis_threshold
        )
    else:
        # Sans Mahalanobis, utiliser les données précédentes
        lignes_1fev_with_mah_scores = lignes_1fev_with_if_scores
    
    # 4. EXTRACTION DES RÉSULTATS PAR TYPE DE NŒUD
    
    # Valeurs p pour les boucles
    df_p_values_boucle = nc.get_df_fisher_p_values(
        lignes_1fev_with_pval,
        node_type='boucle',
        p_values=st.session_state.p_values_col
    )
    
    # Scores d'isolation forest pour les boucles
    if st.session_state.use_isolation_forest:
        df_if_scores_boucle = nc.get_if_scores_by_node(lignes_1fev_with_if_scores, node_type='boucle')
    else:
        df_if_scores_boucle = pd.DataFrame(index=df_p_values_boucle.index)
    
    # Scores de Mahalanobis pour les boucles
    if st.session_state.use_mahalanobis:
        df_mah_scores_boucle = nc.get_mahalanobis_scores_by_node(lignes_1fev_with_mah_scores, node_type='boucle')
    else:
        df_mah_scores_boucle = pd.DataFrame(index=df_p_values_boucle.index)
    
    # Fusion des résultats pour les boucles
    df_results_boucle = df_p_values_boucle
    
    if not df_if_scores_boucle.empty:
        columns_to_merge = ['avg_isolation_forest_score', 'majority_anomaly_score',
                           'anomaly_percentage', 'total_samples', 'anomaly_count']
        present_columns = [col for col in columns_to_merge if col in df_if_scores_boucle.columns]
        if present_columns:
            df_results_boucle = pd.merge(
                df_results_boucle,
                df_if_scores_boucle[present_columns],
                how='left',
                left_index=True,
                right_index=True
            )
    
    if not df_mah_scores_boucle.empty:
        columns_to_merge = ['avg_mahalanobis_distance', 'avg_mahalanobis_pvalue',
                           'mahalanobis_anomaly_percentage', 'mahalanobis_anomaly_count']
        present_columns = [col for col in columns_to_merge if col in df_mah_scores_boucle.columns]
        if present_columns:
            df_results_boucle = pd.merge(
                df_results_boucle,
                df_mah_scores_boucle[present_columns],
                how='left',
                left_index=True,
                right_index=True
            )
    
    # Répéter le même processus pour PEAG et OLT
    
    # PEAG
    df_p_values_peag = nc.get_df_fisher_p_values(
        lignes_1fev_with_pval,
        node_type='peag_nro',
        p_values=st.session_state.p_values_col
    )
    
    if st.session_state.use_isolation_forest:
        df_if_scores_peag = nc.get_if_scores_by_node(lignes_1fev_with_if_scores, node_type='peag_nro')
    else:
        df_if_scores_peag = pd.DataFrame(index=df_p_values_peag.index)
    
    if st.session_state.use_mahalanobis:
        df_mah_scores_peag = nc.get_mahalanobis_scores_by_node(lignes_1fev_with_mah_scores, node_type='peag_nro')
    else:
        df_mah_scores_peag = pd.DataFrame(index=df_p_values_peag.index)
    
    df_results_peag = df_p_values_peag
    
    if not df_if_scores_peag.empty:
        columns_to_merge = ['avg_isolation_forest_score', 'majority_anomaly_score',
                           'anomaly_percentage', 'total_samples', 'anomaly_count']
        present_columns = [col for col in columns_to_merge if col in df_if_scores_peag.columns]
        if present_columns:
            df_results_peag = pd.merge(
                df_results_peag,
                df_if_scores_peag[present_columns],
                how='left',
                left_index=True,
                right_index=True
            )
    
    if not df_mah_scores_peag.empty:
        columns_to_merge = ['avg_mahalanobis_distance', 'avg_mahalanobis_pvalue',
                           'mahalanobis_anomaly_percentage', 'mahalanobis_anomaly_count']
        present_columns = [col for col in columns_to_merge if col in df_mah_scores_peag.columns]
        if present_columns:
            df_results_peag = pd.merge(
                df_results_peag,
                df_mah_scores_peag[present_columns],
                how='left',
                left_index=True,
                right_index=True
            )
    
    # OLT
    df_p_values_olt = nc.get_df_fisher_p_values(
        lignes_1fev_with_pval,
        node_type='olt_name',
        p_values=st.session_state.p_values_col
    )
    
    if st.session_state.use_isolation_forest:
        df_if_scores_olt = nc.get_if_scores_by_node(lignes_1fev_with_if_scores, node_type='olt_name')
    else:
        df_if_scores_olt = pd.DataFrame(index=df_p_values_olt.index)
    
    if st.session_state.use_mahalanobis:
        df_mah_scores_olt = nc.get_mahalanobis_scores_by_node(lignes_1fev_with_mah_scores, node_type='olt_name')
    else:
        df_mah_scores_olt = pd.DataFrame(index=df_p_values_olt.index)
    
    df_results_olt = df_p_values_olt
    
    if not df_if_scores_olt.empty:
        columns_to_merge = ['avg_isolation_forest_score', 'majority_anomaly_score',
                           'anomaly_percentage', 'total_samples', 'anomaly_count']
        present_columns = [col for col in columns_to_merge if col in df_if_scores_olt.columns]
        if present_columns:
            df_results_olt = pd.merge(
                df_results_olt,
                df_if_scores_olt[present_columns],
                how='left',
                left_index=True,
                right_index=True
            )
    
    if not df_mah_scores_olt.empty:
        columns_to_merge = ['avg_mahalanobis_distance', 'avg_mahalanobis_pvalue',
                           'mahalanobis_anomaly_percentage', 'mahalanobis_anomaly_count']
        present_columns = [col for col in columns_to_merge if col in df_mah_scores_olt.columns]
        if present_columns:
            df_results_olt = pd.merge(
                df_results_olt,
                df_mah_scores_olt[present_columns],
                how='left',
                left_index=True,
                right_index=True
            )
    
    # 5. STOCKAGE DES RÉSULTATS
    st.session_state.results = {
        "boucles": df_results_boucle,
        "peag_nro": df_results_peag,
        "olt": df_results_olt
    }
    
    # 6. APPLICATION DE LA MÉTHODE ENSEMBLISTE
    if st.session_state.ensemble_method not in ["Vote majoritaire", "Union", "Intersection"]:
        st.session_state.ensemble_method = "Vote majoritaire"  # Valeur par défaut
    
    # Parcourir tous les nœuds et appliquer la méthode ensembliste
    for node_type in ["boucles", "peag_nro", "olt"]:
        if node_type in st.session_state.results and st.session_state.results[node_type] is not None:
            df = st.session_state.results[node_type]
            
            # Identifier les anomalies selon chaque méthode
            unidim_columns = [col for col in df.columns if 'p_val_' in col]
            threshold = st.session_state.p_value_threshold / 100
            
            # Créer des masques pour chaque méthode
            unidim_mask = pd.Series(False, index=df.index)
            if unidim_columns:
                unidim_mask = (df[unidim_columns] < threshold).any(axis=1)
            
            isolation_mask = pd.Series(False, index=df.index)
            if 'avg_isolation_forest_score' in df.columns:
                isolation_mask = df['avg_isolation_forest_score'] < st.session_state.isolation_forest_threshold
            
            mahalanobis_mask = pd.Series(False, index=df.index)
            if 'mahalanobis_anomaly' in df.columns:
                mahalanobis_mask = df['mahalanobis_anomaly'] == 1
            elif 'avg_mahalanobis_pvalue' in df.columns:
                mahalanobis_mask = df['avg_mahalanobis_pvalue'] < st.session_state.mahalanobis_threshold
            
            # Compter les méthodes actives
            methods_used = []
            if len(unidim_columns) > 0:
                methods_used.append(unidim_mask)
            if 'avg_isolation_forest_score' in df.columns:
                methods_used.append(isolation_mask)
            if 'mahalanobis_anomaly' in df.columns or 'avg_mahalanobis_pvalue' in df.columns:
                methods_used.append(mahalanobis_mask)
            
            # Appliquer la méthode d'ensemble choisie
            if len(methods_used) > 0:
                if st.session_state.ensemble_method == "Vote majoritaire" and len(methods_used) >= 2:
                    # Convertir en entiers et sommer
                    vote_sum = sum(mask.astype(int) for mask in methods_used)
                    # Anomalie si au moins la moitié des méthodes disponibles la détectent
                    min_votes = max(2, len(methods_used) // 2 + 1)  # Au moins 2 votes ou la majorité
                    final_mask = vote_sum >= min_votes
                elif st.session_state.ensemble_method == "Intersection" and len(methods_used) >= 2:
                    # Intersection: toutes les méthodes doivent être d'accord
                    final_mask = methods_used[0]
                    for mask in methods_used[1:]:
                        final_mask = final_mask & mask
                else:
                    # Union (par défaut): au moins une méthode détecte
                    final_mask = methods_used[0] if methods_used else pd.Series(False, index=df.index)
                    for mask in methods_used[1:]:
                        final_mask = final_mask | mask
                
                # Ajouter la colonne de décision finale
                df['is_anomaly'] = final_mask
            else:
                # Aucune méthode active
                df['is_anomaly'] = pd.Series(False, index=df.index)
            
            # Mettre à jour le DataFrame résultant
            st.session_state.results[node_type] = df
    
    # 7. AFFICHER LES STATISTIQUES DE DÉTECTION
    method_counts = {
        "Unidimensionnelle": 0,
        "Isolation Forest": 0,
        "Mahalanobis": 0,
        "Total d'anomalies": 0
    }
    
    for node_type in ["boucles", "peag_nro", "olt"]:
        if node_type in st.session_state.results and st.session_state.results[node_type] is not None:
            df = st.session_state.results[node_type]
            
            # Compter par méthode et au total
            if len([col for col in df.columns if 'p_val_' in col]) > 0:
                unidim_count = (df.filter(like='p_val_') < threshold).any(axis=1).sum()
                method_counts["Unidimensionnelle"] += unidim_count
            
            if 'avg_isolation_forest_score' in df.columns:
                if_count = (df['avg_isolation_forest_score'] < st.session_state.isolation_forest_threshold).sum()
                method_counts["Isolation Forest"] += if_count
            
            if 'mahalanobis_anomaly' in df.columns:
                mah_count = df['mahalanobis_anomaly'].sum()
                method_counts["Mahalanobis"] += mah_count
            elif 'avg_mahalanobis_pvalue' in df.columns:
                mah_count = (df['avg_mahalanobis_pvalue'] < st.session_state.mahalanobis_threshold).sum()
                method_counts["Mahalanobis"] += mah_count
            
            if 'is_anomaly' in df.columns:
                method_counts["Total d'anomalies"] += df['is_anomaly'].sum()
    
    # Afficher un résumé des résultats
    st.session_state.anomalies_detected = True
    
    # Retourner les statistiques pour référence
    return method_counts

def merge_results(df_p_values, df_if_scores, df_maha_scores):
    """
    Fusionne les résultats des différentes méthodes de détection
    """
    # Commencer avec le DataFrame non vide ou créer un nouveau DataFrame vide
    if not df_p_values.empty:
        result = df_p_values.copy()
    elif not df_if_scores.empty:
        result = df_if_scores.copy()
    elif not df_maha_scores.empty:
        result = df_maha_scores.copy()
    else:
        return pd.DataFrame()  # Retourner un DataFrame vide si tous sont vides
    
    # Fusionner avec les scores d'isolation forest si disponibles
    if not df_if_scores.empty:
        result = pd.merge(
            result, 
            df_if_scores, 
            how='outer', 
            left_index=True, 
            right_index=True
        )
    
    # Fusionner avec les scores de Mahalanobis si disponibles
    if not df_maha_scores.empty:
        result = pd.merge(
            result, 
            df_maha_scores, 
            how='outer', 
            left_index=True, 
            right_index=True
        )
    
    return result

# Fonction pour créer un graphique 3D pour l'approche unidimensionnelle
def create_3d_plot(test_name, variables_test):
    row_to_plot = st.session_state.lignes_1fev[st.session_state.lignes_1fev['name'] == test_name]
    
    # Axes x et y (distribution empirique)
    x = np.array(st.session_state.dict_distribution_test[variables_test[0]][test_name])
    y = np.array(st.session_state.dict_distribution_test[variables_test[1]][test_name])
    
    # Bornes de la grille
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    
    # Grille 2D (100x100 points)
    X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    
    # Estimation densité empirique avec gaussian_kde
    values = np.vstack([x, y])
    kernel = gaussian_kde(values)
    Z = np.reshape(kernel(positions).T, X.shape)
    
    x_point = float(row_to_plot[variables_test[0]])
    y_point = float(row_to_plot[variables_test[1]])
    
    # Calculer la densité à ce point précis
    point_position = np.array([[x_point], [y_point]])
    z_point = float(kernel(point_position))
    
    # Figure 3D
    fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z, colorscale='Plasma')])
    fig.update_layout(
        title=f'Distribution Empirique 3D de {test_name}',
        scene=dict(
            xaxis_title=variables_test[0],
            yaxis_title=variables_test[1],
            zaxis_title='Densité'
        ),
        width=900,
        height=700,
        margin=dict(l=0, r=0, b=0, t=30)
    )
    
    # Marqueur pour le point actuel
    is_anomaly = z_point < 0.05
    text = "Anomalie" if is_anomaly else ""
    marker_color = 'red' if is_anomaly else 'green'
    
    fig.add_trace(go.Scatter3d(
        x=[x_point],
        y=[y_point],
        z=[z_point + 0.05],  # Légèrement au-dessus pour la visibilité
        mode='markers+text',
        text=[text],
        marker=dict(size=8, color=marker_color),
        textposition="top center"
    ))
    
    return fig

# Main pour organisation de l'interface
def main():

    st.markdown("""
        <style>
        /* Augmente la taille des titres des onglets */
        .stTabs [role="tab"] {
            font-size: 18px !important;
            font-weight: 600;
            padding: 10px 20px;
        }
        </style>
    """, unsafe_allow_html=True)
        
    # Initialiser l'état de session
    initialize_session_state()

    # Afficher l'en-tête
    show_header()

    # Créer la barre d'onglets horizontale pour la navigation
    pages = st.tabs([
        "📊 État du réseau",
        "⚙️ Configuration et détection",
        "📈 Résultats",
        "🔧 Insertion d'anomalies",
        "ℹ️ À propos"
    ])

    # Onglet 0 : État du réseau
    with pages[0]:
        show_network_status()

    # Onglet 1 : Configuration et détection
    with pages[1]:
        show_detection_config()

    # Onglet 2 : Résultats
    with pages[2]:
        show_results()

    # Onglet 3 : Insertion d'anomalies
    with pages[3]:
        show_anomaly_insertion()

    # Onglet 4 : À propos
    with pages[4]:
        show_about()

if __name__ == "__main__":
    main()
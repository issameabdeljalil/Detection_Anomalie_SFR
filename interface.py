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
import plotly.graph_objects as go
from scipy.stats import gaussian_kde
from nodes_checker import NodesChecker
from utils import import_json_to_dict
from detection_anomalies_IsolationForest import IsolationForestDetector
import joblib
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
        st.session_state.lignes_1fev = pd.read_csv('data/results/lignes_1fev.csv', index_col=0)
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
        st.session_state.detection_method = "unidimensionnelle"  # Méthode de détection par défaut
        
        # Initialisation du détecteur Isolation Forest
        st.session_state.isolation_forest = IsolationForestDetector()
        st.session_state.isolation_forest_trained = False
        st.session_state.isolation_forest_threshold = 0.05
        
        # Variables pour stocker les résultats des analyses
        st.session_state.results_unidim = {
            "boucles": None,
            "peag": None,
            "olt": None
        }
        st.session_state.results_isof = {
            "boucles": None,
            "peag": None,
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


# Page tableau de bord
def show_home():
    st.header("📊 Tableau de bord")
    
    # Section KPIs principaux
    st.subheader("Indicateurs clés de performance")
    
    # Première ligne - Statistiques générales et métriques réseau
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Nombre de chaînes techniques", 
                 value=len(st.session_state.lignes_1fev['name'].unique()))
    
    # Calcul des temps moyens DNS et latence
    avg_dns = st.session_state.lignes_1fev['avg_dns_time'].mean()
    avg_latence = st.session_state.lignes_1fev['avg_latence_scoring'].mean()
    
    with col2:
        st.metric(label="Temps DNS moyen (ms)", 
                  value=f"{avg_dns:.2f}")
    with col3:
        st.metric(label="Latence scoring moyenne (ms)", 
                  value=f"{avg_latence:.2f}")
    
    # Deuxième ligne - Indicateurs de santé réseau
    st.subheader("Santé du réseau")
    
    # Calcul des indicateurs de variabilité
    col1, col2, col3 = st.columns(3)
    with col1:
        # Calcul de la stabilité DNS (basé sur std_dns_time)
        std_dns = st.session_state.lignes_1fev['std_dns_time'].mean()
        dns_stability = 100 - min(100, (std_dns / avg_dns * 100))
        st.metric(label="Stabilité DNS (%)", 
                  value=f"{dns_stability:.1f}")
    
    with col2:
        # Calcul de la qualité scoring (basé sur avg_score_scoring)
        avg_score = st.session_state.lignes_1fev['avg_score_scoring'].mean()
        score_quality = min(100, (avg_score / 5) * 100)  # Supposant que le score max est 5
        st.metric(label="Qualité scoring (%)", 
                  value=f"{score_quality:.1f}")
    
    with col3:
        # Calcul de la stabilité latence
        std_latence = st.session_state.lignes_1fev['std_latence_scoring'].mean()
        latence_stability = 100 - min(100, (std_latence / avg_latence * 100))
        st.metric(label="Stabilité latence (%)", 
                  value=f"{latence_stability:.1f}")
    
    # Troisième ligne - Visualisation de la distribution
    st.subheader("Distribution des métriques clés")
    
    # Créer un histogramme pour visualiser les distributions
    fig = create_metrics_distribution()
    st.plotly_chart(fig, use_container_width=True)
    
    # Méthode active d'analyse
    st.info(f"Méthode de détection active: **{st.session_state.detection_method}**")
    
    # Si des anomalies ont été détectées, afficher un résumé
    if st.session_state.anomalies_detected:
        st.subheader("Résumé des anomalies détectées")
        
        # Afficher les résultats selon la méthode utilisée
        if st.session_state.detection_method == "unidimensionnelle":
            display_unidim_anomaly_summary()
        else:
            display_isolation_forest_anomaly_summary()
    
    # Mini ReadMe sur le projet
    st.markdown("---")
    st.subheader("À propos de l'application")
    st.write("""
    Cette application de détection d'anomalies permet de:
    
    - **Simuler** l'injection d'anomalies dans différentes parties du réseau
    - **Détecter** les anomalies par deux approches complémentaires
    - **Visualiser** les résultats en 3D pour mieux comprendre les anomalies
    
    Utilisez le menu ci-dessus pour naviguer entre les différentes fonctionnalités.
    """)

# Fonction pour créer les visualisations de distribution des métriques
def create_metrics_distribution():
    
    # Créer un subplot avec 3 graphiques
    fig = make_subplots(rows=1, cols=3, 
                        subplot_titles=("Distribution des temps DNS", 
                                        "Distribution des scores", 
                                        "Distribution des latences"))
    
    # Histogramme pour les temps DNS
    fig.add_trace(
        go.Histogram(
            x=st.session_state.lignes_1fev['avg_dns_time'],
            name="Temps DNS",
            marker_color='#1f77b4',
            opacity=0.7
        ),
        row=1, col=1
    )
    
    # Histogramme pour les scores
    fig.add_trace(
        go.Histogram(
            x=st.session_state.lignes_1fev['avg_score_scoring'],
            name="Scores",
            marker_color='#2ca02c',
            opacity=0.7
        ),
        row=1, col=2
    )
    
    # Histogramme pour les latences
    fig.add_trace(
        go.Histogram(
            x=st.session_state.lignes_1fev['avg_latence_scoring'],
            name="Latences",
            marker_color='#d62728',
            opacity=0.7
        ),
        row=1, col=3
    )
    
    # Mise à jour de la mise en page
    fig.update_layout(
        height=300,
        bargap=0.1,
        showlegend=False
    )
    
    return fig

# Fonction pour afficher le résumé des anomalies (approche unidimensionnelle)
def display_unidim_anomaly_summary():
    results = st.session_state.results_unidim
    
    # Créer 3 colonnes
    col1, col2, col3 = st.columns(3)
    
    # Définir la couleur pour les anomalies et la sévérité
    threshold = st.session_state.p_value_threshold / 100
    
    with col1:
        if results["boucles"] is not None:
            anomalous_boucles = results["boucles"][results["boucles"].min(axis=1) < threshold]
            st.metric("Boucles anormales", len(anomalous_boucles))
            
            if not anomalous_boucles.empty:
                # Trouver les boucles les plus problématiques
                worst_boucles = anomalous_boucles.min(axis=1).nsmallest(3)
                st.write("Boucles les plus critiques:")
                for idx, val in worst_boucles.items():
                    severity = get_severity_label(val)
                    st.markdown(f"• **{idx}** ({severity})")
    
    with col2:
        if results["peag"] is not None:
            anomalous_peag = results["peag"][results["peag"].min(axis=1) < threshold]
            st.metric("PEAG anormaux", len(anomalous_peag))
            
            if not anomalous_peag.empty:
                # Trouver les PEAG les plus problématiques
                worst_peag = anomalous_peag.min(axis=1).nsmallest(3)
                st.write("PEAG les plus critiques:")
                for idx, val in worst_peag.items():
                    severity = get_severity_label(val)
                    st.markdown(f"• **{idx}** ({severity})")
    
    with col3:
        if results["olt"] is not None:
            anomalous_olt = results["olt"][results["olt"].min(axis=1) < threshold]
            st.metric("OLT anormaux", len(anomalous_olt))
            
            if not anomalous_olt.empty:
                # Trouver les OLT les plus problématiques
                worst_olt = anomalous_olt.min(axis=1).nsmallest(3)
                st.write("OLT les plus critiques:")
                for idx, val in worst_olt.items():
                    severity = get_severity_label(val)
                    st.markdown(f"• **{idx}** ({severity})")
    
    # Ajout d'une visualisation des métriques principalement affectées
    if results["boucles"] is not None and results["peag"] is not None and results["olt"] is not None:
        display_affected_metrics_chart(results)

# Fonction pour afficher le résumé des anomalies (approche Isolation Forest)
def display_isolation_forest_anomaly_summary():
    results = st.session_state.results_isof
    
    # Créer 3 colonnes
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if results["boucles"] is not None:
            anomalous_boucles = results["boucles"][results["boucles"]["is_anomalous"]]
            st.metric("Boucles anormales", len(anomalous_boucles))
            
            if not anomalous_boucles.empty:
                # Trouver les boucles les plus problématiques (selon la proportion)
                worst_boucles = anomalous_boucles.sort_values("proportion", ascending=False).head(3)
                st.write("Boucles les plus critiques:")
                for idx, row in worst_boucles.iterrows():
                    st.markdown(f"• **{idx}** ({row['proportion']:.1%})")
    
    with col2:
        if results["peag"] is not None:
            anomalous_peag = results["peag"][results["peag"]["is_anomalous"]]
            st.metric("PEAG anormaux", len(anomalous_peag))
            
            if not anomalous_peag.empty:
                # Trouver les PEAG les plus problématiques
                worst_peag = anomalous_peag.sort_values("proportion", ascending=False).head(3)
                st.write("PEAG les plus critiques:")
                for idx, row in worst_peag.iterrows():
                    st.markdown(f"• **{idx}** ({row['proportion']:.1%})")
    
    with col3:
        if results["olt"] is not None:
            anomalous_olt = results["olt"][results["olt"]["is_anomalous"]]
            st.metric("OLT anormaux", len(anomalous_olt))
            
            if not anomalous_olt.empty:
                # Trouver les OLT les plus problématiques
                worst_olt = anomalous_olt.sort_values("proportion", ascending=False).head(3)
                st.write("OLT les plus critiques:")
                for idx, row in worst_olt.iterrows():
                    st.markdown(f"• **{idx}** ({row['proportion']:.1%})")
    
    # Si on a les résultats des anomalies, ajouter une heatmap
    if hasattr(st.session_state, 'df_with_anomalies'):
        display_anomaly_heatmap()

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

# Fonction pour afficher un graphique sur les métriques les plus affectées
def display_affected_metrics_chart(results):
    
    # Collecter toutes les p-values pour chaque métrique
    metrics = st.session_state.p_values_col
    metrics_display = [m.replace('p_val_', '') for m in metrics]
    
    # Calculer le nombre d'anomalies par métrique
    threshold = st.session_state.p_value_threshold / 100
    anomaly_counts = []
    
    # Pour chaque métrique, combien de fois elle est sous le seuil
    for metric in metrics:
        count = 0
        for node_type in ["boucles", "peag", "olt"]:
            if results[node_type] is not None:
                count += (results[node_type][metric] < threshold).sum()
        anomaly_counts.append(count)
    
    # graphique à barres
    fig = go.Figure(data=[
        go.Bar(
            x=metrics_display,
            y=anomaly_counts,
            marker_color=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#c2c2f0', '#ffb3e6']
        )
    ])
    
    fig.update_layout(
        title="Métriques les plus affectées par les anomalies",
        xaxis_title="Métrique",
        yaxis_title="Nombre d'anomalies",
        height=300,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Fonction pour afficher une carte de chaleur des anomalies (Isolation Forest)
def display_anomaly_heatmap():
    
    # Préparer les données pour la heatmap
    df_anomalies = st.session_state.df_with_anomalies
    
    # Calculer le pourcentage d'anomalies par OLT et PEAG
    cross_tab = pd.crosstab(
        df_anomalies['peag_nro'], 
        df_anomalies['olt_name'],
        values=df_anomalies['anomaly_score'].apply(lambda x: 1 if x == -1 else 0),
        aggfunc='mean'
    ).fillna(0)
    
    # Limiter à 10 PEAG et 10 OLT maximum pour la lisibilité
    if cross_tab.shape[0] > 10 or cross_tab.shape[1] > 10:
        # Identifier les PEAG et OLT avec le plus d'anomalies
        peag_anomaly_count = cross_tab.sum(axis=1).sort_values(ascending=False).head(10).index
        olt_anomaly_count = cross_tab.sum(axis=0).sort_values(ascending=False).head(10).index
        cross_tab = cross_tab.loc[peag_anomaly_count, olt_anomaly_count]
    
    fig = go.Figure(data=go.Heatmap(
        z=cross_tab.values * 100,  # En pourcentage
        x=cross_tab.columns,
        y=cross_tab.index,
        colorscale='Reds',
        colorbar=dict(title="% Anomalies")
    ))
    
    fig.update_layout(
        title="Heatmap des anomalies par OLT et PEAG",
        xaxis_title="OLT",
        yaxis_title="PEAG",
        height=400,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    st.plotly_chart(fig, use_container_width=True)

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
                st.write(f"Insertion de {round(valeur_insertion, 2)} dans {var_test} pour {len(boucle_names)} boucles")
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
                st.write(f"Insertion de {round(valeur_insertion, 2)} dans {var_test} pour {len(peag_names)} PEAG")
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
                st.write(f"Insertion de {round(valeur_insertion, 2)} dans {var_test} pour {len(olt_names)} OLT")
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

# Page de configuration et détection
def show_detection_config():
    st.header("⚙️ Configuration et lancement de la détection")
    
    # Choix de la méthode de détection
    st.subheader("Méthode de détection")
    detection_method = st.radio(
        "Choisissez la méthode de détection d'anomalies :",
        ["Approche unidimensionnelle", "Isolation Forest (multidimensionnelle)"],
        index=0 if st.session_state.detection_method == "unidimensionnelle" else 1,
        horizontal=True
    )
    
    st.session_state.detection_method = "unidimensionnelle" if detection_method == "Approche unidimensionnelle" else "isolation_forest"
    
    # Configuration du seuil selon la méthode
    st.subheader("Configuration du seuil")
    
    if st.session_state.detection_method == "unidimensionnelle":
        new_threshold = st.slider(
            "Seuil (%) de rejet α pour la détection d'anomalies", 
            min_value=0.5, 
            max_value=20.0, 
            value=st.session_state.p_value_threshold, 
            step=0.5,
            format="%.1f"
        )
        st.session_state.p_value_threshold = new_threshold
        st.info(f"Les p-values inférieures à {new_threshold}% seront considérées comme anormales")
    else:
        # Vérifier si le modèle est entraîné
        if not st.session_state.isolation_forest_trained:
            st.warning("Le modèle Isolation Forest n'est pas encore entraîné.")
            train_button = st.button("Entraîner le modèle Isolation Forest", type="primary")
            
            if train_button:
                with st.spinner("Entraînement du modèle en cours..."):
                    try:
                        # Charger le dataset complet pour l'entraînement
                        df_complet = pd.read_csv("/Users/issameabdeljalil/Desktop/M2_MOSEF/challenge_Nexialog/Detection_Anomalie_SFR/data/corrected/new_df.csv")
                        # Entraîner le modèle
                        st.session_state.isolation_forest.train_model(df_complet, contamination=0.02)
                        st.session_state.isolation_forest_trained = True
                        st.success("Modèle Isolation Forest entraîné avec succès !")
                    except Exception as e:
                        st.error(f"Erreur lors de l'entraînement du modèle : {str(e)}")
                        st.info("Essai de chargement d'un modèle existant...")
                        try:
                            st.session_state.isolation_forest.model = joblib.load('isolation_forest_model.joblib')
                            st.session_state.isolation_forest.scaler = joblib.load('scaler.joblib')
                            st.session_state.isolation_forest_trained = True
                            st.success("Modèle Isolation Forest chargé depuis un fichier existant")
                        except:
                            st.error("Impossible de charger un modèle existant.")
        
        new_threshold = st.slider(
            "Seuil de proportion d'anomalies pour considérer un nœud comme anormal", 
            min_value=0.01, 
            max_value=0.5, 
            value=st.session_state.isolation_forest_threshold, 
            step=0.01,
            format="%.2f"
        )
        st.session_state.isolation_forest_threshold = new_threshold
        st.info(f"Les nœuds avec plus de {new_threshold:.0%} d'anomalies seront considérés comme anormaux")
    
    # Lancement de la détection
    st.markdown("---")
    launch_button_text = "🚀 Lancer la détection d'anomalies"
    
    if st.button(launch_button_text, type="primary"):
        if st.session_state.detection_method == "isolation_forest" and not st.session_state.isolation_forest_trained:
            st.error("Le modèle Isolation Forest n'est pas entraîné. Veuillez l'entraîner avant de lancer la détection.")
        else:
            with st.spinner("Détection d'anomalies en cours..."):
                run_anomaly_detection()
                st.session_state.anomalies_detected = True
                st.success("Détection d'anomalies terminée avec succès!")
                st.balloons()  # Animation pour célébrer la fin de l'analyse

# Page des résultats
def show_results():
    st.header("📈 Résultats de la détection d'anomalies")
    
    if not st.session_state.anomalies_detected:
        st.warning("Aucune détection d'anomalies n'a été lancée. Veuillez configurer et lancer une détection d'abord.")
        return
    
    # Création d'onglets pour les différents types de résultats
    tab1, tab2, tab3, tab4 = st.tabs(["Boucles", "PEAG", "OLT", "Visualisation 3D"])
    
    if st.session_state.detection_method == "unidimensionnelle":
        results = st.session_state.results_unidim
        
        with tab1:
            st.subheader("Boucles anormales")
            if results["boucles"] is not None:
                st.dataframe(results["boucles"].style.highlight_between(
                    left=0, 
                    right=st.session_state.p_value_threshold / 100.0,
                    props='background-color: #ff0000'
                ))
        
        with tab2:
            st.subheader("PEAG anormaux")
            if results["peag"] is not None:
                st.dataframe(results["peag"].style.highlight_between(
                    left=0, 
                    right=st.session_state.p_value_threshold / 100.0,
                    props='background-color: #ff0000'
                ))
        
        with tab3:
            st.subheader("OLT anormaux")
            if results["olt"] is not None:
                st.dataframe(results["olt"].style.highlight_between(
                    left=0, 
                    right=st.session_state.p_value_threshold / 100.0,
                    props='background-color: #ff0000'
                ))
        
        with tab4:
            st.subheader("Visualisation 3D - Approche Unidimensionnelle")
            
            col_test_names, col_test_vars = st.columns([1, 2])
            
            with col_test_names:
                test_name = st.selectbox(
                    "Choix de la chaîne technique",
                    options=sorted(list(st.session_state.lignes_1fev['name'].unique()))
                )
            with col_test_vars:
                variables_test = st.multiselect(
                    "Variables de test (sélectionnez exactement 2)",
                    options=st.session_state.variables_test
                )
            
            if len(variables_test) != 2:
                st.error("Veuillez sélectionner exactement 2 variables de test à représenter en 3D")
            else:
                # Créer et afficher le graphique 3D unidimensionnel
                try:
                    fig = create_3d_unidim_plot(test_name, variables_test)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Erreur lors de la création du graphique 3D : {str(e)}")
    else:
        results = st.session_state.results_isof
        
        with tab1:
            st.subheader("Boucles anormales")
            if results["boucles"] is not None:
                st.dataframe(results["boucles"].style.apply(
                    lambda x: ['background-color: #ff0000' if x['is_anomalous'] else '' for i in x.index],
                    axis=1
                ))
        
        with tab2:
            st.subheader("PEAG anormaux")
            if results["peag"] is not None:
                st.dataframe(results["peag"].style.apply(
                    lambda x: ['background-color: #ff0000' if x['is_anomalous'] else '' for i in x.index],
                    axis=1
                ))
        
        with tab3:
            st.subheader("OLT anormaux")
            if results["olt"] is not None:
                st.dataframe(results["olt"].style.apply(
                    lambda x: ['background-color: #ff0000' if x['is_anomalous'] else '' for i in x.index],
                    axis=1
                ))
        
        with tab4:
            st.subheader("Visualisation 3D - Approche Isolation Forest")
            
            col_test_names, col_test_vars = st.columns([1, 2])
            
            with col_test_names:
                test_name = st.selectbox(
                    "Choix de la chaîne technique",
                    options=sorted(list(st.session_state.lignes_1fev['name'].unique())),
                    key="if_test_name"
                )
            with col_test_vars:
                variables_test_if = st.multiselect(
                    "Variables de test (sélectionnez exactement 2)",
                    options=st.session_state.variables_test,
                    key="if_variables"
                )
            
            if len(variables_test_if) != 2:
                st.error("Veuillez sélectionner exactement 2 variables de test à représenter en 3D")
            else:
                # Créer et afficher le graphique 3D Isolation Forest
                try:
                    fig_if = st.session_state.isolation_forest.create_3d_plot(
                        st.session_state.df_with_anomalies, test_name, variables_test_if
                    )
                    st.plotly_chart(fig_if, use_container_width=True)
                except Exception as e:
                    st.error(f"Erreur lors de la création du graphique 3D : {str(e)}")

# Page d'aide
def show_help():
    st.header("❓ Aide et documentation")
    
    st.write("""
    ## Guide d'utilisation de l'application
    
    Cette application permet de détecter des anomalies dans un réseau de télécommunications en utilisant deux approches complémentaires:
    
    1. **Approche unidimensionnelle** : Analyse chaque dimension séparément avec des techniques de filtrage Hodrick-Prescott,
       de réduction des valeurs aberrantes, et d'estimation par noyau (KDE).
       
    2. **Isolation Forest** : Approche multidimensionnelle qui détecte les anomalies en isolant les observations
       qui s'écartent du comportement habituel.
    
    ### Comment utiliser l'application
    
    1. **Insertion d'anomalies** : Permet de simuler des anomalies dans le réseau en ajoutant des valeurs spécifiques
       à certaines métriques pour des nœuds choisis (Boucles, PEAG, OLT).
       
    2. **Configuration et détection** : Permet de choisir la méthode de détection et de configurer les seuils de sensibilité.
       
    3. **Résultats** : Affiche les nœuds anormaux sous forme de tableaux et permet de visualiser les anomalies en 3D.
       
    4. **Tableau de bord** : Résume les informations clés et l'état actuel de l'analyse.
    
    ### Glossaire
    
    - **OLT** : Optical Line Terminal, équipement central du réseau fibre optique
    - **PEAG** : Point d'Entrée d'Accès Goulte, point de raccordement intermédiaire
    - **Boucle** : Segment du réseau reliant plusieurs points d'accès
    - **p-value** : Probabilité d'observer une valeur au moins aussi extrême que celle observée
    """)
    
    # FAQ
    st.subheader("Questions fréquentes")
    
    with st.expander("Comment insérer une anomalie ?"):
        st.write("""
        1. Allez dans la section "Insertion d'anomalies"
        2. Sélectionnez le type de nœud (Boucle, PEAG ou OLT)
        3. Choisissez les nœuds spécifiques à modifier
        4. Sélectionnez les variables de test à modifier
        5. Entrez la valeur à ajouter
        6. Cliquez sur "Insérer anomalies"
        """)
    
    with st.expander("Quelle méthode de détection choisir ?"):
        st.write("""
        - **Approche unidimensionnelle** : Meilleure pour comprendre quelles métriques spécifiques sont anormales
        - **Isolation Forest** : Plus puissant pour détecter des anomalies complexes impliquant plusieurs métriques
        
        Il est souvent utile d'utiliser les deux approches de manière complémentaire.
        """)
    
    with st.expander("Comment interpréter les résultats ?"):
        st.write("""
        **Approche unidimensionnelle** :
        - Les p-values inférieures au seuil (par défaut 5%) indiquent des anomalies
        - Plus la p-value est faible, plus l'anomalie est significative
        
        **Isolation Forest** :
        - Le score d'anomalie est représenté en 3D, avec les pics indiquant des régions anormales
        - Les nœuds avec une proportion d'anomalies supérieure au seuil sont considérés comme anormaux
        """)

# Fonction pour exécuter la détection d'anomalies
def run_anomaly_detection():
    if st.session_state.detection_method == "unidimensionnelle":
        # Instance de classe : recherche les noeuds anormaux
        nc = NodesChecker()
        # Calcul des p_values à partir des distributions empiriques
        lignes_1fev_with_pval = NodesChecker.add_p_values(
            st.session_state.lignes_1fev.copy(), 
            st.session_state.dict_distribution_test
        )
        
        # Boucles
        df_p_values_boucle = nc.get_df_fisher_p_values(
            lignes_1fev_with_pval,
            node_type='boucle',
            p_values=st.session_state.p_values_col
        )
        st.session_state.results_unidim["boucles"] = df_p_values_boucle
        
        # Filtrer les boucles défaillantes
        boucles_defaillantes = df_p_values_boucle[df_p_values_boucle.min(axis=1) < st.session_state.p_value_threshold / 100].index
        lignes_1fev_filtered = lignes_1fev_with_pval.copy()
        if len(boucles_defaillantes) > 0:
            lignes_1fev_filtered = lignes_1fev_filtered[~lignes_1fev_filtered['boucle'].isin(boucles_defaillantes)]
        
        # PEAG
        df_p_values_peag = nc.get_df_fisher_p_values(
            lignes_1fev_filtered,
            node_type='peag_nro',
            p_values=st.session_state.p_values_col
        )
        st.session_state.results_unidim["peag"] = df_p_values_peag
        
        # Filtrer les PEAG défaillants
        peag_defaillants = df_p_values_peag[df_p_values_peag.min(axis=1) < st.session_state.p_value_threshold / 100].index
        if len(peag_defaillants) > 0:
            lignes_1fev_filtered = lignes_1fev_filtered[~lignes_1fev_filtered['peag_nro'].isin(peag_defaillants)]
        
        # OLT
        df_p_values_olt = nc.get_df_fisher_p_values(
            lignes_1fev_filtered,
            node_type='olt_name',
            p_values=st.session_state.p_values_col
        )
        st.session_state.results_unidim["olt"] = df_p_values_olt
        
        # Sauvegarder le dataframe avec p-values pour la visualisation
        st.session_state.lignes_1fev_with_pval = lignes_1fev_with_pval
        
    else:  # Isolation Forest
        if not st.session_state.isolation_forest_trained:
            raise RuntimeError("Le modèle Isolation Forest n'est pas entraîné.")
        
        # Appliquer l'Isolation Forest sur les données actuelles
        df_with_anomalies = st.session_state.isolation_forest.predict(st.session_state.lignes_1fev)
        
        # Sauvegarder le dataframe avec scores d'anomalies pour la visualisation
        st.session_state.df_with_anomalies = df_with_anomalies
        
        # Calcul des nœuds anormaux selon chaque type
        threshold = st.session_state.isolation_forest_threshold
        
        # Boucles
        df_anomalies_boucle = st.session_state.isolation_forest.get_anomalies_by_node(
            df_with_anomalies, 'boucle', threshold=threshold
        )
        st.session_state.results_isof["boucles"] = df_anomalies_boucle
        
        # Filtrer les boucles anormales
        boucles_anormales = df_anomalies_boucle[df_anomalies_boucle['is_anomalous']].index.tolist()
        df_filtered = df_with_anomalies.copy()
        if boucles_anormales:
            df_filtered = df_filtered[~df_filtered['boucle'].isin(boucles_anormales)]
        
        # PEAG
        df_anomalies_peag = st.session_state.isolation_forest.get_anomalies_by_node(
            df_filtered, 'peag_nro', threshold=threshold
        )
        st.session_state.results_isof["peag"] = df_anomalies_peag
        
        # Filtrer les PEAG anormaux
        peag_anormaux = df_anomalies_peag[df_anomalies_peag['is_anomalous']].index.tolist()
        if peag_anormaux:
            df_filtered = df_filtered[~df_filtered['peag_nro'].isin(peag_anormaux)]
        
        # OLT
        df_anomalies_olt = st.session_state.isolation_forest.get_anomalies_by_node(
            df_filtered, 'olt_name', threshold=threshold
        )
        st.session_state.results_isof["olt"] = df_anomalies_olt

# Fonction pour créer un graphique 3D pour l'approche unidimensionnelle
def create_3d_unidim_plot(test_name, variables_test):
    row_to_plot = st.session_state.lignes_1fev_with_pval[st.session_state.lignes_1fev_with_pval['name'] == test_name]
    
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
    # Initialiser l'état de session
    initialize_session_state()
    
    # Afficher l'en-tête
    show_header()
    
    # Créer le menu de navigation horizontal
    menu = ["📊 Tableau de bord", "🔧 Insertion d'anomalies", "⚙️ Configuration et détection", "📈 Résultats", "❓ Aide"]
    choice = st.radio("Navigation", menu, horizontal=True)
    
    st.markdown("---")
    
    # Afficher la page correspondante au choix
    if choice == "📊 Tableau de bord":
        show_home()
    elif choice == "🔧 Insertion d'anomalies":
        show_anomaly_insertion()
    elif choice == "⚙️ Configuration et détection":
        show_detection_config()
    elif choice == "📈 Résultats":
        show_results()
    elif choice == "❓ Aide":
        show_help()

if __name__ == "__main__":
    main()
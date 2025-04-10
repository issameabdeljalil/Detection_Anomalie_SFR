"""
Challenge Nexialog MoSEF
Interface Streamlit réorganisée :
- Tableau de bord principal avec vue d'ensemble
- Configuration et lancement de la détection d'anomalies
- Affichage des résultats filtrés sur les anomalies avec leur méthode de détection
- Insertion optionnelle d'anomalies dans des noeuds choisis (Boucles, PEAG, OLT)
- Visualisation 3D des anomalies (approche unidimensionnelle ou multidimensionnelle)
"""

import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils import import_json_to_dict
from nodes_checker import NodesChecker
from anomaly_detection_isolation_forest import MultiIsolationForestDetector

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
        # st.session_state.lignes_1fev = pd.read_csv('data/results/lignes_1fev.csv', index_col=0).replace([np.inf, -np.inf], np.nan).dropna().head(300)
        st.session_state.lignes_1fev = pd.read_csv('data/results/lignes_1fev.csv', index_col=0).replace([np.inf, -np.inf], np.nan).dropna()
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
        st.session_state.isolation_forest = MultiIsolationForestDetector(chain_id_col = 'name')
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
    
    # Mini ReadMe sur le projet
    st.markdown("---")
    st.subheader("À propos de l'application")
    st.write("""
    Cette application de détection d'anomalies permet de:
    
    - **Configurer et lancer** la détection d'anomalies avec deux approches complémentaires
    - **Visualiser** les résultats et comprendre les anomalies détectées
    - **Simuler** (optionnellement) l'injection d'anomalies dans différentes parties du réseau
    
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

# Page de configuration et détection
def show_detection_config():
    st.header("⚙️ Configuration et lancement de la détection d'anomalies")
  
    # Configuration du seuil selon la méthode
    st.subheader("Configuration des seuils")
    
    new_threshold = st.slider(
        "Seuil (%) de rejet α pour la détection d'anomalies unidimensionnelle", 
        min_value=0.0, 
        max_value=20.0, 
        value=st.session_state.p_value_threshold, 
        step=0.05,
        format="%.1f"
    )
    st.session_state.p_value_threshold = new_threshold
    st.info(f"Les variables avec une p-value inférieure à {new_threshold}% seront considérées comme anormales")
    
    new_threshold = st.slider(
        "Seuil de la moyenne des scores d'Isolation Forest pour la détection multidimensionnelle", 
        min_value=-1.0, 
        max_value=0.5, 
        value=st.session_state.isolation_forest_threshold, 
        step=0.05,
        format="%.2f"
    )
    st.session_state.isolation_forest_threshold = new_threshold
    st.info(f"Les nœuds avec une moyenne de scores d'anomalies inférieure à {new_threshold} seront considérés comme anormaux")
    
    # Lancement de la détection
    st.markdown("---")
    launch_button_text = "🚀 Lancer la détection d'anomalies"
    
    if st.button(launch_button_text, type="primary"):
        with st.spinner("Détection d'anomalies en cours ..."):
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
    
    # Afficher d'abord le résumé des anomalies détectées
    st.subheader("Résumé des anomalies détectées")
    display_anomaly_summary()
    
    # # Afficher les visualisations des anomalies par type de nœud
    # st.markdown("---")
    # st.subheader("Distribution des anomalies")
    # display_node_anomaly_chart()
    
    # Création d'onglets pour les différents types de résultats
    tab1, tab2, tab3, tab4 = st.tabs(["Boucles", "PEAG", "OLT", "Visualisation 3D"])
    
    results = st.session_state.results
    threshold_p_val = st.session_state.p_value_threshold / 100
    threshold_if = st.session_state.isolation_forest_threshold
    
    with tab1:
        st.subheader("Boucles anormales")
        if results["boucles"] is not None:
            # Filtrer seulement les boucles anormales - utiliser uniquement 3 p-values principales
            df = results["boucles"].copy()
            # N'utiliser que les 3 p-values principales (moyennes) pour être cohérent avec le résumé
            main_p_values = ['p_val_avg_dns_time', 'p_val_avg_score_scoring', 'p_val_avg_latence_scoring']
            p_val_min = df[main_p_values].min(axis=1)
            mask_p_val = p_val_min < threshold_p_val
            mask_if = df['avg_isolation_forest_score'] < threshold_if
            filtered_df = df[mask_p_val | mask_if].sort_values(by="avg_isolation_forest_score")
            
            # Ajouter une colonne indiquant la méthode de détection
            filtered_df['méthode_détection'] = ''
            filtered_df.loc[mask_p_val & mask_if, 'méthode_détection'] = 'Les deux méthodes'
            filtered_df.loc[mask_p_val & ~mask_if, 'méthode_détection'] = 'Unidimensionnelle'
            filtered_df.loc[~mask_p_val & mask_if, 'méthode_détection'] = 'Isolation Forest'

            if filtered_df.empty:
                st.info("Aucune boucle anormale n'a été détectée avec les seuils actuels.")
            else:
                st.write(f"Nombre de boucles anormales détectées: {len(filtered_df)} sur {len(df)}.")
                
                # Fonction pour colorer les cellules
                def color_cells(val, idx, col_name):
                    if col_name in st.session_state.p_values_col and val < threshold_p_val:
                        return 'background-color: #ffcccc'
                    elif col_name == 'avg_isolation_forest_score' and val < threshold_if:
                        return 'background-color: #ffcccc'
                    return ''
                
                # Appliquer la coloration
                styled_df = filtered_df.style.apply(lambda row: [
                    color_cells(row[col], row.name, col) for col in filtered_df.columns
                ], axis=1)
                
                st.dataframe(styled_df)
    
    with tab2:
        st.subheader("PEAG anormaux")
        if results["peag_nro"] is not None:
            # Filtrer seulement les PEAG anormaux - utiliser uniquement 3 p-values principales
            df = results["peag_nro"].copy()
            # N'utiliser que les 3 p-values principales (moyennes) pour être cohérent avec le résumé
            main_p_values = ['p_val_avg_dns_time', 'p_val_avg_score_scoring', 'p_val_avg_latence_scoring']
            p_val_min = df[main_p_values].min(axis=1)
            mask_p_val = p_val_min < threshold_p_val
            mask_if = df['avg_isolation_forest_score'] < threshold_if
            filtered_df = df[mask_p_val | mask_if].sort_values(by="avg_isolation_forest_score")
            
            # Ajouter une colonne indiquant la méthode de détection
            filtered_df['méthode_détection'] = ''
            filtered_df.loc[mask_p_val & mask_if, 'méthode_détection'] = 'Les deux méthodes'
            filtered_df.loc[mask_p_val & ~mask_if, 'méthode_détection'] = 'Unidimensionnelle'
            filtered_df.loc[~mask_p_val & mask_if, 'méthode_détection'] = 'Isolation Forest'
            
            if filtered_df.empty:
                st.info("Aucun PEAG anormal n'a été détecté avec les seuils actuels.")
            else:
                st.write(f"Nombre de PEAG anormaux détectés: {len(filtered_df)} sur {len(df)}.")
                
                # Fonction pour colorer les cellules
                def color_cells(val, idx, col_name):
                    if col_name in st.session_state.p_values_col and val < threshold_p_val:
                        return 'background-color: #ffcccc'
                    elif col_name == 'avg_isolation_forest_score' and val < threshold_if:
                        return 'background-color: #ffcccc'
                    return ''
                
                # Appliquer la coloration
                styled_df = filtered_df.style.apply(lambda row: [
                    color_cells(row[col], row.name, col) for col in filtered_df.columns
                ], axis=1)
                
                st.dataframe(styled_df)
    
    with tab3:
        st.subheader("OLT anormaux")
        if results["olt"] is not None:
            # Filtrer seulement les OLT anormaux - utiliser uniquement 3 p-values principales
            df = results["olt"].copy()
            # N'utiliser que les 3 p-values principales (moyennes) pour être cohérent avec le résumé
            main_p_values = ['p_val_avg_dns_time', 'p_val_avg_score_scoring', 'p_val_avg_latence_scoring']
            p_val_min = df[main_p_values].min(axis=1)
            mask_p_val = p_val_min < threshold_p_val
            mask_if = df['avg_isolation_forest_score'] < threshold_if
            filtered_df = df[mask_p_val | mask_if].sort_values(by="avg_isolation_forest_score")
            
            # Ajouter une colonne indiquant la méthode de détection
            filtered_df['méthode_détection'] = ''
            filtered_df.loc[mask_p_val & mask_if, 'méthode_détection'] = 'Les deux méthodes'
            filtered_df.loc[mask_p_val & ~mask_if, 'méthode_détection'] = 'Unidimensionnelle'
            filtered_df.loc[~mask_p_val & mask_if, 'méthode_détection'] = 'Isolation Forest'
            
            if filtered_df.empty:
                st.info("Aucun OLT anormal n'a été détecté avec les seuils actuels.")
            else:
                st.write(f"Nombre d'OLT anormaux détectés: {len(filtered_df)} sur {len(df)}.")
                
                # Fonction pour colorer les cellules
                def color_cells(val, idx, col_name):
                    if col_name in st.session_state.p_values_col and val < threshold_p_val:
                        return 'background-color: #ffcccc'
                    elif col_name == 'avg_isolation_forest_score' and val < threshold_if:
                        return 'background-color: #ffcccc'
                    return ''
                
                # Appliquer la coloration
                styled_df = filtered_df.style.apply(lambda row: [
                    color_cells(row[col], row.name, col) for col in filtered_df.columns
                ], axis=1)
                
                st.dataframe(styled_df)
    
    with tab4:
        st.subheader("Visualisation 3D")
        
        col_test_names, col_test_vars = st.columns([1, 2])
        
        with col_test_names:
            test_name = st.selectbox(
                "Choix de la chaîne technique",
                options=sorted(list(st.session_state.lignes_1fev_copy['name'].unique()))
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
                fig = create_3d_plot(test_name, variables_test)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Erreur lors de la création du graphique 3D : {str(e)}")

# Page d'insertion d'anomalies (optionnelle)
def show_anomaly_insertion():
    st.header("🔧 Insertion d'anomalies (Optionnel)")
    
    st.info("""
    Cette section vous permet d'insérer des anomalies simulées dans le réseau pour tester le système de détection.
    Cette étape est optionnelle - si vous travaillez avec des données réelles, vous pouvez ignorer cette section.
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
    
    1. **Tableau de bord** : Visualise l'état actuel du réseau et donne un aperçu général des métriques.
    
    2. **Configuration et détection** : Permet de configurer les seuils de détection et de lancer l'analyse.
       
    3. **Résultats** : Affiche le résumé des anomalies et les détails par type de nœud (Boucles, PEAG, OLT).
    
    4. **Insertion d'anomalies (Optionnel)** : Permet de simuler des anomalies dans le réseau pour tester le système.
       
    5. **Aide** : Documentation sur l'utilisation de l'application.
    
    ### Glossaire
    
    - **OLT** : Optical Line Terminal, équipement central du réseau fibre optique
    - **PEAG** : Point d'Entrée d'Accès Goulte, point de raccordement intermédiaire
    - **Boucle** : Segment du réseau reliant plusieurs points d'accès
    - **p-value** : Probabilité d'observer une valeur au moins aussi extrême que celle observée
    """)
    
    # FAQ
    st.subheader("Questions fréquentes")
    
    with st.expander("Comment fonctionne la détection d'anomalies ?"):
        st.write("""
        L'application utilise deux approches complémentaires :
        
        - **Approche unidimensionnelle** : Pour chaque métrique (DNS, scoring, latence), nous calculons une distribution de référence avec le filtre Hodrick-Prescott et une estimation par noyau (KDE). Nous comparons ensuite les nouvelles observations à cette distribution pour calculer des p-values. Une p-value faible indique une anomalie.
        
        - **Isolation Forest** : Cette méthode multidimensionnelle considère toutes les métriques ensemble. Elle isole les observations anormales en créant des partitions aléatoires de l'espace. Les observations qui sont facilement isolées (moins de partitions nécessaires) sont considérées comme des anomalies.
        """)
    
    with st.expander("Comment interpréter les résultats ?"):
        st.write("""
        - **p-values** : Plus elles sont basses, plus l'anomalie est significative. Une valeur inférieure au seuil α (par défaut 5%) indique une anomalie.
        
        - **Score d'Isolation Forest** : Les valeurs négatives indiquent des anomalies. Plus le score est négatif, plus l'anomalie est significative.
        
        - **Méthode de détection** : Indique quelle approche a détecté l'anomalie (unidimensionnelle, Isolation Forest, ou les deux).
        """)
    
    with st.expander("Comment simuler des anomalies ?"):
        st.write("""
        1. Allez dans la section "Insertion d'anomalies"
        2. Sélectionnez le type de nœud (Boucle, PEAG ou OLT)
        3. Choisissez les nœuds spécifiques à modifier
        4. Sélectionnez les variables de test à modifier
        5. Entrez la valeur à ajouter (une valeur élevée, comme 10 ou plus, créera une anomalie évidente)
        6. Cliquez sur "Insérer anomalies"
        7. Retournez à "Configuration et détection" pour lancer l'analyse
        """)

# Fonction pour afficher le résumé des anomalies
def display_anomaly_summary():
    results = st.session_state.results
    # Définir les seuils
    threshold_p_val = st.session_state.p_value_threshold / 100
    threshold_if = st.session_state.isolation_forest_threshold
    
    # Créer deux colonnes principales
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
                    'tickangle': 0,
                    'tickfont': {'size': 12}
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
        
        fig.update_layout(
            height=200,
            width=320,
            margin=dict(l=20, r=50, t=50, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
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

# # Fonction pour afficher le graphique des anomalies par type de nœud
# def display_node_anomaly_chart():
#     """
#     Crée un graphique de synthèse montrant la distribution des anomalies par type de nœud
#     et leur répartition par variable.
#     """
#     if not st.session_state.anomalies_detected:
#         return
    
#     # Créer deux colonnes pour les graphiques
#     col1, col2 = st.columns(2)
    
#     with col1:
#         # Créer un diagramme à barres pour comparer les proportions d'anomalies par type de nœud
#         node_types = ['PEAG', 'OLT']
#         anomaly_percentages = []

#         if st.session_state.results["peag_nro"] is not None:
#             df = st.session_state.results["peag_nro"]
#             p_val_min = df[['p_val_avg_dns_time', 'p_val_avg_score_scoring', 'p_val_avg_latence_scoring']].min(axis=1)
#             condition = (p_val_min < st.session_state.p_value_threshold / 100) | (df['avg_isolation_forest_score'] < st.session_state.isolation_forest_threshold)
#             anomaly_percentages.append(round(100 * sum(condition) / len(df), 2))
#         else:
#             anomaly_percentages.append(0)
        
#         if st.session_state.results["olt"] is not None:
#             df = st.session_state.results["olt"]
#             p_val_min = df[['p_val_avg_dns_time', 'p_val_avg_score_scoring', 'p_val_avg_latence_scoring']].min(axis=1)
#             condition = (p_val_min < st.session_state.p_value_threshold / 100) | (df['avg_isolation_forest_score'] < st.session_state.isolation_forest_threshold)
#             anomaly_percentages.append(round(100 * sum(condition) / len(df), 2))
#         else:
#             anomaly_percentages.append(0)
        
#         fig = go.Figure([
#             go.Bar(
#                 x=node_types,
#                 y=anomaly_percentages,
#                 marker_color=['#66B3FF', '#99FF99'],
#                 text=anomaly_percentages,
#                 textposition='auto',
#             )
#         ])
        
#         fig.update_layout(
#             title="Pourcentage de nœuds anormaux par type (IsolationForest + Unidimensionnel)",
#             xaxis_title="Type de nœud",
#             yaxis_title="Pourcentage d'anomalies (%)",
#             yaxis=dict(range=[0, 100]),
#             height=350,
#         )
        
#         st.plotly_chart(fig, use_container_width=True)
    
#     with col2:
#         # Graphique à barres horizontales des métiques affectées
#         metrics = ['avg_dns_time', 'std_dns_time', 'avg_score_scoring', 
#                   'std_score_scoring', 'avg_latence_scoring', 'std_latence_scoring']
        
#         # Noms plus lisibles pour l'affichage
#         display_names = [
#             'Temps DNS moyen', 
#             'Écart-type DNS', 
#             'Score scoring moyen', 
#             'Écart-type scoring', 
#             'Latence scoring moyenne', 
#             'Écart-type latence'
#         ]
        
#         # Compter le nombre d'anomalies par métrique
#         anomaly_counts = [0] * len(metrics)
#         threshold_p_val = st.session_state.p_value_threshold / 100
        
#         for i, metric in enumerate(metrics):
#             p_val_col = f'p_val_{metric}'
            
#             for node_type in ["boucles", "peag_nro", "olt"]:
#                 if st.session_state.results[node_type] is not None and p_val_col in st.session_state.results[node_type].columns:
#                     anomaly_counts[i] += (st.session_state.results[node_type][p_val_col] < threshold_p_val).sum()
        
#         colors = ['#FF9999', '#66B3FF', '#99FF99', '#FFCC99', '#C2C2F0', '#FFB3E6']
        
#         fig = go.Figure()
        
#         fig.add_trace(go.Bar(
#             y=display_names,
#             x=anomaly_counts,
#             orientation='h',
#             marker_color=colors,
#             text=anomaly_counts,
#             textposition='auto',
#         ))
        
#         fig.update_layout(
#             title="Nombre d'anomalies par métrique (détection unidimensionnelle)",
#             xaxis_title="Nombre d'anomalies",
#             yaxis=dict(
#                 categoryorder='total ascending',
#             ),
#             height=350,
#         )
        
#         st.plotly_chart(fig, use_container_width=True)

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

# Fonction pour exécuter la détection d'anomalies
def run_anomaly_detection():
    # Instance de classe : recherche les noeuds anormaux
    nc = NodesChecker()
    
    # Calcul des p_values à partir des distributions empiriques
    lignes_1fev_with_pval = NodesChecker.add_p_values(
        st.session_state.lignes_1fev.copy(), 
        st.session_state.dict_distribution_test
    )

    # Application de l'isolation forest
    lignes_1fev_with_if_scores = st.session_state.isolation_forest.predict(st.session_state.lignes_1fev)
    st.session_state.df_with_anomalies = lignes_1fev_with_if_scores
    
    # Boucles
    # p values
    df_p_values_boucle = nc.get_df_fisher_p_values(
        lignes_1fev_with_pval,
        node_type='boucle',
        p_values=st.session_state.p_values_col
    )
    # scores d'isolation forest
    df_if_scores_boucle = nc.get_if_scores_by_node(lignes_1fev_with_if_scores, node_type='boucle')
    df_results_boucle = pd.merge(df_p_values_boucle, 
                                  df_if_scores_boucle[['avg_isolation_forest_score', 
                                                       'majority_anomaly_score',
                                                       'anomaly_percentage',
                                                       'total_samples',
                                                       'anomaly_count']], 
                                  how='left', 
                                  left_index=True, right_index=True)

    st.session_state.results["boucles"] = df_results_boucle
    
    # Filtrer les boucles défaillantes
    boucles_defaillantes = df_results_boucle[df_results_boucle['avg_isolation_forest_score'] < st.session_state.isolation_forest_threshold].index
    
    if len(boucles_defaillantes) > 0:
        lignes_1fev_with_pval = lignes_1fev_with_pval[~lignes_1fev_with_pval['boucle'].isin(boucles_defaillantes)]
        lignes_1fev_with_if_scores = lignes_1fev_with_if_scores[~lignes_1fev_with_if_scores['boucle'].isin(boucles_defaillantes)]
    
    # PEAG
    # p values
    df_p_values_peag = nc.get_df_fisher_p_values(
        lignes_1fev_with_pval,
        node_type='peag_nro',
        p_values=st.session_state.p_values_col
    )
    # scores d'isolation forest
    df_if_scores_peag = nc.get_if_scores_by_node(lignes_1fev_with_if_scores, node_type='peag_nro')
    df_results_peag = pd.merge(df_p_values_peag, 
                                  df_if_scores_peag[['avg_isolation_forest_score', 
                                                       'majority_anomaly_score',
                                                       'anomaly_percentage',
                                                       'total_samples',
                                                       'anomaly_count']], 
                                  how='left', 
                                  left_index=True, right_index=True)
    st.session_state.results["peag_nro"] = df_results_peag
    
    # Filtrer les PEAG défaillants
    peag_defaillants = df_results_peag[df_results_peag['avg_isolation_forest_score'] < st.session_state.isolation_forest_threshold].index
   
    if len(peag_defaillants) > 0:
        lignes_1fev_with_pval = lignes_1fev_with_pval[~lignes_1fev_with_pval['peag_nro'].isin(peag_defaillants)]
        lignes_1fev_with_if_scores = lignes_1fev_with_if_scores[~lignes_1fev_with_if_scores['peag_nro'].isin(peag_defaillants)]
    
    # OLT
    # p values
    df_p_values_olt = nc.get_df_fisher_p_values(
        lignes_1fev_with_pval,
        node_type='olt_name',
        p_values=st.session_state.p_values_col
    )
    # scores d'isolation forest
    df_if_scores_olt = nc.get_if_scores_by_node(lignes_1fev_with_if_scores, node_type='olt_name')
    df_results_olt = pd.merge(df_p_values_olt, 
                                  df_if_scores_olt[['avg_isolation_forest_score', 
                                                       'majority_anomaly_score',
                                                       'anomaly_percentage',
                                                       'total_samples',
                                                       'anomaly_count']], 
                                  how='left', 
                                  left_index=True, right_index=True)
    st.session_state.results["olt"] = df_results_olt

# Main pour organisation de l'interface
def main():
    # Initialiser l'état de session
    initialize_session_state()
    
    # Afficher l'en-tête
    show_header()
    
    # Créer le menu de navigation horizontal - nouvel ordre
    menu = ["📊 Tableau de bord", "⚙️ Configuration et détection", "📈 Résultats", "🔧 Insertion d'anomalies (Optionnel)", "❓ Aide"]
    choice = st.radio("Navigation", menu, horizontal=True)
    
    st.markdown("---")
    
    # Afficher la page correspondante au choix
    if choice == "📊 Tableau de bord":
        show_home()
    elif choice == "⚙️ Configuration et détection":
        show_detection_config()
    elif choice == "📈 Résultats":
        show_results()
    elif choice == "🔧 Insertion d'anomalies (Optionnel)":
        show_anomaly_insertion()
    elif choice == "❓ Aide":
        show_help()

if __name__ == "__main__":
    main()
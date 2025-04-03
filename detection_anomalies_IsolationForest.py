"""
Module de détection d'anomalies multidimensionnelles avec un Isolation Forest.
Conçu pour s'intégrer à l'architecture existante du Challenge Nexialog.
"""
from sklearn.ensemble import IsolationForest
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import joblib

class IsolationForestDetector:
    """
    Classe pour la détection d'anomalies multidimensionnelles avec un Isolation Forest.
    """
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.features = ['avg_dns_time', 'std_dns_time', 'avg_latence_scoring',
                        'std_latence_scoring', 'avg_score_scoring', 'std_score_scoring']
    
    def train_model(self, df, contamination=0.02, min_samples=100):
        """
        Entraîne un modèle Isolation Forest sur le dataframe fourni
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame contenant les données
        contamination : float, optionnel
            Proportion estimée d'anomalies dans le jeu de données (entre 0 et 0.5)
        min_samples : int, optionnel
            Nombre minimum d'observations par combinaison pour l'inclure dans l'analyse
        
        Returns:
        --------
        self : retourne l'instance elle-même pour permettre le chaînage des méthodes
        """
        # Préparation des données
        df = df.copy()
        df[self.features] = df[self.features].round(3)  # Éviter les "faux" doublons
        
        # Filtrage des combinaisons avec trop peu d'observations
        if 'PEAG_OLT_PEBIB' not in df.columns:
            df['PEAG_OLT_PEBIB'] = df['peag_nro'] + '_' + df['olt_name'] + '_' + df['pebib'] + '_' + df['boucle']
        
        counts = df['PEAG_OLT_PEBIB'].value_counts()
        valid_combinations = counts[counts >= min_samples].index
        df_filtered = df[df['PEAG_OLT_PEBIB'].isin(valid_combinations)]
        
        # Suppression des lignes avec écart-type à zéro
        std_cols = [col for col in self.features if col.startswith('std_')]
        for col in std_cols:
            df_filtered = df_filtered[df_filtered[col] > 0]
        
        # Normalisation des données
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(df_filtered[self.features])
        
        # Training du modèle
        self.model = IsolationForest(
            n_estimators=100,
            max_samples='auto',
            contamination=contamination,
            max_features=len(self.features),
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X)
        
        # Sauvegarde du modèle et du scaler
        joblib.dump(self.model, 'isolation_forest_model.joblib')
        joblib.dump(self.scaler, 'scaler.joblib')
        
        print(f"Modèle entraîné sur {len(df_filtered)} observations filtrées (sur {len(df)} initiales)")
        return self
    
    def predict(self, df):
        """
        Prédit les anomalies sur le dataframe fourni
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame contenant les données à analyser
        
        Returns:
        --------
        pandas.DataFrame : DataFrame d'origine avec colonnes ajoutées pour les scores d'anomalie
        """
        if self.model is None:
            try:
                self.model = joblib.load('isolation_forest_model.joblib')
                self.scaler = joblib.load('scaler.joblib')
            except FileNotFoundError:
                raise RuntimeError("Modèle non entraîné. Appelez train_model() d'abord.")
        
        df_copy = df.copy()
        X = self.scaler.transform(df_copy[self.features])
        
        # Prédiction et ajout des scores
        df_copy['anomaly_score'] = self.model.predict(X)  # -1 pour anomalie, 1 pour normal
        df_copy['isolation_forest_score'] = self.model.decision_function(X)  # Score continu, plus négatif = plus anomal
        
        return df_copy
    
    def create_3d_plot(self, df, test_name, selected_features):
        """
        Crée un graphique 3D pour visualiser les anomalies dans un espace à 2 dimensions + densité
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame contenant les données avec les scores d'anomalie
        test_name : str
            Identifiant de la chaîne technique à visualiser
        selected_features : list
            Liste de 2 variables à représenter sur les axes X et Y
        
        Returns:
        --------
        plotly.graph_objects.Figure : Figure 3D avec la surface de densité et les points d'anomalie
        """
        if len(selected_features) != 2:
            raise ValueError("Exactement 2 features doivent être sélectionnés")
        
        # Filtrer les données pour la chaîne technique spécifique
        row_to_plot = df[df['name'] == test_name]
        
        if len(row_to_plot) == 0:
            raise ValueError(f"Aucune donnée trouvée pour la chaîne technique {test_name}")
        
        feature_data = df[selected_features].values
        feature_data_normalized = self.scaler.transform(df[self.features])[:, [self.features.index(f) for f in selected_features]]
        
        # grille 2D pour la visualisation
        x_min, x_max = np.percentile(feature_data[:, 0], [1, 99])
        y_min, y_max = np.percentile(feature_data[:, 1], [1, 99])
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 100),
            np.linspace(y_min, y_max, 100)
        )
        
        # Préparer les données pour la prédiction de la grille
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        grid_features = np.zeros((grid_points.shape[0], len(self.features)))
        
        # Calculer la moyenne pour chaque feature dans l'ensemble des données
        mean_features = df[self.features].mean().values
        
        # Remplir la grille avec les valeurs moyennes pour toutes les features
        for i in range(len(self.features)):
            grid_features[:, i] = mean_features[i]
        
        # Remplacer les deux features sélectionnées par les valeurs de la grille
        for i, feature in enumerate(selected_features):
            idx = self.features.index(feature)
            grid_features[:, idx] = grid_points[:, i]
        
        grid_features_normalized = self.scaler.transform(grid_features)
        
        # Calculer le score d'anomalie pour chaque point de la grille
        grid_scores = self.model.decision_function(grid_features_normalized)
        Z = grid_scores.reshape(xx.shape)
        
        # Normaliser les scores pour la visualisation (plus le score est négatif, plus c'est anormal)
        # On inverse pour que les anomalies forment des "montagnes" plutôt que des "vallées"
        Z_normalized = -Z  # Inversion pour représenter les anomalies comme des pics
        Z_normalized = (Z_normalized - Z_normalized.min()) / (Z_normalized.max() - Z_normalized.min())
        
        fig = go.Figure()
        
        # Ajouter la surface de score d'anomalie (les pics sont des anomalies potentielles)
        fig.add_trace(
            go.Surface(
                x=xx, y=yy, z=Z_normalized,
                colorscale='Plasma',
                opacity=0.8,
                showscale=True,
                colorbar=dict(
                    title="Score d'anomalie",
                    titleside="right"
                )
            )
        )
        
        # Ajouter le point de la chaîne technique à évaluer
        x_point = float(row_to_plot[selected_features[0]])
        y_point = float(row_to_plot[selected_features[1]])
        
        # Calculer le score d'anomalie du point spécifique
        point_features = row_to_plot[self.features].values[0].reshape(1, -1)
        point_features_normalized = self.scaler.transform(point_features)
        point_score = self.model.decision_function(point_features_normalized)[0]
        
        # Convertir le score en coordonnée Z normalisée pour l'affichage
        point_score_normalized = -point_score  # Inversion pour représenter les anomalies comme des pics
        point_score_normalized = (point_score_normalized - Z.min()) / (Z.max() - Z.min())
        
        # Déterminer si c'est une anomalie
        is_anomaly = row_to_plot['anomaly_score'].values[0] == -1
        marker_color = 'red' if is_anomaly else 'green'
        marker_text = "Anomalie" if is_anomaly else "Normal"
        
        fig.add_trace(
            go.Scatter3d(
                x=[x_point],
                y=[y_point],
                z=[point_score_normalized + 0.05],  # Légèrement au-dessus pour la visibilité
                mode='markers+text',
                text=[marker_text],
                marker=dict(size=8, color=marker_color),
                textposition="top center"
            )
        )
        
        fig.update_layout(
            title=f'Détection d\'anomalies par Isolation Forest pour {test_name}',
            scene=dict(
                xaxis_title=selected_features[0],
                yaxis_title=selected_features[1],
                zaxis_title='Score d\'anomalie',
                aspectratio=dict(x=1, y=1, z=0.8)
            ),
            width=900,
            height=700,
            margin=dict(l=0, r=0, b=0, t=30)
        )
        
        return fig
    
    def get_anomalies_by_node(self, df, node_type, threshold=0.05):
        """
        Identifie les nœuds anormaux en agrégeant les scores par nœud
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame avec scores d'anomalie générés par predict()
        node_type : str
            Type de nœud à analyser ('boucle', 'peag_nro' ou 'olt_name')
        threshold : float, optionnel
            Seuil pour considérer un nœud comme anormal (basé sur la proportion d'anomalies)
        
        Returns:
        --------
        pandas.DataFrame : DataFrame avec les nœuds et leurs scores d'anomalie
        """
        # Calculer la proportion d'anomalies par nœud
        anomaly_count = df[df['anomaly_score'] == -1].groupby(node_type).size()
        total_count = df.groupby(node_type).size()
        
        result = pd.DataFrame({
            'anomaly_count': anomaly_count,
            'total_count': total_count,
            'proportion': (anomaly_count / total_count).fillna(0),
            'mean_isolation_forest_score': df.groupby(node_type)['isolation_forest_score'].mean()
        })
        
        # Tri par proportion d'anomalies décroissante
        result = result.sort_values('proportion', ascending=False)
        result['is_anomalous'] = result['proportion'] > threshold
        
        return result
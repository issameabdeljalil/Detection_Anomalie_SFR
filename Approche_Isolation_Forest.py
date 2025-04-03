from sklearn.ensemble import IsolationForest
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import plotly.graph_objects as go

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
        Train du Isolation Forest sur le jeu de donnée
        
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
        None, mais enregistre le modèle et le scaler pour usage ultérieur
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
        
        # Entraînement du modèle
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
        
        # Obtenir les valeurs pour les axes X et Y du modèle d'Isolation Forest
        feature_data = df[selected_features].values
        feature_data_normalized = self.scaler.transform(df[self.features])[:, [self.features.index(f) for f in selected_features]]
        
        # Créer une grille 2D pour la visualisation
        x_min, x_max = np.percentile(feature_data[:, 0], [1, 99])
        y_min, y_max = np.percentile(feature_data[:, 1], [1, 99])
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 100),
            np.linspace(y_min, y_max, 100)
        )
        
        # Préparer les données pour la prédiction de la grille
        grid_data = np.c_[xx.ravel(), yy.ravel()]
        temp_data = np.zeros((grid_data.shape[0], len(self.features)))
        temp_data[:, [self.features.index(f) for f in selected_features]] = grid_data
        
        # Calculer le score d'anomalie pour chaque point de la grille
        grid_scores = self.model.decision_function(temp_data)
        Z = grid_scores.reshape(xx.shape)
        
        # Normaliser les scores pour la visualisation
        Z_normalized = (Z - Z.min()) / (Z.max() - Z.min())
        
        # Créer la figure 3D
        fig = go.Figure()
        
        # Ajouter la surface de densité (inversée pour que les anomalies soient des "vallées")
        fig.add_trace(
            go.Surface(
                x=xx, y=yy, z=1-Z_normalized,
                colorscale='Plasma',
                opacity=0.8,
                showscale=True,
                colorbar=dict(
                    title="Anomalie",
                    titleside="right"
                )
            )
        )
        
        # Ajouter le point de la chaîne technique à évaluer
        x_point = float(row_to_plot[selected_features[0]])
        y_point = float(row_to_plot[selected_features[1]])
        
        # Trouver sa valeur Z dans le modèle (inversée comme la surface)
        point_score = row_to_plot['isolation_forest_score'].values[0]
        point_score_normalized = (point_score - Z.min()) / (Z.max() - Z.min())
        z_point = 1 - point_score_normalized
        
        # Déterminer si c'est une anomalie
        is_anomaly = row_to_plot['anomaly_score'].values[0] == -1
        marker_color = 'red' if is_anomaly else 'green'
        marker_text = "Anomalie" if is_anomaly else "Normal"
        
        # Ajouter le point à la visualisation
        fig.add_trace(
            go.Scatter3d(
                x=[x_point],
                y=[y_point],
                z=[z_point + 0.05],  # Légèrement au-dessus pour la visibilité
                mode='markers+text',
                text=[marker_text],
                marker=dict(size=8, color=marker_color),
                textposition="top center"
            )
        )
        
        # Configurer la mise en page
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
        
        proportion = (anomaly_count / total_count).fillna(0)
        mean_score = df.groupby(node_type)['isolation_forest_score'].mean()
        
        result = pd.DataFrame({
            'anomaly_count': anomaly_count,
            'total_count': total_count,
            'proportion': proportion,
            'mean_isolation_forest_score': mean_score
        })
        
        # Tri par proportion d'anomalies décroissante
        result = result.sort_values('proportion', ascending=False)
        result['is_anomalous'] = result['proportion'] > threshold
        
        return result
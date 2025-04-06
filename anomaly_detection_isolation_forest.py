"""
Module de détection d'anomalies multidimensionnelles avec un Isolation Forest par chaîne technique.
Conçu pour s'intégrer à l'architecture existante du Challenge Nexialog.
"""
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.filters.hp_filter import hpfilter
import plotly.graph_objects as go
from tqdm import tqdm

class MultiIsolationForestDetector:
    """
    Classe pour la détection d'anomalies multidimensionnelles avec un Isolation Forest
    Un modèle distinct est créé pour chaque chaine technique
    """
    
    def __init__(self, chain_id_col = 'PEAG_OLT_PEBIB'):
        self.models = {}  # Dictionnaire pour stocker les modèles par chaîne technique
        self.scalers = {}  # Dictionnaire pour stocker les scalers par chaîne technique
        self.features = ['avg_dns_time', 'std_dns_time', 'avg_latence_scoring',
                         'std_latence_scoring', 'avg_score_scoring', 'std_score_scoring']
        self.chain_id_col = chain_id_col  # Colonne identifiant la chaîne technique
        self.trend_hp = {} # trend hp à enlever lors de la prédiction

        # Créer le répertoire models s'il n'existe pas
        os.makedirs('models', exist_ok=True)
    
    def _hp_preprocess_data(self, df, chain):
        """
        Prétraite les données en appliquant le filtre Hodrick-Prescott pour une chaîne spécifique
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame contenant les données
        chain : str
            Identifiant de la chaîne technique
            
        Returns:
        --------
        pandas.DataFrame : DataFrame prétraité
        """
        df_processed = df.copy()
        
        # Filtrer les données pour cette chaîne spécifique
        chain_data = df_processed[df_processed[self.chain_id_col] == chain]
        
        self.trend_hp[chain] = {}

        # Appliquer le filtre HP sur les données chronologiques propres à cette chaîne
        for feature in self.features:
            # S'assurer que les données sont triées chronologiquement
            if 'date_hour' in chain_data.columns:
                chain_data = chain_data.sort_values('date_hour')
            
            # cas où il ya pas assez de données sur une chaine technique
            if len(chain_data[feature]) < 2:
                cycle_hp = chain_data[feature]
                self.trend_hp[chain][feature] = 0
            else:
                # Appliquer le filtre HP uniquement sur les données de cette chaîne
                cycle_hp, trend_hp = hpfilter(chain_data[feature], lamb=1000)
        
                self.trend_hp[chain][feature] = trend_hp.iloc[-1] # stockage du trend de chaque chain pour chaque feature

            # Mettre à jour les valeurs dans le DataFrame original
            chain_data[feature] = cycle_hp
            
        
        # Mettre à jour le DataFrame d'origine avec les valeurs filtrées
        df_processed.loc[df_processed[self.chain_id_col] == chain, self.features] = chain_data[self.features]
                
        return df_processed
    
    def train_models(self, df, contamination=0.02, min_samples=100):
        """
        Entraîne un modèle Isolation Forest pour chaque chaîne technique
        
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
        # S'assurer que la colonne d'identification des chaînes existe
        if self.chain_id_col not in df.columns:
            raise ValueError(f"La colonne {self.chain_id_col} n'existe pas dans le DataFrame")
        
        # Identifier toutes les chaînes techniques uniques
        chains = df[self.chain_id_col].unique()
        
        print(f"Entraînement de {len(chains)} modèles d'Isolation Forest...")
        
        # Pour chaque chaîne technique
        for chain in tqdm(chains):
            # Filtrer les données pour cette chaîne
            chain_data = df[df[self.chain_id_col] == chain]
            
            # Vérifier si assez d'observations pour cette chaîne
            if len(chain_data) < min_samples:
                continue
                
            # Prétraitement des données spécifique à cette chaîne
            chain_data = self._hp_preprocess_data(chain_data, chain)
            
            # Normalisation des données
            scaler = StandardScaler()
            X = scaler.fit_transform(chain_data[self.features])
            
            # Entraînement du modèle
            model = IsolationForest(
                n_estimators=100,
                max_samples='auto',
                contamination=contamination,
                max_features=len(self.features),
                random_state=999,
                n_jobs=-1
            )
            model.fit(X)
            
            # Stockage du modèle et du scaler
            self.models[chain] = model
            self.scalers[chain] = scaler
            
            # Sauvegarde du modèle et du scaler
            model_filename = f"models/isolation_forest_{chain.replace('/', '_')}.joblib"
            scaler_filename = f"models/scaler_{chain.replace('/', '_')}.joblib"
            
            joblib.dump(model, model_filename)
            joblib.dump(scaler, scaler_filename)
        
        print(f"Modèles entraînés pour {len(self.models)} chaînes techniques sur {len(chains)} possibles")
        return self
    
    def load_models(self, df=None):
        """
        Charge les modèles sauvegardés pour toutes les chaînes techniques présentes dans df
        Si df est None, tente de charger tous les modèles dans le répertoire models/
        
        Parameters:
        -----------
        df : pandas.DataFrame, optionnel
            DataFrame contenant les chaînes techniques à charger
            
        Returns:
        --------
        self : retourne l'instance elle-même pour permettre le chaînage des méthodes
        """
        if df is not None and self.chain_id_col in df.columns:
            chains = df[self.chain_id_col].unique()
            
            for chain in chains:
                try:
                    model_filename = f"models/isolation_forest_{chain.replace('/', '_')}.joblib"
                    scaler_filename = f"models/scaler_{chain.replace('/', '_')}.joblib"
                    
                    self.models[chain] = joblib.load(model_filename)
                    self.scalers[chain] = joblib.load(scaler_filename)
                except FileNotFoundError:
                    pass
        else:
            # Charger tous les modèles disponibles
            model_files = [f for f in os.listdir('models') if f.startswith('isolation_forest_')]
            
            for model_file in model_files:
                chain = model_file.replace('isolation_forest_', '').replace('.joblib', '').replace('_', '/')
                scaler_file = f"scaler_{chain.replace('/', '_')}.joblib"
                
                try:
                    self.models[chain] = joblib.load(f"models/{model_file}")
                    self.scalers[chain] = joblib.load(f"models/{scaler_file}")
                except FileNotFoundError:
                    pass
        
        print(f"Chargement de {len(self.models)} modèles d'Isolation Forest")
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
        # S'assurer que la colonne d'identification des chaînes existe
        if self.chain_id_col not in df.columns:
            raise ValueError(f"La colonne {self.chain_id_col} n'existe pas dans le DataFrame")
        
        df_copy = df.copy()
        
        # Initialiser les colonnes de résultats
        df_copy['anomaly_score'] = np.nan
        df_copy['isolation_forest_score'] = np.nan
        
        
        # Prédire pour chaque chaîne technique
        for chain in df_copy[self.chain_id_col].unique():
            # Filtrer les données pour cette chaîne
            mask = df_copy[self.chain_id_col] == chain
            chain_data = df_copy.loc[mask]
            
            # enlever le trend HP aux 6 séries
            for feature in self.features:
                try:
                    # Soustraire la tendance HP
                    chain_data[feature] = chain_data[feature] - self.trend_hp[chain][feature]
                except:
                    continue
            try:
                # Normaliser les données
                X = self.scalers[chain].transform(chain_data[self.features])
                
                # Prédire et assigner les scores
                model = self.models[chain]
                df_copy.loc[mask, 'anomaly_score'] = model.predict(X)  # -1 pour anomalie, 1 pour normal
                df_copy.loc[mask, 'isolation_forest_score'] = model.decision_function(X)  # Score continu, plus négatif = plus anomal
            except:
                continue
        
        return df_copy
    
    def get_model_info(self):
        """
        Fournit des informations sur les modèles entraînés
        
        Returns:
        --------
        dict : Dictionnaire contenant des informations sur les modèles
        """
        return {
            "nombre_de_modeles": len(self.models),
            "chaines_techniques": list(self.models.keys())
        }
    
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
    
if __name__ == '__main__':
    
    # Chargement des données
    df = pd.read_csv('data/raw/new_df_final.csv')

    col_list = ['avg_dns_time', 'std_dns_time', 'nb_test_scoring','nb_test_dns', 'avg_latence_scoring',
        'std_latence_scoring', 'avg_score_scoring', 'std_score_scoring']
    
    detector = MultiIsolationForestDetector()
    # detector.train_models(df, contamination=0.005, min_samples=100)

    lignes_1fev = pd.read_csv('data/results/lignes_1fev.csv', index_col=0)
    lignes_1fev = lignes_1fev.rename(columns={'name': 'PEAG_OLT_PEBIB'})

    # Prédiction des anomalies
    detector.load_models(lignes_1fev)
    results = detector.predict(lignes_1fev)
    print(detector.get_model_info()['nombre_de_modeles'])

    # Affichage des résultats
    print(f"Nombre d'anomalies détectées: {(results['anomaly_score'] == -1).sum()}")
    print(f"Pourcentage d'anomalies: {(results['anomaly_score'] == -1).mean() * 100:.2f}%")
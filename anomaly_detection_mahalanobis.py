"""
Module de détection d'anomalies utilisant la distance de Mahalanobis robuste
avec l'estimateur MCD (Minimum Covariance Determinant).
"""
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.covariance import MinCovDet
from scipy.stats import chi2
from statsmodels.tsa.filters.hp_filter import hpfilter
import plotly.graph_objects as go
from tqdm import tqdm

class MahalanobisDetector:
    """
    Classe pour la détection d'anomalies basée sur la distance de Mahalanobis
    avec MCD pour la robustesse aux outliers. Un modèle distinct est créé
    pour chaque chaîne technique.
    """
    
    def __init__(self, chain_id_col='PEAG_OLT_PEBIB', threshold=0.01):
        self.models = {}  # Dictionnaire pour stocker les modèles par chaîne technique
        self.features = ['avg_dns_time', 'std_dns_time', 'avg_latence_scoring',
                         'std_latence_scoring', 'avg_score_scoring', 'std_score_scoring']
        self.chain_id_col = chain_id_col  # Colonne identifiant la chaîne technique
        self.trend_hp = {}  # Trend HP à enlever lors de la prédiction
        self.threshold = threshold  # Seuil de p-value pour les anomalies
        
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
            
            # Récupérer la série et remplacer les valeurs NaN/Inf
            series = chain_data[feature].replace([np.inf, -np.inf], np.nan)
            series = series.ffill().bfill()  # Forward fill puis backward fill
            
            # Cas où il n'y a pas assez de données
            if len(series) < 2:
                cycle_hp = series - series.mean() if len(series) > 0 else series
                self.trend_hp[chain][feature] = series.mean() if len(series) > 0 else 0
            else:
                try:
                    # Appliquer le filtre HP
                    cycle_hp, trend_hp = hpfilter(series, lamb=1000)
                    self.trend_hp[chain][feature] = trend_hp.iloc[-1]
                except Exception as e:
                    # En cas d'erreur, utiliser le centrage simple
                    cycle_hp = series - series.mean()
                    self.trend_hp[chain][feature] = series.mean()

            # Mettre à jour les valeurs dans le DataFrame
            chain_data[feature] = cycle_hp
        
        # Mettre à jour le DataFrame d'origine
        df_processed.loc[df_processed[self.chain_id_col] == chain, self.features] = chain_data[self.features]
                
        return df_processed
    
    def train_models(self, df, support_fraction=0.8, min_samples=100):
        """
        Entraîne un modèle MCD pour chaque chaîne technique en ne conservant que les
        variables à variance suffisante pour chaque chaîne.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame contenant les données
        support_fraction : float, optionnel
            Fraction des données à utiliser pour l'estimation (entre 0.5 et 1)
        min_samples : int, optionnel
            Nombre minimum d'observations par combinaison pour l'inclure
        
        Returns:
        --------
        self : retourne l'instance elle-même
        """
        # S'assurer que la colonne d'identification des chaînes existe
        if self.chain_id_col not in df.columns:
            raise ValueError(f"La colonne {self.chain_id_col} n'existe pas dans le DataFrame")
        
        # Identifier toutes les chaînes techniques uniques
        chains = df[self.chain_id_col].unique()
        
        print(f"Entraînement de {len(chains)} modèles Mahalanobis-MCD...")
        
        success_count = 0
        error_count = 0
        ignored_count = 0
        pinv_count = 0
        
        # Pour chaque chaîne technique
        for chain in tqdm(chains):
            # Filtrer les données pour cette chaîne
            chain_data = df[df[self.chain_id_col] == chain]
            
            # Vérifier si assez d'observations pour cette chaîne
            if len(chain_data) < min_samples:
                ignored_count += 1
                continue
                
            # Prétraitement des données spécifique à cette chaîne
            chain_data = self._hp_preprocess_data(chain_data, chain)
            
            # Extraire les features
            X = chain_data[self.features].values
            
            # Remplacer les valeurs NaN ou Inf
            X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
            
            # Identifier les colonnes à variance suffisante
            variances = np.var(X, axis=0)
            valid_features_idx = np.where(variances >= 1e-8)[0]
            
            # S'il y a moins de 2 features valides, utiliser le pseudo-inverse
            if len(valid_features_idx) < 2:
                try:
                    # Calculer la moyenne et la matrice de covariance
                    mean = np.mean(X, axis=0)
                    cov = np.cov(X, rowvar=False)
                    
                    # Utiliser le pseudo-inverse au lieu de l'inverse normal
                    pinv_cov = np.linalg.pinv(cov)
                    
                    # Stocker le modèle sous forme de pseudo-inverse
                    self.models[chain] = {
                        'model_type': 'pinv',
                        'mean': mean,
                        'pinv_cov': pinv_cov,
                        'features': self.features
                    }
                    
                    pinv_count += 1
                    success_count += 1
                except Exception as e:
                    error_count += 1
                    continue
            else:
                # Cas standard - utiliser MinCovDet
                X_filtered = X[:, valid_features_idx]
                selected_features = [self.features[i] for i in valid_features_idx]
                
                # Ajuster le support_fraction si nécessaire
                min_support = max(0.5, (len(valid_features_idx) + 1) / X_filtered.shape[0])
                actual_support = max(min_support, support_fraction)
                
                try:
                    # Entraîner le modèle MCD
                    mcd = MinCovDet(support_fraction=actual_support, random_state=42)
                    mcd.fit(X_filtered)
                    
                    # Stocker le modèle et les informations sur les features
                    self.models[chain] = {
                        'model_type': 'standard',
                        'model': mcd,
                        'features_idx': valid_features_idx,
                        'features': selected_features
                    }
                    
                    # Sauvegarde du modèle
                    model_filename = f"models/mahalanobis_{chain.replace('/', '_')}.joblib"
                    joblib.dump(self.models[chain], model_filename)
                    
                    success_count += 1
                except Exception as e:
                    error_count += 1
                    continue
        
        print(f"Modèles entraînés avec succès: {success_count}/{len(chains)}")
        print(f"Modèles avec pseudo-inverse: {pinv_count}/{success_count}")
        print(f"Erreurs d'entraînement: {error_count}/{len(chains)}")
        print(f"Chaînes ignorées (trop peu d'échantillons): {ignored_count}/{len(chains)}")
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
        self : retourne l'instance elle-même
        """
        if df is not None and self.chain_id_col in df.columns:
            chains = df[self.chain_id_col].unique()
            
            for chain in chains:
                try:
                    model_filename = f"models/mahalanobis_{chain.replace('/', '_')}.joblib"
                    self.models[chain] = joblib.load(model_filename)
                except FileNotFoundError:
                    pass
        else:
            # Charger tous les modèles disponibles
            model_files = [f for f in os.listdir('models') if f.startswith('mahalanobis_')]
            
            for model_file in model_files:
                chain = model_file.replace('mahalanobis_', '').replace('.joblib', '').replace('_', '/')
                
                try:
                    self.models[chain] = joblib.load(f"models/{model_file}")
                except FileNotFoundError:
                    pass
        
        print(f"Chargement de {len(self.models)} modèles Mahalanobis-MCD")
        return self
    
    def predict(self, df, threshold=None):
        """
        Prédit les anomalies sur le dataframe fourni en normalisant les distances de Mahalanobis
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame contenant les données à analyser
        threshold : float, optionnel
            Seuil de p-value pour considérer une observation comme anomalie
        
        Returns:
        --------
        pandas.DataFrame : DataFrame avec colonnes ajoutées pour les distances et p-values
        """
        # S'assurer que la colonne d'identification des chaînes existe
        if self.chain_id_col not in df.columns:
            raise ValueError(f"La colonne {self.chain_id_col} n'existe pas dans le DataFrame")
        
        df_copy = df.copy()
        
        # Utiliser le seuil par défaut si aucun n'est fourni
        if threshold is None:
            threshold = self.threshold
        
        # Initialiser les colonnes de résultats
        df_copy['mahalanobis_distance'] = np.nan
        df_copy['mahalanobis_distance_normalized'] = np.nan
        df_copy['mahalanobis_pvalue'] = np.nan
        df_copy['mahalanobis_anomaly'] = 0  # Default: pas d'anomalie
        
        # Listes pour collecter toutes les distances et p-values
        all_distances = []
        all_pvalues = []
        all_chains_data = {}  # Pour stocker temporairement les données par chaîne
        
        # Prédire pour chaque chaîne technique
        total_chains = 0
        processed_chains = 0
        
        for chain in df_copy[self.chain_id_col].unique():
            total_chains += 1
            if chain not in self.models:
                continue  # Sauter cette chaîne si pas de modèle
                
            # Filtrer les données pour cette chaîne
            mask = df_copy[self.chain_id_col] == chain
            chain_data = df_copy.loc[mask]
            processed_chains += 1
            
            # Enlever le trend HP aux séries
            for feature in self.features:
                try:
                    # Soustraire la tendance HP
                    chain_data.loc[:, feature] = chain_data[feature] - self.trend_hp[chain][feature]
                except:
                    continue
                    
            # Récupérer le modèle et les informations
            model_info = self.models[chain]
            model_type = model_info.get('model_type', 'standard')
            
            # Extraire les features selon le type de modèle
            if model_type == 'standard':
                # Approche standard avec MinCovDet
                model = model_info['model']
                features_idx = model_info['features_idx']
                
                # Extraire uniquement les features utilisées lors de l'entraînement
                X = chain_data[self.features].values
                X_filtered = X[:, features_idx]
                
                # Remplacer les valeurs NaN ou Inf
                X_filtered = np.nan_to_num(X_filtered, nan=0, posinf=0, neginf=0)
                
                # Calculer les distances de Mahalanobis
                try:
                    distances = np.sqrt(model.mahalanobis(X_filtered))
                    
                    # Utiliser un nombre de degrés de liberté ajusté
                    adjusted_dof = min(len(features_idx), 3)  # Valeur plus conservative
                    
                    # Stocker les valeurs pour normalisation ultérieure
                    all_distances.extend(distances)
                    all_chains_data[chain] = {
                        'mask': mask,
                        'distances': distances,
                        'adjusted_dof': adjusted_dof
                    }
                    
                except Exception as e:
                    continue
                    
            elif model_type == 'pinv':
                # Approche avec pseudo-inverse
                mean = model_info['mean']
                pinv_cov = model_info['pinv_cov']
                
                # Extraire toutes les features
                X = chain_data[self.features].values
                
                # Remplacer les valeurs NaN ou Inf
                X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
                
                try:
                    # Calculer les distances de Mahalanobis avec pseudo-inverse
                    diff = X - mean
                    distances = np.sqrt(np.sum(np.dot(diff, pinv_cov) * diff, axis=1))
                    
                    # Utiliser un nombre de degrés de liberté ajusté
                    adjusted_dof = min(np.linalg.matrix_rank(pinv_cov), 3)  # Valeur plus conservative
                    adjusted_dof = max(1, adjusted_dof)  # Assurer que dof est au moins 1
                    
                    # Stocker les valeurs pour normalisation ultérieure
                    all_distances.extend(distances)
                    all_chains_data[chain] = {
                        'mask': mask,
                        'distances': distances,
                        'adjusted_dof': adjusted_dof
                    }
                    
                except Exception as e:
                    continue
        
        print(f"Prédiction effectuée sur {processed_chains}/{total_chains} chaînes")
        
        # Normalisation des distances et calcul des p-values
        if all_distances:
            # Calculer les statistiques globales des distances
            distance_mean = np.mean(all_distances)
            distance_std = np.std(all_distances)
            
            # Normaliser les distances et calculer les p-values pour chaque chaîne
            for chain, data in all_chains_data.items():
                mask = data['mask']
                distances = data['distances']
                adjusted_dof = data['adjusted_dof']
                
                # Normaliser les distances
                normalized_distances = (distances - distance_mean) / distance_std
                
                # Calculer les p-values sur les distances normalisées
                pvalues = [1 - chi2.cdf(d**2, adjusted_dof) for d in normalized_distances]
                
                # Assigner les résultats
                df_copy.loc[mask, 'mahalanobis_distance'] = distances
                df_copy.loc[mask, 'mahalanobis_distance_normalized'] = normalized_distances
                df_copy.loc[mask, 'mahalanobis_pvalue'] = pvalues
                df_copy.loc[mask, 'mahalanobis_anomaly'] = [1 if p < threshold else 0 for p in pvalues]
                
                # Collecter pour les statistiques globales
                all_pvalues.extend(pvalues)
            
            # Statistiques finales sur les anomalies détectées
            anomaly_count = df_copy['mahalanobis_anomaly'].sum()
            anomaly_percentage = (anomaly_count / len(df_copy)) * 100
            print(f"Nombre d'anomalies détectées: {anomaly_count}")
            print(f"Pourcentage d'anomalies: {anomaly_percentage:.2f}%")
        
        return df_copy


    def get_model_info(self):
        """
        Fournit des informations sur les modèles entraînés
        
        Returns:
        --------
        dict : Dictionnaire contenant des informations sur les modèles
        """
        feature_usage = {feature: 0 for feature in self.features}
        model_types = {'standard': 0, 'pinv': 0}
        
        for chain, model_info in self.models.items():
            model_type = model_info.get('model_type', 'standard')
            model_types[model_type] = model_types.get(model_type, 0) + 1
            
            if model_type == 'standard' and 'features' in model_info:
                for feature in model_info['features']:
                    feature_usage[feature] += 1
            elif model_type == 'pinv':
                # Pour le pseudo-inverse, toutes les features sont utilisées
                for feature in self.features:
                    feature_usage[feature] += 1
        
        return {
            "nombre_de_modeles": len(self.models),
            "chaines_techniques": list(self.models.keys()),
            "utilisation_features": feature_usage,
            "types_de_modeles": model_types
        }
    
    def create_3d_plot(self, df, test_name, selected_features):
        """
        Crée un graphique 3D pour visualiser les anomalies dans un espace à 2 dimensions
        
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
        row_to_plot = df[df[self.chain_id_col] == test_name]
        
        if len(row_to_plot) == 0:
            raise ValueError(f"Aucune donnée trouvée pour la chaîne technique {test_name}")
        
        if test_name not in self.models:
            raise ValueError(f"Aucun modèle trouvé pour la chaîne technique {test_name}")
        
        # Récupérer le modèle et les informations sur les features
        model_info = self.models[test_name]
        model_type = model_info.get('model_type', 'standard')
        
        if model_type == 'standard':
            features = model_info['features']
            
            # Vérifier si les deux features sélectionnées sont disponibles
            if not all(f in features for f in selected_features):
                available_features = [f for f in selected_features if f in features]
                missing_features = [f for f in selected_features if f not in features]
                
                if not available_features:
                    raise ValueError(f"Aucune des features sélectionnées n'est disponible pour cette chaîne. Features disponibles: {features}")
                
                # Utiliser la première feature disponible à la place de celle manquante
                if missing_features:
                    for i, f in enumerate(selected_features):
                        if f in missing_features:
                            selected_features[i] = [feat for feat in features if feat not in selected_features][0]
                    print(f"Features ajustées à {selected_features} (originales non disponibles pour cette chaîne)")
            
            # Index des features sélectionnées dans le modèle
            selected_idx = [features.index(f) for f in selected_features]
            
            # Préparer les données pour la chaîne
            X = row_to_plot[features].values
            
            # Coordonnées pour la grille 3D
            x_min, x_max = np.percentile(X[:, selected_idx[0]], [1, 99])
            y_min, y_max = np.percentile(X[:, selected_idx[1]], [1, 99])
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50), np.linspace(y_min, y_max, 50))
            
            # Préparer tous les points de la grille
            grid = np.zeros((xx.size, len(features)))
            for i, feat_idx in enumerate(range(len(features))):
                mean_val = np.mean(X[:, i])
                grid[:, i] = mean_val
            
            # Remplacer les valeurs pour les features sélectionnées
            grid_idx_x = features.index(selected_features[0])
            grid_idx_y = features.index(selected_features[1])
            grid[:, grid_idx_x] = xx.ravel()
            grid[:, grid_idx_y] = yy.ravel()
            
            # Calculer les scores d'anomalie pour la grille
            model = model_info['model']
            grid_distances = model.mahalanobis(grid)
            Z = grid_distances.reshape(xx.shape)
            
        elif model_type == 'pinv':
            # Préparer les données
            X = row_to_plot[self.features].values
            mean = model_info['mean']
            pinv_cov = model_info['pinv_cov']
            
            # Index des features sélectionnées
            feat1_idx = self.features.index(selected_features[0])
            feat2_idx = self.features.index(selected_features[1])
            
            # Coordonnées pour la grille 3D
            x_min, x_max = np.percentile(X[:, feat1_idx], [1, 99])
            y_min, y_max = np.percentile(X[:, feat2_idx], [1, 99])
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50), np.linspace(y_min, y_max, 50))
            
            # Préparer tous les points de la grille
            grid = np.zeros((xx.size, len(self.features)))
            for i in range(len(self.features)):
                grid[:, i] = mean[i]  # Valeur moyenne pour chaque feature
            
            # Remplacer les valeurs pour les features sélectionnées
            grid[:, feat1_idx] = xx.ravel()
            grid[:, feat2_idx] = yy.ravel()
            
            # Calculer les distances de Mahalanobis pour la grille
            diff = grid - mean
            grid_distances = np.sqrt(np.sum(np.dot(diff, pinv_cov) * diff, axis=1))
            Z = grid_distances.reshape(xx.shape)
        
        # Créer la figure
        fig = go.Figure()
        
        # Ajouter la surface 3D des distances de Mahalanobis
        fig.add_trace(go.Surface(
            x=xx, y=yy, z=Z,
            colorscale='Viridis',
            opacity=0.8
        ))
        
        # Ajouter le point actuel
        point_x = float(row_to_plot[selected_features[0]])
        point_y = float(row_to_plot[selected_features[1]])
        
        if model_type == 'standard':
            # Index des features sélectionnées
            idx_x = features.index(selected_features[0])
            idx_y = features.index(selected_features[1])
            
            # Préparer le point pour Mahalanobis
            point_data = np.zeros(len(features))
            for i in range(len(features)):
                point_data[i] = X[0, i]  # Utiliser les valeurs du premier point
            
            # Calculer la distance
            model = model_info['model']
            point_z = float(model.mahalanobis(point_data.reshape(1, -1))[0])
            
        elif model_type == 'pinv':
            # Préparer le point
            point_data = X[0]  # Utiliser le premier point
            
            # Calculer la distance avec pseudo-inverse
            diff = point_data - mean
            point_z = float(np.sqrt(np.sum(np.dot(diff, pinv_cov) * diff)))
        
        # Déterminer si c'est une anomalie
        is_anomaly = row_to_plot['mahalanobis_anomaly'].values[0] == 1 if 'mahalanobis_anomaly' in row_to_plot.columns else False
        marker_color = 'red' if is_anomaly else 'green'
        marker_text = "Anomalie" if is_anomaly else "Normal"
        
        fig.add_trace(go.Scatter3d(
            x=[point_x],
            y=[point_y],
            z=[point_z + 1],  # Légèrement au-dessus pour la visibilité
            mode='markers+text',
            text=[marker_text],
            marker=dict(size=8, color=marker_color),
            textposition="top center"
        ))
        
        fig.update_layout(
            title=f'Détection d\'anomalies par Mahalanobis pour {test_name}',
            scene=dict(
                xaxis_title=selected_features[0],
                yaxis_title=selected_features[1],
                zaxis_title='Distance de Mahalanobis',
                aspectratio=dict(x=1, y=1, z=0.8)
            ),
            width=900,
            height=700,
            margin=dict(l=0, r=0, b=0, t=30)
        )
        
        return fig

if __name__ == '__main__':
    
    # Exemple d'utilisation
    df = pd.read_csv('data/raw/new_df_final.csv')
    
    detector = MahalanobisDetector()
    detector.train_models(df, support_fraction=0.8, min_samples=100)
    
    lignes_1fev = pd.read_csv('data/results/lignes_1fev.csv', index_col=0)
    lignes_1fev = lignes_1fev.rename(columns={'name': 'PEAG_OLT_PEBIB'})
    
    # Prédiction des anomalies
    detector.load_models(lignes_1fev)
    results = detector.predict(lignes_1fev)
    print(f"Nombre d'anomalies détectées: {results['mahalanobis_anomaly'].sum()}")
    print(f"Pourcentage d'anomalies: {results['mahalanobis_anomaly'].mean() * 100:.2f}%")
    
    # Info sur les modèles
    model_info = detector.get_model_info()
    print(f"Nombre de modèles: {model_info['nombre_de_modeles']}")
    print(f"Types de modèles: {model_info['types_de_modeles']}")
    print(f"Utilisation des features: {model_info['utilisation_features']}")
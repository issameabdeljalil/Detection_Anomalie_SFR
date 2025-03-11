from sklearn.ensemble import IsolationForest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import joblib

def detect_anomalies_multidimensional(df, seuil=100, contamination=0.02, features=None):
    """
    Détecte des anomalies multidimensionnelles à l'aide de l'algorithme Isolation Forest
    
    Paramètres:
    df : pandas.DataFrame
        DataFrame contenant les données
    seuil : int, optionnel
        Nombre minimum d'observations par combinaison PEAG_OLT_PEBIB pour l'inclure dans l'analyse
    contamination : float, optionnel
        Proportion estimée d'anomalies dans le jeu de données (entre 0 et 0.5)
    features : list, optionnel
        Liste des colonnes à utiliser pour la détection d'anomalies
        
    Sortie:
        - 'anomaly_score': score d'anomalie (-1 si anomalie, 1 si normal selon le seuil de contamination)
        - 'isolation_forest_score': score brut de l'isolation forest (plus il est négatif, plus c'est anomal)
    """
    initial_shape = df.shape
    print(f'Shape du DF original : {initial_shape}')
    
    # On utilise ces 6 colonnes
    if features is None:
        features = ['avg_dns_time', 'std_dns_time', 'std_latence_scoring', 
                   'std_score_scoring', 'avg_latence_scoring', 'avg_score_scoring']
    
    
    df[features] = df[features].round(3) # Arrondir les valeurs pour éviter des "faux" doublons 
    df = df.drop_duplicates()
    
    #colonne combinant les identifiants
    df['PEAG_OLT_PEBIB'] = df['peag_nro'] + df['olt_name'] + df['pebib'] + df['boucle'] + df['code_departement'].astype(str)
    
    df_counts = df['PEAG_OLT_PEBIB'].value_counts().reset_index() 
    df_counts.columns = ['PEAG_OLT_PEBIB', 'count']
    valid_values = df_counts[df_counts['count'] >= seuil]['PEAG_OLT_PEBIB'] # on ne veut garder que les combinaisons avec le nombre d'observations >= seuil
    df = df[df['PEAG_OLT_PEBIB'].isin(valid_values)]
    
    # on supprime lignes avec std = 0
    std_cols = [col for col in features if col.startswith('std_')]
    for col in std_cols:
        df = df[df[col] != 0]
    
    print(f'Shape du DF filtré : {df.shape}')
    print(f"Suppression de {initial_shape[0] - df.shape[0]} lignes")

    scaler = StandardScaler()
    X = scaler.fit_transform(df[features])
    
    # Train du modèle Isolation Forest
    model = IsolationForest(
        n_estimators=100,           # Nombre d'arbres
        max_samples='auto',         # Nombre d'échantillons pour construire chaque arbre
        contamination=contamination, # Proportion estimée d'anomalies
        max_features=len(features),  # Nombre de features à considérer
        random_state=42,            
        n_jobs=-1                   
    )
    
    model.fit(X)
    
    # Prédiction anomalies (1: normal, -1: anomalie)
    df['anomaly_score'] = model.predict(X)
    df['isolation_forest_score'] = model.decision_function(X) # score brut d'anomalie (plus négatif = plus anomal)
    
    # on sauvegarde les paramètres du modèle et scaler pour les réutiliser
    joblib.dump(model, 'isolation_forest_model.joblib')
    joblib.dump(scaler, 'scaler.joblib')
    
    return df

def plot_anomalies(df, features=None, n_anomalies=10):
    """
    Affiche les anomalies détectées et les distributions des features
    """
    if features is None:
        features = ['avg_dns_time', 'std_dns_time', 'std_latence_scoring', 
                   'std_score_scoring', 'avg_latence_scoring', 'avg_score_scoring']
    
    # Sélection des anomalies et des observations normales
    anomalies = df[df['anomaly_score'] == -1]
    normal = df[df['anomaly_score'] == 1]
    
    print(f"Nombre d'anomalies détectées: {len(anomalies)} ({len(anomalies)/len(df)*100:.2f}%)")
    
    # On affiche les anomalies les plus importantes (scores les plus négatifs)
    top_anomalies = anomalies.sort_values('isolation_forest_score').head(n_anomalies)
    print("\nTop anomalies détectées:")
    print(top_anomalies[['PEAG_OLT_PEBIB', 'date_hour'] + features + ['isolation_forest_score']])
    
    # graphique distributions
    n_features = len(features)
    fig, axes = plt.subplots(n_features, 1, figsize=(12, n_features*3))
    
    for i, feature in enumerate(features):
        ax = axes[i] if n_features > 1 else axes
        
        # Histogramme des valeurs normales
        ax.hist(normal[feature], bins=50, alpha=0.5, color='blue', label='Normal')
        ax.hist(anomalies[feature], bins=50, alpha=0.5, color='red', label='Anomalie')
        
        ax.set_title(f'Distribution de {feature}')
        ax.legend()
    
    plt.tight_layout()
    plt.show()
    
    if n_features > 1:
        import seaborn as sns
        
        plt.figure(figsize=(15, 15))
        sns.pairplot(
            pd.concat([
                anomalies[features + ['anomaly_score']].head(n_anomalies).assign(type='Anomalie'),
                normal[features + ['anomaly_score']].sample(n=n_anomalies).assign(type='Normal')
            ]),
            hue='type',
            palette={'Normal': 'blue', 'Anomalie': 'red'}
        )
        plt.suptitle('Comparaison des anomalies et des observations normales', y=1.02)
        plt.show()
        
def predict_new_anomalies(new_data, features=None):
    """
    Prédit les anomalies sur de nouvelles données en utilisant un modèle pré-entraîné
    
    """
    if features is None:
        features = ['avg_dns_time', 'std_dns_time', 'std_latence_scoring', 
                   'std_score_scoring', 'avg_latence_scoring', 'avg_score_scoring']
    
    model = joblib.load('isolation_forest_model.joblib')
    scaler = joblib.load('scaler.joblib')
    X = scaler.transform(new_data[features])
    
    # Prédiction anomalies
    new_data['anomaly_score'] = model.predict(X)
    new_data['isolation_forest_score'] = model.decision_function(X)
    
    return new_data

if __name__ == "__main__":
    df = pd.read_parquet('data/raw/250203_tests_fixe_dns_sah_202412_202501.parquet', engine="pyarrow")
    df.dropna(inplace=True)

    if 'PEAG_OLT_PEBIB' not in df.columns:
        df['PEAG_OLT_PEBIB'] = df['peag_nro'] + df['olt_name'] + df['pebib'] + df['boucle'] + df['code_departement'].astype(str)

    features = ['avg_dns_time', 'std_dns_time', 'avg_latence_scoring', 
               'std_latence_scoring', 'avg_score_scoring', 'std_score_scoring']
    
    # Détecter les anomalies
    df_with_anomalies = detect_anomalies_multidimensional(
        df, 
        seuil=100,            # Nombre minimum d'observations par combinaison
        contamination=0.02,   # 2% des données sont considérées comme anomalies
        features=features
    )
    
    df_with_anomalies.to_csv("anomalies_multidim.csv", index=False)
    plot_anomalies(df_with_anomalies, features, n_anomalies=15)
    
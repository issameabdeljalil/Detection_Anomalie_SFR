from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import pandas as pd
import joblib

def detect_anomalies_multidimensional(df, seuil=100, contamination=0.02, features=None):
    """
    Détecte des anomalies multidimensionnelles à l'aide de l'algorithme Isolation Forest.
    """
    if features is None:
        features = ['avg_dns_time', 'std_dns_time', 'std_latence_scoring', 
                   'std_score_scoring', 'avg_latence_scoring', 'avg_score_scoring']
    
    df[features] = df[features].round(3)
    df = df.drop_duplicates()
    
    df['PEAG_OLT_PEBIB'] = df['peag_nro'] + df['olt_name'] + df['pebib'] + df['boucle'] + df['code_departement'].astype(str)
    
    df_counts = df['PEAG_OLT_PEBIB'].value_counts().reset_index()
    df_counts.columns = ['PEAG_OLT_PEBIB', 'count']
    valid_values = df_counts[df_counts['count'] >= seuil]['PEAG_OLT_PEBIB']
    df = df[df['PEAG_OLT_PEBIB'].isin(valid_values)]
    
    std_cols = [col for col in features if col.startswith('std_')]
    for col in std_cols:
        df = df[df[col] != 0]

    scaler = StandardScaler()
    X = scaler.fit_transform(df[features])

    model = IsolationForest(n_estimators=100, contamination=contamination, max_features=len(features), random_state=42, n_jobs=-1)
    model.fit(X)

    df['anomaly_score'] = model.predict(X)
    df['isolation_forest_score'] = model.decision_function(X)

    joblib.dump(model, 'isolation_forest_model.joblib')
    joblib.dump(scaler, 'scaler.joblib')

    return df

def predict_new_anomalies(new_data, features=None):
    """
    Prédit les anomalies sur de nouvelles données en utilisant un modèle pré-entraîné.
    """
    if features is None:
        features = ['avg_dns_time', 'std_dns_time', 'std_latence_scoring', 
                   'std_score_scoring', 'avg_latence_scoring', 'avg_score_scoring']
    
    model = joblib.load('isolation_forest_model.joblib')
    scaler = joblib.load('scaler.joblib')
    X = scaler.transform(new_data[features])

    new_data['anomaly_score'] = model.predict(X)
    new_data['isolation_forest_score'] = model.decision_function(X)

    return new_data

if __name__ == "__main__":
    df = pd.read_parquet('data/raw/250203_tests_fixe_dns_sah_202412_202501.parquet', engine="pyarrow")
    df.dropna(inplace=True)

    features = ['avg_dns_time', 'std_dns_time', 'avg_latence_scoring', 
               'std_latence_scoring', 'avg_score_scoring', 'std_score_scoring']

    df_with_anomalies = detect_anomalies_multidimensional(df, seuil=100, contamination=0.02, features=features)
    df_with_anomalies.to_csv("anomalies_multidim.csv", index=False)

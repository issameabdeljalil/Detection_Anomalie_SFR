import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_anomalies(df, features=None, n_anomalies=10):
    """
    Affiche les anomalies détectées et les distributions des features.
    """
    if features is None:
        features = ['avg_dns_time', 'std_dns_time', 'std_latence_scoring', 
                   'std_score_scoring', 'avg_latence_scoring', 'avg_score_scoring']
    
    anomalies = df[df['anomaly_score'] == -1]
    normal = df[df['anomaly_score'] == 1]

    print(f"Nombre d'anomalies détectées: {len(anomalies)} ({len(anomalies)/len(df)*100:.2f}%)")

    top_anomalies = anomalies.sort_values('isolation_forest_score').head(n_anomalies)
    print("\nTop anomalies détectées:")
    print(top_anomalies[['PEAG_OLT_PEBIB', 'date_hour'] + features + ['isolation_forest_score']])

    n_features = len(features)
    fig, axes = plt.subplots(n_features, 1, figsize=(12, n_features*3))

    for i, feature in enumerate(features):
        ax = axes[i] if n_features > 1 else axes
        ax.hist(normal[feature], bins=50, alpha=0.5, color='blue', label='Normal')
        ax.hist(anomalies[feature], bins=50, alpha=0.5, color='red', label='Anomalie')

        ax.set_title(f'Distribution de {feature}')
        ax.legend()

    plt.tight_layout()
    plt.show()

    if n_features > 1:
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

if __name__ == "__main__":
    df_with_anomalies = pd.read_csv("anomalies_multidim.csv")
    features = ['avg_dns_time', 'std_dns_time', 'avg_latence_scoring', 
               'std_latence_scoring', 'avg_score_scoring', 'std_score_scoring']

    plot_anomalies(df_with_anomalies, features, n_anomalies=15)

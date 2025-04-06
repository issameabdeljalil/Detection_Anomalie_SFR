import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.filters.hp_filter import hpfilter
from scipy.stats import gaussian_kde

from utils import export_dict_to_json, import_json_to_dict
from graph_creator import GraphCreator

class AnomalyDetectionOneDim:
    """
    Pour une combinaison donnée : Boucle X PEAG X OLT
    Classe permettant de trouver une anomalie sur chaque dimension des tests :
    avg_dns_time
    avg_score_scoring
    avg_latence_scoring
    std_dns_time
    std_score_scoring
    std_latence_scoring

    Utilise des techniques de filtrage Hodrick-Prescott, de réduction des valeurs aberrantes,
    et d'estimation par noyau (KDE) des distributions empiriques pour calculer des p-values.
    """
    
    def __init__(self):
        pass

    @staticmethod
    def get_test_vectors(df, col_list) -> dict:
        """
        création d'un dict avec colonne de test comme keys et
        en value un autre dict qui contient les PEAG_OLT_PEBIB en keys et leurs vecteurs
        en values
        
        Parameters:
        df : df
        col_list : colonnes associées aux tests
        
        Returns:
        dict : dict avec colonne de test comme keys et
        en value un autre dict qui contient les PEAG_OLT_PEBIB en kyes et leurs vecteurs
        en values
    
        """
        # tri des données
        df_sorted = df.sort_values("date_hour")
        # dict des PEAG_OLT_PEBIB
        result_dict = {col: df_sorted.groupby("PEAG_OLT_PEBIB")[col].apply(list).to_dict() for col in col_list}

        return result_dict
    
    @staticmethod
    def sum_nb_test(dict_test):
        """
        Pour chaque colonne débutant par 'nb_', et pour chaque clé dans cette colonne,
        remplace la valeur (qui doit être une liste de nombres) par la somme de cette liste.

        Args:
            dict_test (dict): dictionnaire imbriqué.

        Returns:
            dict: dictionnaire modifié avec sommes calculées.
        """
        for col in dict_test:
            if col.startswith('nb_'):
                for key in dict_test[col]:
                    dict_test[col][key] = sum(dict_test[col][key])
        return dict_test
    
    @staticmethod
    def filter_anomalies(cycle_hp, coeff_lower:float, coeff_upper:float, percent_lower=2.5, percent_upper=97.5):
        """
        Filtre le vecteur cycle_hp en éliminant progressivement les valeurs aberrantes par un procédé probabiliste.
        Les valeurs comprises entre les percentiles définis par percent_lower et percent_upper (par défaut 2.5% et 97.5%)
        se voient attribuer un poids de 1, tandis que celles situées en dehors de cet intervalle subissent une décroissance
        exponentielle. L'intensité de cette décroissance est régulée par coeff_lower pour les valeurs inférieures et par
        coeff_upper pour les valeurs supérieures, de façon à obtenir un poids proche de 0.01 aux extrémités.

        Pour chaque élément de cycle_hp, un tirage aléatoire détermine sa conservation : la donnée est retenue avec une
        probabilité égale à son poids, ce qui permet de privilégier les valeurs centrales tout en réduisant l'influence des
        valeurs extrêmes.

        Parameters:
        -----------
        cycle_hp : numpy.ndarray
            Vecteur de données à filtrer.
        coeff_lower : float
            Coefficient utilisé pour la décroissance exponentielle des valeurs inférieures au percentile bas.
        coeff_upper : float
            Coefficient utilisé pour la décroissance exponentielle des valeurs supérieures au percentile haut.
        percent_lower : float, optionnel
            Percentile inférieur définissant la borne basse du segment central (par défaut 2.5).
        percent_upper : float, optionnel
            Percentile supérieur définissant la borne haute du segment central (par défaut 97.5).

        Returns:
        --------
        numpy.ndarray
            Vecteur filtré contenant uniquement les données conservées après application du critère probabiliste.
        """
        # calcul des percentiles et des bornes extrêmes
        p_lower = np.percentile(cycle_hp, percent_lower)
        p_upper = np.percentile(cycle_hp, percent_upper)
        min_val = np.min(cycle_hp)
        max_val = np.max(cycle_hp)
        
        # calcul des alphas
        alpha_lower = coeff_lower/ (p_lower - min_val) if (p_lower - min_val) > 0 else 1.0
        alpha_upper = coeff_upper / (max_val - p_upper) if (max_val - p_upper) > 0 else 1.0

        # initialisation du tableau de poids à 1 pour toutes les valeurs
        weights = np.ones_like(cycle_hp)
        
        # pour les valeurs < au p_lower percentile : décroissance exponentielle
        lower_mask = cycle_hp < p_lower
        weights[lower_mask] = np.exp(-alpha_lower * (p_lower - cycle_hp[lower_mask]))
        
        # pour les valeurs > au p_upper percentile : décroissance exponentielle
        upper_mask = cycle_hp > p_upper
        weights[upper_mask] = np.exp(-alpha_upper * (cycle_hp[upper_mask] - p_upper))
        
        # tirage aléatoire pour conserver ou supprimer la donnée
        random_values = np.random.rand(cycle_hp.shape[0])
        keep_mask = random_values < weights
        
        return cycle_hp[keep_mask]
    
    @staticmethod
    def compute_kde(vector, nb_test):
        """
        calcule une estimation par noyau (kde) de la distribution du vecteur
        avec bandwith = silverman_bw
        Parameters:
        -----------
        vector : numpy.ndarray
            Vecteur de données (par exemple, shape=(1456,)).
        nb_test : int
            Nombre de tests réalisé pour cette chaine technique
            
        Returns:
        --------
        kde : scipy.stats.gaussian_kde
            Objet KDE estimant la distribution du vecteur.
        """
        vector_clean = vector[np.isfinite(vector)]
        n = nb_test
        std_dev = np.std(vector_clean, ddof=1)
        # calcul de la largeur de bande selon Silverman
        silverman_bw = (4 / (3 * n)) ** (1 / 5) * std_dev 

        kde = gaussian_kde(vector_clean, bw_method=silverman_bw)
        return kde

    @staticmethod
    def compute_p_value(point, kde, nsamples=100000, alternative="two-sided"):
        """
        Compare un point à la distribution estimée par le KDE et calcule une p-value
        
        Pour ce faire, on génère nsamples échantillons à partir du KDE, on estime la CDF 
        (la proportion d'échantillons <= point) et on définit la p-value selon le type de test choisi :
        
        - "two-sided" : p_value = 2 * min(CDF(point), 1 - CDF(point))
        - "greater"   : p_value = 1 - CDF(point)   (p-value unilatérale à droite)
        - "less"      : p_value = CDF(point)       (p-value unilatérale à gauche)
        
        Parameters:
        -----------
        point : float
            La valeur à tester.
        kde : scipy.stats.gaussian_kde
            L'estimation par noyau de la distribution.
        nsamples : int, optionnel
            Nombre d'échantillons pour approximer la CDF (par défaut 100000).
        alternative : str, optionnel
            Le type de test pour la p-value : "two-sided", "greater", ou "less".
        
        Returns:
        --------
        p_value : float
            La p-value associée au point.
        """
        # Génération d'échantillons à partir du KDE
        samples = kde.resample(nsamples).flatten()
        # Estimation de la CDF
        cdf_val = np.mean(samples <= point)
        
        if alternative == "two-sided":
            p_value = 2 * min(cdf_val, 1 - cdf_val)
        elif alternative == "greater":
            p_value = 1 - cdf_val
        elif alternative == "less":
            p_value = cdf_val
            
        # La p-value ne peut excéder 1
        return min(p_value, 1.0)

    def get_p_values_one_row(self, row, dict_test):
        """
        Calcule les 6 p-values pour une ligne de dataframe
        
        Paramètres
        ----------
        row : pandas.Series
            Ligne de dataframe contenant les colonnes :
            'avg_dns_time', 'avg_score_scoring', 'avg_latence_scoring',
            'std_dns_time', 'std_score_scoring', 'std_latence_scoring'
        dict_test : dictionnaire qui a les distributions de chaque combinaison
        Retourne
        --------
        dict
            Dictionnaire avec les p-values associées à chaque métrique.
        """
        
        # Calcul pour avg_dns_time
        cycle_hp, trend_hp = hpfilter(dict_test['avg_dns_time'][row['name']], lamb=1000)
        cycle_hp_filtered = self.filter_anomalies(cycle_hp, coeff_lower=1, coeff_upper=1, percent_lower=0, percent_upper=99)
        kde = self.compute_kde(cycle_hp_filtered, dict_test['nb_test_dns'][row['name']])
        p_val_dns = self.compute_p_value(row['avg_dns_time'] - trend_hp[-1], kde, alternative='greater')

        # Calcul pour avg_score_scoring
        cycle_hp, trend_hp = hpfilter(dict_test['avg_score_scoring'][row['name']], lamb=1000)
        cycle_hp_filtered = self.filter_anomalies(cycle_hp, coeff_lower=1, coeff_upper=1, percent_lower=1, percent_upper=100)
        kde = self.compute_kde(cycle_hp_filtered, dict_test['nb_test_scoring'][row['name']])
        p_val_score = self.compute_p_value(row['avg_score_scoring'] - trend_hp[-1], kde, alternative='less')

        # Calcul pour avg_latence_scoring
        cycle_hp, trend_hp = hpfilter(dict_test['avg_latence_scoring'][row['name']], lamb=1000)
        cycle_hp_filtered = self.filter_anomalies(cycle_hp, coeff_lower=20, coeff_upper=1, percent_lower=0, percent_upper=95)
        kde = self.compute_kde(cycle_hp_filtered, dict_test['nb_test_scoring'][row['name']])
        p_val_latence = self.compute_p_value(row['avg_latence_scoring'] - trend_hp[-1], kde, alternative='greater')
        
        # Calcul pour std_dns_time
        cycle_hp, trend_hp = hpfilter(dict_test['std_dns_time'][row['name']], lamb=1000)
        cycle_hp_filtered = self.filter_anomalies(cycle_hp, coeff_lower=1, coeff_upper=1, percent_lower=0, percent_upper=100)
        kde = self.compute_kde(cycle_hp_filtered, dict_test['nb_test_dns'][row['name']])
        p_val_std_dns = self.compute_p_value(row['std_dns_time'] - trend_hp[-1], kde, alternative='greater')
        
        # Calcul pour std_score_scoring
        cycle_hp, trend_hp = hpfilter(dict_test['std_score_scoring'][row['name']], lamb=1000)
        cycle_hp_filtered = self.filter_anomalies(cycle_hp, coeff_lower=1, coeff_upper=1, percent_lower=1, percent_upper=99)
        kde = self.compute_kde(cycle_hp_filtered, dict_test['nb_test_scoring'][row['name']])
        p_val_std_score = self.compute_p_value(row['std_score_scoring'] - trend_hp[-1], kde, alternative='greater')
        
        # Calcul pour std_latence_scoring
        cycle_hp, trend_hp = hpfilter(dict_test['std_latence_scoring'][row['name']], lamb=1000)
        cycle_hp_filtered = self.filter_anomalies(cycle_hp, coeff_lower=1, coeff_upper=10, percent_lower=0, percent_upper=95)
        kde = self.compute_kde(cycle_hp_filtered, dict_test['nb_test_scoring'][row['name']])
        p_val_std_latence = self.compute_p_value(row['std_latence_scoring'] - trend_hp[-1], kde, alternative='greater')
        
        return {
            'p_val_avg_dns_time': p_val_dns,
            'p_val_avg_score_scoring': p_val_score,
            'p_val_avg_latence_scoring': p_val_latence,
            'p_val_std_dns_time': p_val_std_dns,
            'p_val_std_score_scoring': p_val_std_score,
            'p_val_std_latence_scoring': p_val_std_latence
        }

if __name__ == '__main__':

    # Preprocessing
    # df = pd.read_csv('data/raw/new_df_final.csv')
    # df = df.groupby('PEAG_OLT_PEBIB').apply(lambda group: group.ffill().bfill())
    # df = df.reset_index(level=0, drop=True)
    # df.to_csv('data/raw/new_df_final.csv')

    col_list = ['avg_dns_time', 'std_dns_time', 'nb_test_scoring','nb_test_dns', 'avg_latence_scoring',
        'std_latence_scoring', 'avg_score_scoring', 'std_score_scoring']
    
    adod = AnomalyDetectionOneDim()

    # # récupération des vecteurs de distribution des tests
    # dict_test = adod.get_test_vectors(df, col_list)
    # # somme les nb de test par combinaison boucle x peag x olt
    # dict_test = adod.sum_nb_test(dict_test)

    # output_path = "data/results/dict_test.json"
    
    # export_dict_to_json(dict_test, output_path)

    input_path = "data/results/dict_test.json"
    dict_test = import_json_to_dict(input_path)

    # quelques graphiques
    # GraphCreator.plot_series(dict_test, '01_peag_301_olt_369_pebib_3BU146401', 'avg_dns_time', title = '_')
    # GraphCreator.plot_score_histogram(dict_test, '01_peag_301_olt_369_pebib_3BU146401', 'avg_dns_time')
    # plt.show()

    # test d'un exemple fictif d'anomalie
    df_t = pd.DataFrame({
        'name':['01_peag_301_olt_369_pebib_3BU146401'],
        'avg_dns_time': [12.98],
        'std_dns_time': [10.61],
        'avg_latence_scoring': [18.52],
        'std_latence_scoring': [10.58],
        'avg_score_scoring': [13.37],
        'std_score_scoring': [10.19]})

    p_values = adod.get_p_values_one_row(df_t.iloc[0], dict_test)
    print(p_values)
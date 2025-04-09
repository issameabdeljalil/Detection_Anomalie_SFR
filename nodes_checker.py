import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from scipy.stats import chi2
from anomaly_detection_one_dim import AnomalyDetectionOneDim # classe de recherche d'anomalies selon la distribution empirique
from anomaly_detection_isolation_forest import MultiIsolationForestDetector # classe pour l'isolation forest par chaine technique
from utils import import_json_to_dict

class NodesChecker:
    """
    Cette classe permet de faire la détection pour un heure précise par noeuds (ex : 'boucle', 'peag_nro', 'olt_name')
    Elle permet d'injecter des anomalies (en ajoutant une valeur spécifique à certaines colonnes pour des noeuds donnés), 
    d'ajouter des p-values issues de tests statistiques, et de combiner ces p-values via le test de Fisher. 
    Permet une correction des p-values selon la méthode de Benjamini-Hochberg et restitue les pvalues corrigées
    sous forme de df par noeud
    """
    def __init__(self):
        pass

    @staticmethod
    def add_anomalies(lignes_1fev: pd.DataFrame,
                    node_col: str,
                    nodes_to_update: list,
                    cols_to_update: list, 
                    value_to_add: float) -> pd.DataFrame:
        """
        Ajoute une anomalie (augmentation d'une valeur) sur des colonnes spécifiques 
        pour certains noeuds
        
        Paramètres:
        -----------
        lignes_1fev : pd.DataFrame
            Le DataFrame contenant les données sur lesquelles l'anomalie sera appliquée.
        node_col : str, optionnel
            Le nom de la colonne qui identifie les noeuds. Par défaut 'olt_name'.
        nodes_to_update : list
            La liste des noeuds (valeurs dans 'node_col') pour lesquels appliquer l'anomalie.
        cols_to_update : list
            La liste des noms de colonnes dans lesquelles l'anomalie (l'ajout de valeur) doit être appliquée.
        value_to_add : float
            La valeur à ajouter aux colonnes spécifiées pour les noeuds ciblés.
        
        Retour:
        --------
        pd.DataFrame : df modifié
        """
        # noeuds à update
        mask = lignes_1fev[node_col].isin(nodes_to_update)
        # ajout de la valeur aux colonnes de test choisies
        lignes_1fev.loc[mask, cols_to_update] += value_to_add
        
        return lignes_1fev
    
    @staticmethod
    def add_p_values(lignes_1fev:pd.DataFrame, dict_test:dict):
        """
        Ajoute les p-values de test pour chaque ligne du DataFrame lignes_1fev.
        
        Parameters:
            lignes_1fev (pd.DataFrame): DataFrame sur lequel ajouter les p-values.
            dict_test (dict): Dictionnaire contenant les vecteurs de distribution pour les tests.
        
        Returns:
            pd.DataFrame: DataFrame enrichi avec les p-values.
        """
        adod = AnomalyDetectionOneDim() # instance de classe de recherche d'anomalies selon la distribution empirique
        for index, row in lignes_1fev.iterrows():
            p_values = adod.get_p_values_one_row(row, dict_test)
            for key in p_values.keys():
                lignes_1fev.loc[index, key] = p_values[key]
        return lignes_1fev
    
    @staticmethod
    def add_isolation_forest_results(lignes_1fev:pd.DataFrame, detector):
        """
        Ajoute les p-values de test pour chaque ligne du DataFrame lignes_1fev.
        
        Parameters:
            lignes_1fev (pd.DataFrame): DataFrame sur lequel ajouter les p-values.
            detector : class MultiIsolationForestDetector() avec models loaded
        
        Returns:
            pd.DataFrame: DataFrame enrichi avec les resultas de l'isolation forest.
        """
        results = detector.predict(lignes_1fev)
        return results

    
    @staticmethod
    def fisher_combined_pvalue(pvalues: np.array) -> float:
        """
        realise le test combiné de Fisher à partir d'un vecteur de p-values
        pour trouver les boucles anormales (p-value combinée)
        
        Parameters:
        -----------
        pvalues : np.array
            un vecteur numpy contenant les p-values à agréger

        Returns:
        --------
        float
            La p-value combinée selon le test de Fisher
        """
        # remplace les 0 par la plus petite valeur positive représentable
        pvalues = np.clip(pvalues, a_min=np.finfo(float).tiny, a_max=1)
        # calcul de la statistique de Fisher
        chi_stat = -2 * np.sum(np.log(pvalues))
        # degrés de liberté : 2 fois le nombre de p-values
        dof = 2 * len(pvalues)
        # calcul de la p-value combinée
        combined_pvalue = 1 - chi2.cdf(chi_stat, dof)

        return combined_pvalue
    
    @staticmethod
    def benjamini_hochberg_correction(pvalues):
        """
        Corrige un vecteur de p-values selon la méthode Benjamini-Hochberg
        
        Paramètres:
        - pvalues: liste ou array numpy de p-values.
        
        Retourne:
        - Array numpy de p-values corrigées.
        """
        pvalues = np.array(pvalues)
        n = len(pvalues)
        
        # Étape 1 : trier les p-values et garder l'index d'origine
        indices_trie = np.argsort(pvalues)
        p_triees = pvalues[indices_trie]
        
        # Étape 2 : calculer la p-value ajustée pour chaque rang
        p_corrigees_trie = np.empty(n, dtype=float)
        for i, p in enumerate(p_triees):
            rang = i + 1  # les rangs commencent à 1
            p_corrigees_trie[i] = p * n / rang
        
        # Étape 3 : assurer la monotonicité : p_corr[i] >= p_corr[i+1]
        p_corrigees_trie = np.minimum.accumulate(p_corrigees_trie[::-1])[::-1]
        
        # Remettre les p-values dans l'ordre d'origine
        p_corrigees = np.empty(n, dtype=float)
        p_corrigees[indices_trie] = np.minimum(p_corrigees_trie, 1.0)
        
        return p_corrigees
    
    def corriger_p_values(self, dict_p_val_fisher: dict):
        """
        Corrige les p-values contenues dans dict_p_val_fisher

        Paramètres :
            dict_p_val_fisher (dict): Dictionnaire contenant les p-values associées à 
            différents noeuds et tests.

        Retourne :
            dict: Le dictionnaire mis à jour avec les p-values corrigées.
        """
        # etape 1 : extraction de toutes les p-values et memoire de leur position
        pvalues_list = []
        indices = []  # Liste de tuples (noeud, nom_test)
        for noeud, tests in dict_p_val_fisher.items():
            for test, p_val in tests.items():
                indices.append((noeud, test))
                pvalues_list.append(p_val)

        # etape 2 : benjamini_hochberg_correction
        pvalues_array = np.array(pvalues_list)
        pvalues_corrigees = self.benjamini_hochberg_correction(pvalues_array)

        # etape 3 : reinjecte les p-values corrected dans dict_p_val_fisher
        for (noeud, test), p_corr in zip(indices, pvalues_corrigees):
            dict_p_val_fisher[noeud][test] = p_corr

        return dict_p_val_fisher

    def get_df_fisher_p_values(self, lignes_1fev_with_pval:pd.DataFrame, node_type:str, p_values:list):
        """
        calcule et renvoie un df contenant les p-values combinées (tests de Fisher) pour chaque noeud, 
        corrigées via la méthode Benjamini-Hochberg afin de limiter le risque de p-hacking.
        
        Paramètres:
        - lignes_1fev_with_pval (pd.DataFrame) : DataFrame contenant les données avec les p-values.
        - node_type (str) : Type de noeud utilisé pour le regroupement des données.
                            Exemple : 'boucle', 'peag_nro' ou 'olt_name'
        - p_values (list) : liste des keys p_values
        
        Retourne:
        - df_p_values (pd.DataFrame) : DataFrame indexé par les noeuds avec en colonnes les p-values combinées corrigées.
        """
        unique_noeuds = lignes_1fev_with_pval[node_type].unique()

        dict_p_val_fisher = {}
        
        for noeud in unique_noeuds:
            dict_p_val_fisher[noeud] = {}
            # un noeud selectionnee
            noeud_mask = lignes_1fev_with_pval[node_type] == noeud
            for p_value in p_values:
                p_array = lignes_1fev_with_pval.loc[noeud_mask, p_value].to_numpy()
                # calcul de la p-value combinée avec Fisher
                dict_p_val_fisher[noeud][p_value] = self.fisher_combined_pvalue(p_array)
        
        # correction des p_values avec la méthode Benjamini-Hochberg (pour éviter p-hacking)
        dict_p_val_fisher = self.corriger_p_values(dict_p_val_fisher)
        
        # conversion en dataframe
        df_p_values = pd.DataFrame.from_dict(dict_p_val_fisher, orient='index')

        return df_p_values
    
    @staticmethod
    def get_if_scores_by_node(lignes_1fev_with_if_scores, node_type:str):
        """
        Agrège les scores d'Isolation Forest par nœud (moyenne pour les scores continus, 
        vote majoritaire pour les indicateurs d'anomalie).
        
        Paramètres:
        - lignes_1fev_with_if_scores (pd.DataFrame) : DataFrame contenant les données avec les scores isolation forest
        - node_type (str) : Type de noeud utilisé pour le regroupement des données.
                            Exemple : 'boucle', 'peag_nro' ou 'olt_name'

        Retourne:
        - df_p_values (pd.DataFrame) : DataFrame indexé par les noeuds avec en colonnes les p-values combinées corrigées.
        """
        unique_noeuds = lignes_1fev_with_if_scores[node_type].unique()
        
        node_results = {}
        
        for node in unique_noeuds:
            # Filtrer les données pour ce nœud
            node_mask = lignes_1fev_with_if_scores[node_type] == node
            node_data = lignes_1fev_with_if_scores.loc[node_mask]
            
            # Calculer la moyenne des scores continus
            avg_isolation_score = node_data['isolation_forest_score'].mean()
            
            # Vote majoritaire pour anomaly_score (-1 pour anomalie, 1 pour normal)
            anomaly_count = (node_data['anomaly_score'] == -1).sum()
            normal_count = (node_data['anomaly_score'] == 1).sum()
            majority_vote = -1 if anomaly_count > normal_count else 1
            
            # Calculer le pourcentage d'anomalies
            anomaly_percentage = (anomaly_count / len(node_data)) * 100 if len(node_data) > 0 else 0
            
            # Stocker les résultats
            node_results[node] = {
                'avg_isolation_forest_score': avg_isolation_score,
                'majority_anomaly_score': majority_vote,
                'anomaly_percentage': anomaly_percentage,
                'total_samples': len(node_data),
                'anomaly_count': anomaly_count
            }
        
        df_node_results = pd.DataFrame.from_dict(node_results, orient='index')
        # Trier par score moyen (du plus anomal au moins anomal)
        df_node_results = df_node_results.sort_values('avg_isolation_forest_score')
        
        return df_node_results

if __name__ == '__main__':

    # df = pd.read_csv('data/raw/new_df_final.csv')

    # ## Simulation

    # # reprise la derniere heure pour simuler le 1er janvier à 00:00:00
    # lignes_31jan = df[df['date_hour'] == '2025-01-31 23:00:00'].copy()
    # lignes_1fev = lignes_31jan.copy()
    # lignes_1fev.rename(columns={'PEAG_OLT_PEBIB':'name'}, inplace=True)
    # lignes_1fev['date_hour'] = '2025-02-01 00:00:00'
    # print('len avant la suppression des noeuds deffaillants:', len(lignes_1fev))
    
    # with open('data/results/dict_deffaillants.json', 'r', encoding='utf-8') as f:
    #     dict_defaillants = json.load(f)

    # lignes_1fev = lignes_1fev[~lignes_1fev['peag_nro'].isin(dict_defaillants['peag_defaillants'])]
    # lignes_1fev = lignes_1fev[~lignes_1fev['olt_name'].isin(dict_defaillants['olt_defaillants'])]
    # lignes_1fev = lignes_1fev[~lignes_1fev['boucle'].isin(dict_defaillants['boucles_defaillantes'])]
    
    # print('len apres la suppression des noeuds deffaillants:', len(lignes_1fev))
    
    # lignes_1fev.to_csv('data/results/lignes_1fev.csv')
    ##

    lignes_1fev = pd.read_csv('data/results/lignes_1fev.csv', index_col=0)

    input_path = "data/results/dict_test.json"
    # importation des vecteurs de distribution par test
    dict_test = import_json_to_dict(input_path)

    # colonnes dans lesquelles injecter des anomalies
    cols_to_update = [
    'avg_score_scoring',
    'avg_latence_scoring',
    'avg_dns_time',
    'std_score_scoring',
    'std_latence_scoring',
    'std_dns_time'
    ]


    # instance de classe : recherche les noeuds anormaux
    nc = NodesChecker()
    # rajout de + 10 de valuers aux tests de 4 peag et 4 olt
    # lignes_1fev = nc.add_anomalies(lignes_1fev, 
    #             'peag_nro', 
    #             ['01_peag_1', '01_peag_2', '01_peag_3', '01_peag_22'],
    #             cols_to_update,
    #             10.0)
    lignes_1fev = nc.add_anomalies(lignes_1fev, 
                'olt_name', 
                ['01_olt_1', '01_olt_2', '01_olt_3', '01_olt_23', '95_olt_5624', 
                 '01_olt_70', '95_olt_5625', '95_olt_5630', '95_olt_5627', '95_olt_5626'],
                cols_to_update,
                10.0)
    # calcul des p_values à partir des distributions empiriques
    lignes_1fev = nc.add_p_values(lignes_1fev, dict_test)

    # calcul des isolation forest scores
    detector = MultiIsolationForestDetector(chain_id_col = 'name')
    detector.load_models(lignes_1fev)
    lignes_1fev_with_if_scores = detector.predict(lignes_1fev)
    
    p_values_col = [
        'p_val_avg_dns_time',
        'p_val_avg_score_scoring',
        'p_val_avg_latence_scoring',
        'p_val_std_dns_time',
        'p_val_std_score_scoring',
        'p_val_std_latence_scoring'
    ]

    # On regarde les boucles defaillantes -> on enlève ces boucles defaillantes
    # On regarde les PEAG defaillants -> on enlève ces PEAG defaillants
    # On regarde les OLT defaillants

    # p_values issues du test de Fisher pour les boucles
    df_p_values_boucle = nc.get_df_fisher_p_values(lignes_1fev, node_type = 'boucle', p_values = p_values_col)

    # scores d'isolation forest
    df_if_scores_boucle = nc.get_if_scores_by_node(lignes_1fev_with_if_scores, node_type = 'boucle')



   
    boucles_defaillantes = df_p_values_boucle[df_p_values_boucle['p_val_avg_dns_time'] < 0.01].index.unique()
    lignes_1fev = lignes_1fev[~lignes_1fev['boucle'].isin(boucles_defaillantes)] # on enlève les boucles defaillantes

    # p_values issues du test de Fisher pour les peag
    df_p_values_peag = nc.get_df_fisher_p_values(lignes_1fev, node_type = 'peag_nro', p_values = p_values_col)
    # scores d'isolation forest
    df_if_scores_peag = nc.get_if_scores_by_node(lignes_1fev_with_if_scores, node_type = 'peag_nro')
    
    peag_defaillants = df_p_values_peag[df_p_values_peag['p_val_avg_dns_time'] < 0.01].index.unique()
    lignes_1fev = lignes_1fev[~lignes_1fev['peag_nro'].isin(peag_defaillants)]  # on enlève les PEAG defaillants

    # p_values issues du test de Fisher pour les olt
    df_p_values_olt = nc.get_df_fisher_p_values(lignes_1fev, node_type = 'olt_name', p_values = p_values_col)
    # scores d'isolation forest
    df_if_scores_olt = nc.get_if_scores_by_node(lignes_1fev_with_if_scores, node_type = 'olt_name')
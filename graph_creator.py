import matplotlib.pyplot as plt
import seaborn as sns

class GraphCreator:
    """
    Classe permettant de générer des graphs pour l'analyse du réseau
    """
    
    def __init__(self):
        pass
    
    @staticmethod
    def plot_score_histogram(dict, name: str, variable: str):
        """
        Pour tracer l'histogramme d'un OLT_PEAG_boucle spécifique et d'une variable spécifique
        
        Parameters:
        -----------
        dict : dict 
            Dictionnaire des OLT
        name : str 
            Nom de l'OLT_PEAG_boucle pour lequel tracer l'histogramme
        variable : str 
            Nom de la variable : avg_dns_time, etc
        
        Returns:
        --------
        fig : matplotlib.figure.Figure
            Figure contenant l'histogramme
        """
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        sns.histplot(dict[variable][name], bins=100, kde=True, ax=ax)
        
        ax.set_xlabel('Test')
        ax.set_ylabel('Fréquence')
        ax.set_title(f'Distribution de {variable} pour {name}')
        
        ax.legend()
        plt.tight_layout()
        
        return fig
    
    @staticmethod
    def plot_anomalies_histogram(cycle_hp, cycle_hp_filtered, title: str):
        """
        Plot les distributions avec et sans anomalies
        
        Parameters:
        -----------
        cycle_hp : array-like
            Distribution avec anomalies
        cycle_hp_filtered : array-like
            Distribution sans anomalies (H0)
        title : str
            Titre du graphique
            
        Returns:
        --------
        fig : matplotlib.figure.Figure
            Figure contenant l'histogramme
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(cycle_hp, bins=300, kde=False, ax=ax, stat="density", label='Distribution avec anomalies')
        sns.histplot(cycle_hp_filtered, bins=80, kde=False, ax=ax, stat="density", label="Distribution sans anomalies (H0)")
        
        ax.set_xlim(-2, 3)
        ax.set_xticks(range(-2, 3))
        plt.title(title)
        ax.set_xlabel('avg_dns_time corrigé')
        ax.set_ylabel('Densité de probabilité')
        ax.legend()
        plt.tight_layout()
        
        return fig  # J'ai ajouté le return fig qui manquait dans la fonction originale
    
    @staticmethod
    def plot_series(data_dict, name: str, variable: str, title: str, y_value=None):
        """
        Pour tracer la série d'une OLT_PEAG_boucle spécifique et d'une variable spécifique.
        
        Parameters:
        -----------
        data_dict : dict
            Dictionnaire contenant les données.
        name : str
            Nom de l'OLT_PEAG_boucle pour lequel tracer l'histogramme.
        variable : str
            Nom de la variable à tracer (ex. : avg_dns_time, etc).
        title : str
            Titre du graphique.
        y_value : float, optionnel
            Valeur de y pour tracer une ligne horizontale.
            
        Returns:
        --------
        fig : matplotlib.figure.Figure
            Figure contenant la série temporelle
        """
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(data_dict[variable][name], label=name, color='#38214d')
        
        # ajout de la ligne horizontale si y_value est spécifié
        if y_value is not None:
            ax.axhline(y=y_value, color='red', linestyle='--', linewidth=2)
        
        ax.set_xlabel('t')
        ax.set_ylabel(variable)
        ax.set_title(title)
        ax.legend()
        plt.tight_layout()
        
        return fig
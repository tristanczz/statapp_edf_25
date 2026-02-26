import pandas as pd
import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF
import matplotlib.pyplot as plt
import seaborn as sns


def csv_to_pd(filepath, mois=None, var=None, years=None):
    """
    Charge un CSV à partir d'un chemin donné,
    filtre pour ne garder que les valeurs des variables dans var pour le mois indiqué
    et optionnellement pour une période d'années.

    Paramètres
    ----------
    filepath : str
        Chemin complet vers le fichier CSV (ex: '/home/onyxia/work/statapp_edf_25/data/obs.csv')
    mois : int
        Mois à sélectionner (1=Janvier, 12=Décembre)
    var : list
        Liste des noms des variables à conserver
    years : tuple, optional
        Tuple (année_début, année_fin) pour filtrer la période
        Ex: (1985, 2005) pour extraire les données de 1985 à 2005 inclus

    Retour
    ------
    pd.DataFrame
        DataFrame contenant uniquement les colonnes 'var' filtrées sur le mois et la période demandés.
    """
    if years is not None and (not isinstance(years, tuple) or len(years) != 2):
        raise ValueError(f"years doit être un tuple de 2 éléments (année_début, année_fin)")
    
    # Lire le CSV avec ; et parser la colonne date
    df = pd.read_csv(filepath, sep=';', parse_dates=['date'])

    # Filtrer sur le mois
    if mois is not None:
        df = df[df['date'].dt.month == mois]
    
    # Filtrer sur la période d'années si spécifié
    if years is not None:
        year_start, year_end = years
        df = df[(df['date'].dt.year >= year_start) & (df['date'].dt.year <= year_end)]
    
    # Vérifier qu'il reste des données
    if len(df) == 0:
        period_str = f" pour la période {years[0]}-{years[1]}" if years else ""
        raise ValueError(f"Aucune donnée trouvée pour le mois {mois} et la variable {var}{period_str}")

    # Ne garder que les variables dans var 
    if var is not None:
        df_final = df[var].copy()
    else:
        df_final = df.copy()

    return df_final


def cdf_t_univ(modele_hist, obs_hist, modele_futur):
    """
    Applique la méthode CDF-t pour corriger les biais d'un modèle climatique.

    Paramètres:
    -----------
    modele_hist : Numpy array
        Données historiques du modèle (une seule colonne)
    obs_hist : Numpy array
        Observations historiques (une seule colonne)
    modele_futur : Numpy array
        Projections futures du modèle (une seule colonne)

    Retourne:
    ---------
    modele_non_biaise : pd.DataFrame
        Données futures corrigées avec les dates associées
    ecdf_non_biaise : function
        Fonction de répartition empirique du modèle corrigé
    """

    # Fonctions de répartition empiriques 
    ecdf_modele_futur = ECDF(modele_futur) #F_X1
    ecdf_modele_hist = ECDF(modele_hist) # F_X0

    
    #Quantile mapping en supposant que la relation de stationnarité est vérifiée :
     
    q_futur = ecdf_modele_futur(modele_futur) # F_X1(X1)

    b = np.quantile(obs_hist, q_futur) #F_Y0^{-1}(F_X1(X1))

    c = ecdf_modele_hist(b) #F_X0(b)

    modele_non_biaise = np.quantile(modele_futur, c) #F_Y1^{-1}(c)

    # ECDF du modele non biaisé
    def ecdf_non_biaise(x):
        
        p = ecdf_modele_futur(x)
        b = np.quantile(modele_hist, p)
        return ECDF(obs_hist)(b)

    
    return modele_non_biaise, ecdf_non_biaise







def plot_distributions(data, titre, labels_tuple):
    """
    Trace les distributions de deux DataFrames en les superposant.
    Histogrammes et courbes de densité sont combinés sur le même graphique.

    Paramètres:
    -----------
    dfs_tuple : tuple(DataFrame, DataFrame)
        Tuple contenant deux DataFrames (futur, passé) avec une seule colonne chacun
    titre : str

    Retourne:
    ---------
    None (affiche le graphique)
    """
    # Configurer le style
    sns.set_style("whitegrid")
    
    # Extraire les deux DataFrames du tuple
    data1, data2 = data

    #Récupérer les labels
    label1, label2 = labels_tuple

    # Créer la figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Histogrammes superposés
    ax.hist(data1, bins=30, alpha=0.4, label=f' {label1} (histogramme)', 
            color='#e74c3c', density=True, edgecolor='black', linewidth=0.5)
    ax.hist(data2, bins=30, alpha=0.4, label=f'{label2} (histogramme)', 
            color='#3498db', density=True, edgecolor='black', linewidth=0.5)

    # Courbes de densité par-dessus
    sns.kdeplot(data=data1, label=f' {label1} (densité)', linewidth=2.5, 
                ax=ax, color='#c0392b', alpha=0.9)
    sns.kdeplot(data=data2, label=f'{label2} (densité)', linewidth=2.5, 
                ax=ax, color='#2980b9', alpha=0.9)

    # Labels et titre
    ax.set_xlabel('Valeur', fontsize=12)
    ax.set_ylabel('Densité', fontsize=12)
    ax.legend(fontsize=10, loc='best')
    ax.grid(alpha=0.3, axis='y')

    # Titre général
    ax.set_title(
        titre, 
        fontsize=14, fontweight='bold', pad=15
    )

    plt.tight_layout()
    plt.show()
    plt.close('all')


def comparer_distributions_univ(dis1, dis2):
    """
    Compare les statistiques descriptives de deux distributions.

    Paramètres:
    -----------
    dfs_tuple : Numpy arrays
        Tuple contenant deux DataFrames avec une seule colonne chacun

    Retourne:
    ---------
    DataFrame
        Tableau comparatif des statistiques
    """


    # Calculer les statistiques pour chaque colonne
    stats = {
        'Statistique': ['Moyenne', 'Médiane', 'Écart-type', 'Variance',
                        'Minimum', 'Maximum', 'Q1 (25%)', 'Q3 (75%)',
                        'Étendue'],
        'Distribution 1': [
            dis1.mean(),
            np.median(dis1),
            np.std(dis1),
            np.var(dis1),
            dis1.min(),
            dis1.max(),
            np.quantile(dis1, 0.25),
            np.quantile(dis1, 0.75),
            dis1.max() - dis1.min(),
        ],
        'Distribution 2': [
            dis2.mean(),
            np.median(dis2),
            np.std(dis2),
            np.var(dis2),
            dis2.min(),
            dis2.max(),
            np.quantile(dis2, 0.25),
            np.quantile(dis2, 0.75),
            dis2.max() - dis2.min(),
        ]
    }

    df_stats = pd.DataFrame(stats)
    df_stats = df_stats.round(3)

    return df_stats



"""
def plot_distributions_univ_norm(norm_model, norm_obs, nom_modele, mois, var):

    # Noms des mois
    mois_noms = {
        1: "Janvier", 2: "Février", 3: "Mars", 4: "Avril",
        5: "Mai", 6: "Juin", 7: "Juillet", 8: "Août",
        9: "Septembre", 10: "Octobre", 11: "Novembre", 12: "Décembre"
    }

    # Décomposer les paramètres
    mean_model, std_model = norm_model
    mean_obs, std_obs = norm_obs

    # Créer une plage de valeurs pour tracer les courbes
    x_min = min(mean_model - 4*std_model, mean_obs - 4*std_obs)
    x_max = max(mean_model + 4*std_model, mean_obs + 4*std_obs)
    x = np.linspace(x_min, x_max, 1000)

    # Calculer les densités
    y_model = norm.pdf(x, mean_model, std_model)
    y_obs = norm.pdf(x, mean_obs, std_obs)

    # Configurer le style
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 6))

    # Tracer les courbes
    plt.plot(x, y_model, linewidth=2.5, color='#e74c3c', alpha=0.8,
             label=f'Modèle: μ={mean_model:.2f}, σ={std_model:.2f}')
    plt.plot(x, y_obs, linewidth=2.5, color='#3498db', alpha=0.8,
             label=f'Corrigé: μ={mean_obs:.2f}, σ={std_obs:.2f}')

    # Ajouter des zones ombrées sous les courbes
    plt.fill_between(x, y_model, alpha=0.2, color='#e74c3c')
    plt.fill_between(x, y_obs, alpha=0.2, color='#3498db')

    # Labels et titre
    plt.xlabel('Valeur', fontsize=12)
    plt.ylabel('Densité de probabilité', fontsize=12)
    plt.title(f'Correction du modèle {nom_modele} pour {var} et {mois_noms[mois]}', 
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=11, loc='best')
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def extract_month_data(csv_path, month, columns=None, years=None):
    
    
    if not 1 <= month <= 12:
        raise ValueError(f"Le mois doit être entre 1 et 12, reçu: {month}")
    
    if years is not None and (not isinstance(years, tuple) or len(years) != 2):
        raise ValueError(f"years doit être un tuple de 2 éléments (année_début, année_fin)")
    
    df = pd.read_csv(csv_path, sep=';')
    date_col = df.columns[0]
    
    # Conversion flexible pour gérer différents formats de date
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Filtrer par mois
    df_filtered = df[df[date_col].dt.month == month]
    
    # Filtrer par période d'années si spécifié
    if years is not None:
        year_start, year_end = years
        df_filtered = df_filtered[
            (df_filtered[date_col].dt.year >= year_start) & 
            (df_filtered[date_col].dt.year <= year_end)
        ]
    
    if len(df_filtered) == 0:
        period_str = f" pour la période {years[0]}-{years[1]}" if years else ""
        raise ValueError(f"Aucune donnée trouvée pour le mois {month}{period_str}")
    
    # Si des colonnes spécifiques sont demandées
    if columns is not None:
        data = df_filtered[columns].values
    else:
        data = df_filtered.iloc[:, 1:].values
    
    period_info = f" ({years[0]}-{years[1]})" if years else ""
    print(f"Données extraites de {csv_path}: {len(data)} lignes, {data.shape[1]} variables pour le mois {month}{period_info}")
    
    return data"""
import pandas as pd
import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp


def csv_to_pd_univ(filepath, mois, var, years=None):
    """
    Charge un CSV à partir d'un chemin donné,
    filtre pour ne garder que les valeurs de la variable 'var' pour le mois indiqué
    et optionnellement pour une période d'années.

    Paramètres
    ----------
    filepath : str
        Chemin complet vers le fichier CSV (ex: '/home/onyxia/work/statapp_edf_25/data/obs.csv')
    mois : int
        Mois à sélectionner (1=Janvier, 12=Décembre)
    var : str
        Nom de la variable à conserver
    years : tuple, optional
        Tuple (année_début, année_fin) pour filtrer la période
        Ex: (1985, 2005) pour extraire les données de 1985 à 2005 inclus

    Retour
    ------
    pd.DataFrame
        DataFrame contenant uniquement la colonne 'var' filtrée sur le mois et la période demandés.
    """
    if years is not None and (not isinstance(years, tuple) or len(years) != 2):
        raise ValueError("years doit être un tuple de 2 éléments (année_début, année_fin)")

    # Lire le CSV avec ; et parser la colonne date
    df = pd.read_csv(filepath, sep=';', parse_dates=['date'])

    # Filtrer sur le mois
    df = df[df['date'].dt.month == mois]

    # Filtrer sur la période d'années si spécifié
    if years is not None:
        year_start, year_end = years
        df = df[(df['date'].dt.year >= year_start) & (df['date'].dt.year <= year_end)]

    # Vérifier qu'il reste des données
    if len(df) == 0:
        period_str = f" pour la période {years[0]}-{years[1]}" if years else ""
        raise ValueError(
            f"Aucune donnée pour le mois {mois} "
            f"et la variable {var}{period_str}"
        )

    # Ne garder que la colonne var
    df_final = df[[var]].copy()

    return df_final


def cdf_t_univ(modele_hist, obs_hist, modele_futur):
    """
    Applique la méthode CDF-t pour corriger les biais d'un modèle climatique.

    Paramètres:
    -----------
    modele_hist : DataFrame
        Données historiques du modèle (une seule colonne)
    obs_hist : DataFrame
        Observations historiques (une seule colonne)
    modele_futur : DataFrame
        Projections futures du modèle (une seule colonne)

    Retourne:
    ---------
    DataFrame
        DataFrame avec deux colonnes : 'valeurs_modele' et 'valeurs_corrigees'
    """
    # Extraire les valeurs des colonnes
    m_hist = modele_hist.iloc[:, 0].values
    o_hist = obs_hist.iloc[:, 0].values
    m_futur = modele_futur.iloc[:, 0].values

    # Calculer les fonctions de répartition empiriques (ECDF)
    ecdf_modele_futur = ECDF(m_futur)
    ecdf_modele_hist = ECDF(m_hist)

    # Étape 1:
    p_futur = ecdf_modele_futur(m_futur)

    # Étape 2:
    val_hist = np.quantile(o_hist, p_futur)

    # Étape 3:
    p_obs = ecdf_modele_hist(val_hist)

    # Étape 4:
    obs_futur_corrige = np.quantile(m_futur, p_obs)

    # Retourner un DataFrame avec deux colonnes
    return (
        pd.DataFrame({'valeurs_modele': m_futur}, index=modele_futur.index),
        pd.DataFrame({'valeurs_corrigees': obs_futur_corrige}, index=modele_futur.index)
    )


def plot_distributions(dfs_tuple, titre, labels_tuple):
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
    df1, df2 = dfs_tuple

    # Récupérer les données
    col1, col2 = df1.columns[0], df2.columns[0]
    data1 = df1[col1]
    data2 = df2[col2]

    # Récupérer les labels
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


def comparer_distributions(df1, col1_label, df2, col2_label):
    """
    Compare deux distributions à partir de deux DataFrames à une seule colonne.
    Utilise col1_label et col2_label comme étiquettes dans le tableau de résultat.

    Paramètres
    ----------
    df1, df2 : pandas.DataFrame
        DataFrames contenant chacun une seule colonne
    col1_label, col2_label : str
        Labels pour les colonnes dans le tableau de sortie

    Retour
    ------
    result_df : pandas.DataFrame
        Tableau avec espérances et variances par "label"
    summary_df : pandas.DataFrame
        Tableau avec différence d'espérances, rapport des variances et test KS
    """

    # Récupérer la première colonne de chaque DataFrame
    x = df1.iloc[:, 0].dropna()
    y = df2.iloc[:, 0].dropna()

    # Espérances
    mean_x = x.mean()
    mean_y = y.mean()
    mean_diff = mean_y - mean_x

    # Variances
    var_x = x.var(ddof=1)
    var_y = y.var(ddof=1)
    var_ratio = var_y / var_x if var_x != 0 else np.nan

    # Test de Kolmogorov–Smirnov
    ks_stat, ks_pvalue = ks_2samp(x, y)

    # Tableau résultat par label
    result_df = pd.DataFrame({
        "Distribution": [col1_label, col2_label],
        "Espérance": [mean_x, mean_y],
        "Variance": [var_x, var_y]
    })

    # Tableau récapitulatif
    summary_df = pd.DataFrame({
        "Différence des espérances": [mean_diff],
        "Rapport des variances": [var_ratio],
        "KS statistique": [ks_stat],
        "KS p-value": [ks_pvalue]
    })

    return result_df, summary_df


def extract_month_data(csv_path, month, columns=None, years=None):
    """
    Extrait les données d'un mois spécifique d'un fichier CSV pour une période donnée.

    Parameters:
    -----------
    csv_path : str
        Chemin vers le fichier CSV (séparateur ';')
    month : int
        Numéro du mois à extraire (1-12)
    columns : list, optional
        Liste des colonnes à extraire (dans l'ordre souhaité)
    years : tuple, optional
        Tuple (année_début, année_fin) pour filtrer la période
        Ex: (1985, 2005) pour extraire les données de 1985 à 2005 inclus

    Returns:
    --------
    numpy.ndarray
        Array de shape (n_samples, n_variables) avec les données du mois et de la période
    """
    if not 1 <= month <= 12:
        raise ValueError(f"Le mois doit être entre 1 et 12, reçu: {month}")

    if years is not None and (not isinstance(years, tuple) or len(years) != 2):
        raise ValueError("years doit être un tuple de 2 éléments (année_début, année_fin)")

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
    print(
        f"Données extraites de {csv_path}: "
        f"{len(data)} lignes, {data.shape[1]} variables "
        f"pour le mois {month}{period_info}"
    )

    return data

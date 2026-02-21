import numpy as np
import pandas as pd
import ot
from sklearn.preprocessing import StandardScaler


# ==============================================================================
# FONCTIONS UTILITAIRES
# ==============================================================================

def build_common_grid(X, Y, bin_size):
    """
    Construit les bords (edges) d'une grille reguliere commune a X et Y.

    On ne calcule PAS les centres ici : le produit cartesien des centres
    (np.meshgrid) explose en memoire pour des donnees multivariees avec
    beaucoup de bins. Les centres sont calcules a la demande uniquement
    pour les cellules non vides, via get_centers_nz().

    Retourne
    --------
    edges : list de d arrays - bords des bins dans chaque dimension
    """
    d = X.shape[1]
    mins = np.minimum(X.min(axis=0), Y.min(axis=0)) - bin_size
    maxs = np.maximum(X.max(axis=0), Y.max(axis=0)) + bin_size
    edges = [np.arange(mins[i], maxs[i] + bin_size, bin_size) for i in range(d)]
    return edges


def get_centers_nz(idx_nz, edges):
    """
    Calcule les centres uniquement pour les cellules non vides (idx_nz).

    Evite de materialiser la grille complete. Convertit les indices 1D
    aplatis en indices multi-dimensionnels, puis calcule le centre de
    chaque cellule dans chaque dimension.

    Parametres
    ----------
    idx_nz : array (n_nz,) - indices 1D des cellules non vides
    edges  : list de d arrays - bords des bins

    Retourne
    --------
    centers_nz : array (n_nz, d) - centres des cellules non vides
    """
    d = len(edges)
    n_bins = np.array([len(e) - 1 for e in edges])

    # Conversion index 1D -> indices multi-dim (ordre C)
    multi_idx = np.array(np.unravel_index(idx_nz, n_bins)).T  # (n_nz, d)

    # Centre de chaque cellule = milieu de son bin dans chaque dimension
    centers_nz = np.zeros((len(idx_nz), d))
    for dim in range(d):
        bin_idx = multi_idx[:, dim]
        centers_nz[:, dim] = (edges[dim][bin_idx] + edges[dim][bin_idx + 1]) / 2

    return centers_nz


def assign_cells(points, edges):
    """
    Assigne chaque point a son index de cellule dans la grille (O(d) par point).

    Utilise searchsorted sur les edges : beaucoup plus rapide que chercher
    le plus proche voisin parmi tous les centres (qui serait O(n_cells)).

    Retourne
    --------
    flat_indices : array (n,) - index 1D en ordre C, -1 si hors grille
    """
    n, d = points.shape
    n_bins = np.array([len(e) - 1 for e in edges])

    bin_indices = np.zeros((n, d), dtype=int)
    for dim in range(d):
        bin_indices[:, dim] = np.searchsorted(edges[dim], points[:, dim], side='right') - 1

    in_grid = np.all((bin_indices >= 0) & (bin_indices < n_bins), axis=1)

    flat_indices = np.full(n, -1, dtype=int)
    if in_grid.any():
        flat_indices[in_grid] = np.ravel_multi_index(
            bin_indices[in_grid].T,
            n_bins
        )

    return flat_indices


def compute_OT_plan(X, Y, bin_size):
    """
    Calcule le plan de transport optimal discret entre X et Y.

    On ne materialise jamais la grille complete ni gamma complet.
    On travaille uniquement sur les cellules non vides.

    Retourne
    --------
    gamma_nz  : array (n_nz_X, n_nz_Y) - plan reduit aux cellules non vides
    edges     : list de d arrays         - bords de la grille commune
    centers_X : array (n_nz_X, d)        - centres des cellules non vides de X
    centers_Y : array (n_nz_Y, d)        - centres des cellules non vides de Y
    p_X_full  : array (n_cells,)          - poids de X sur grille complete
    idx_X     : array (n_nz_X,)           - indices complets des cellules non vides de X
    idx_Y     : array (n_nz_Y,)           - indices complets des cellules non vides de Y
    """
    edges = build_common_grid(X, Y, bin_size)

    H_X, _ = np.histogramdd(X, bins=edges)
    H_Y, _ = np.histogramdd(Y, bins=edges)

    p_X = H_X.ravel() / X.shape[0]
    p_Y = H_Y.ravel() / Y.shape[0]

    mask_X = p_X > 1e-10
    mask_Y = p_Y > 1e-10

    idx_X = np.where(mask_X)[0]
    idx_Y = np.where(mask_Y)[0]

    p_X_nz = p_X[mask_X] / p_X[mask_X].sum()
    p_Y_nz = p_Y[mask_Y] / p_Y[mask_Y].sum()

    # Centres uniquement pour les cellules non vides (pas de meshgrid global)
    centers_X = get_centers_nz(idx_X, edges)   # (n_nz_X, d)
    centers_Y = get_centers_nz(idx_Y, edges)   # (n_nz_Y, d)

    C = ot.dist(centers_X, centers_Y, metric='sqeuclidean')
    gamma_nz = ot.emd(p_X_nz, p_Y_nz, C, numItermax=1_000_000)     # (n_nz_X, n_nz_Y)

    return gamma_nz, edges, centers_X, centers_Y, p_X, idx_X, idx_Y


def compute_rescaling_matrix(X0, Y0, method='diagonal'):
    """
    Calcule la matrice de rescaling D (equation 6 de Robin et al. 2019).

    'diagonal' : D = diag(sigma_{Y0} / sigma_{X0})
    'cholesky' : D = Cho(Sigma_{Y0}) . Cho(Sigma_{X0})^{-1}
    """
    S_X0 = np.cov(X0.T)
    S_Y0 = np.cov(Y0.T)

    if method == 'diagonal':
        s_X0 = np.sqrt(np.diag(S_X0))
        s_Y0 = np.sqrt(np.diag(S_Y0))
        D = np.diag(s_Y0 / (s_X0 + 1e-10))

    elif method == 'cholesky':
        eps = 1e-6
        try:
            L_X0 = np.linalg.cholesky(S_X0 + eps * np.eye(S_X0.shape[0]))
            L_Y0 = np.linalg.cholesky(S_Y0 + eps * np.eye(S_Y0.shape[0]))
            D = L_Y0 @ np.linalg.inv(L_X0)
        except np.linalg.LinAlgError:
            print("Warning : Cholesky echoue, bascule sur 'diagonal'.")
            D = compute_rescaling_matrix(X0, Y0, method='diagonal')

    else:
        raise ValueError(f"Methode de rescaling inconnue : '{method}'")

    return D


# ==============================================================================
# ALGORITHME OTC (cas stationnaire) - Algorithm 1 du papier
# ==============================================================================

def OTC(X, Y, bin_size):
    """
    Optimal Transport Correction - correction stationnaire.

    Pour chaque point X_l :
      1. Trouver sa cellule i via assign_cells (O(d)).
      2. Retrouver sa position dans gamma_nz via searchsorted sur idx_X.
      3. Tirer une cellule destination j selon la loi conditionnelle.
      4. Tirer uniformement un point dans la cellule j.

    Retourne
    --------
    Z : array (n, d)
    """
    d = X.shape[1]

    gamma_nz, edges, centers_X, centers_Y, p_X, idx_X, idx_Y = \
        compute_OT_plan(X, Y, bin_size)

    cell_indices = assign_cells(X, edges)

    Z = np.zeros_like(X)

    for l in range(len(X)):
        i = cell_indices[l]
        pos = np.searchsorted(idx_X, i)

        if i >= 0 and pos < len(idx_X) and idx_X[pos] == i:
            probs = gamma_nz[pos, :]
            total = probs.sum()

            if total < 1e-10:
                Z[l] = X[l]
                continue

            probs = probs / total

            j_nz = np.random.choice(len(idx_Y), p=probs)
            Z[l] = centers_Y[j_nz] + np.random.uniform(-bin_size / 2, bin_size / 2, size=d)

        else:
            Z[l] = X[l]

    return Z


# ==============================================================================
# ALGORITHME dOTC (cas non-stationnaire) - Algorithm 2 du papier
# avec modification : nearest neighbor au lieu du plan gamma(Y0 -> X0)
# ==============================================================================

def dOTC_simplified(X0, X1, Y0, bin_size=0.5, rescaling='diagonal'):
    """
    Dynamical Optimal Transport Correction (version simplifiee).

    Implemente l'algorithme dOTC de Robin et al. (2019) avec la modification :
    nearest neighbor pour associer chaque y0 a une cellule de X0.

    Note : attend des donnees deja normalisees. Utiliser dOTC_df() qui
    gere la normalisation et la denormalisation automatiquement.

    Etapes
    ------
    1. Calculer phi = plan OT entre X0 et X1
    2. Extraire les vecteurs v_ik et leurs poids conditionnels
    3. Calculer D = matrice de rescaling
    4. Pour chaque y0 : nearest neighbor dans X0, cellule i,
       v_bar_i = somme_k w_ik * v_ik, Y1 = y0 + D @ v_bar_i
    5. Z1 = OTC(X1, Y1_estimated)

    Retourne
    --------
    Y1_estimated : array (n_y, d)
    Z1           : array (n_x1, d)
    """
    n, d = X0.shape
    n_y = Y0.shape[0]

    # =========================================================================
    # ETAPE 1 : Plan phi entre X0 et X1
    # =========================================================================
    gamma_nz, edges_phi, centers_X0_nz, centers_X1_nz, p_X0_phi, idx_X0, idx_X1 = \
        compute_OT_plan(X0, X1, bin_size)

    # =========================================================================
    # ETAPE 2 : Vecteurs d'evolution v_ik
    #
    # v_dict[a] = liste de (v_ik, w_ik) pour la cellule a (position dans
    # gamma_nz). On indexe par 'a' (position reduite) et non par 'i'
    # (index complet), car on retrouvera 'a' par searchsorted a l'etape 4.
    # =========================================================================
    v_dict = {}

    for a in range(len(idx_X0)):
        bs = np.where(gamma_nz[a, :] > 1e-10)[0]

        if len(bs) == 0:
            continue

        transports = []
        for b in bs:
            v_ik = centers_X1_nz[b] - centers_X0_nz[a]       # vecteur de deplacement
            w_ik = gamma_nz[a, b] / p_X0_phi[idx_X0[a]]       # poids conditionnel
            transports.append((v_ik, w_ik))

        v_dict[a] = transports

    # =========================================================================
    # ETAPE 3 : Matrice de rescaling D
    # =========================================================================
    D = compute_rescaling_matrix(X0, Y0, method=rescaling)

    # =========================================================================
    # ETAPE 4 : Estimation de Y1
    #
    # a) Nearest neighbor par blocs (evite un tableau n_y x n en memoire).
    # b) Assigner les x0_nn a leur cellule via assign_cells (O(d)).
    # c) Retrouver la position 'a' dans gamma_nz par searchsorted.
    # d) Calculer le deplacement moyen et deplacer y0.
    # =========================================================================
    Y1_estimated = np.zeros((n_y, d))

    # a) Nearest neighbor par blocs
    block_size = 500
    nn_indices = np.zeros(n_y, dtype=int)

    for start in range(0, n_y, block_size):
        end = min(start + block_size, n_y)
        block = Y0[start:end]
        dists = np.linalg.norm(
            block[:, np.newaxis, :] - X0[np.newaxis, :, :],
            axis=2
        )
        nn_indices[start:end] = np.argmin(dists, axis=1)

    # b) Cellules des x0 nearest neighbors
    x0_nns = X0[nn_indices]
    cell_indices_nn = assign_cells(x0_nns, edges_phi)

    # c) et d) Deplacement
    for l in range(n_y):
        i = cell_indices_nn[l]
        pos = np.searchsorted(idx_X0, i)

        if i >= 0 and pos < len(idx_X0) and idx_X0[pos] == i and pos in v_dict:
            v_mean = np.zeros(d)
            for v_ik, w_ik in v_dict[pos]:
                v_mean += w_ik * v_ik

            Y1_estimated[l] = Y0[l] + D @ v_mean

        else:
            Y1_estimated[l] = Y0[l]

    # =========================================================================
    # ETAPE 5 : Correction de X1 via OTC(X1, Y1_estimated)
    # =========================================================================
    Z1 = OTC(X1, Y1_estimated, bin_size)

    return Y1_estimated, Z1


# ==============================================================================
# WRAPPER DATAFRAME AVEC NORMALISATION
# ==============================================================================

def dOTC_df(X0, X1, Y0, bin_size=0.5, rescaling='diagonal'):
    """
    Wrapper de dOTC_simplified avec normalisation z-score automatique.

    Recommande pour des donnees climatiques multivariees aux echelles
    tres differentes (ex: temperature ~[-20,40], radiation ~[0,500]).

    La normalisation est apprise sur X0 (periode de calibration du modele)
    et appliquee a X0, X1 et Y0 avant dOTC. Les sorties sont
    denormalisees avec les memes parametres avant d'etre retournees.

    Pourquoi normaliser sur X0 ?
    - X0 est la reference du modele en calibration.
    - Appliquer la meme transformation a X1 et Y0 preserve les ecarts
      relatifs entre periodes, ce que dOTC cherche a capturer.

    Parametres
    ----------
    X0, X1, Y0 : DataFrames pandas
    bin_size    : float (defaut=0.5, adapte aux donnees normalisees ~[-3,3])
    rescaling   : 'diagonal' ou 'cholesky'

    Retourne
    --------
    Y1_estimated : DataFrame - en unites originales (denormalisees)
    Z1           : DataFrame - en unites originales (denormalisees)
    """
    cols = X0.columns

    # Apprentissage du scaler sur X0 uniquement
    scaler = StandardScaler().fit(X0.values)

    # Normalisation des trois jeux de donnees
    X0_n = scaler.transform(X0.values)
    X1_n = scaler.transform(X1.values)
    Y0_n = scaler.transform(Y0.values)

    # dOTC dans l'espace normalise
    Y1_est_n, Z1_n = dOTC_simplified(X0_n, X1_n, Y0_n, bin_size, rescaling)

    # Denormalisation : retour aux unites originales
    Y1_est = scaler.inverse_transform(Y1_est_n)
    Z1     = scaler.inverse_transform(Z1_n)

    return pd.DataFrame(Y1_est, columns=cols), pd.DataFrame(Z1, columns=cols)


# ==============================================================================
# UTILITAIRE DE DIAGNOSTIC
# ==============================================================================

def grid_size_estimate(X, Y, bin_size):
    """
    Affiche le nombre de bins par dimension et le nombre total de cellules
    pour un bin_size donne. Utile pour choisir bin_size avant de lancer dOTC.
    """
    mins = np.minimum(X.min(axis=0), Y.min(axis=0)) - bin_size
    maxs = np.maximum(X.max(axis=0), Y.max(axis=0)) + bin_size
    n_bins = np.ceil((maxs - mins) / bin_size).astype(int)
    print(f"Bins par dimension : {n_bins.tolist()}")
    print(f"Nombre total de cellules : {np.prod(n_bins):,.0f}")
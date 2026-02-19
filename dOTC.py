import numpy as np
import pandas as pd
import ot


# ==============================================================================
# FONCTIONS UTILITAIRES
# ==============================================================================

def build_common_grid(X, Y, bin_size):
    """
    Construit une grille régulière commune couvrant à la fois X et Y.

    Retourne
    --------
    edges   : list de d arrays - bords des bins dans chaque dimension
    centers : array (n_cells, d) - centre de chaque cellule, en ordre C
              (cohérent avec np.histogramdd et np.ravel_multi_index)
    """
    d = X.shape[1]

    mins = np.minimum(X.min(axis=0), Y.min(axis=0)) - bin_size
    maxs = np.maximum(X.max(axis=0), Y.max(axis=0)) + bin_size

    edges = [np.arange(mins[i], maxs[i] + bin_size, bin_size) for i in range(d)]

    dim_centers = [(e[:-1] + e[1:]) / 2 for e in edges]

    # indexing='ij' : cohérent avec histogramdd (première dim varie en premier)
    grids = np.meshgrid(*dim_centers, indexing='ij')
    centers = np.column_stack([g.ravel() for g in grids])   # (n_cells, d)

    return edges, centers


def assign_cells(points, edges):
    """
    Assigne chaque point à son index de cellule dans la grille définie par edges.

    Cette fonction est O(d) par point (calcul direct par searchsorted),
    contrairement à l'ancienne find_cell_index qui était O(n_cells) par point
    et rendait le code infiniment lent sur de vraies données.

    L'index retourné est un index 1D aplati en ordre C, cohérent avec
    histogramdd et build_common_grid (indexing='ij').

    Paramètres
    ----------
    points : array (n, d)
    edges  : list de d arrays - bords des bins (retournés par build_common_grid)

    Retourne
    --------
    flat_indices : array (n,) - index 1D dans la grille pour chaque point,
                  -1 si le point est hors grille
    """
    n, d = points.shape
    n_bins = np.array([len(e) - 1 for e in edges])   # nombre de cellules par dimension

    # Pour chaque dimension, trouver le bin de chaque point par searchsorted
    # searchsorted(..., side='right') - 1 donne le bin contenant x
    bin_indices = np.zeros((n, d), dtype=int)
    for dim in range(d):
        bin_indices[:, dim] = np.searchsorted(edges[dim], points[:, dim], side='right') - 1

    # Masque des points dans la grille (bin valide dans toutes les dimensions)
    in_grid = np.all((bin_indices >= 0) & (bin_indices < n_bins), axis=1)

    # Conversion multi-index -> index 1D en ordre C
    flat_indices = np.full(n, -1, dtype=int)
    if in_grid.any():
        flat_indices[in_grid] = np.ravel_multi_index(
            bin_indices[in_grid].T,   # shape (d, n_valid)
            n_bins
        )

    return flat_indices


def compute_OT_plan(X, Y, bin_size):
    """
    Calcule le plan de transport optimal discret entre X et Y.

    On ne stocke JAMAIS gamma sur la grille complete (qui peut avoir des
    millions de cellules -> MemoryError). On travaille uniquement sur les
    cellules non vides et on retourne les indices pour faire le lien.

    La coherence est garantie par idx_X et idx_Y :
      gamma_nz[a, b]  <->  cellules centers[idx_X[a]] et centers[idx_Y[b]]

    Retourne
    --------
    gamma_nz : array (n_nz_X, n_nz_Y) - plan reduit aux cellules non vides
    edges    : list de d arrays         - bords de la grille commune
    centers  : array (n_cells, d)       - centres de toutes les cellules
    p_X      : array (n_cells,)         - poids de X sur grille complete
    p_Y      : array (n_cells,)         - poids de Y sur grille complete
    idx_X    : array (n_nz_X,)          - indices complets des cellules non vides de X
    idx_Y    : array (n_nz_Y,)          - indices complets des cellules non vides de Y
    """
    edges, centers = build_common_grid(X, Y, bin_size)

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

    # Cout quadratique uniquement entre cellules non vides
    C = ot.dist(centers[idx_X], centers[idx_Y], metric='sqeuclidean')

    gamma_nz = ot.emd(p_X_nz, p_Y_nz, C)   # shape (n_nz_X, n_nz_Y)

    return gamma_nz, edges, centers, p_X, p_Y, idx_X, idx_Y


def compute_rescaling_matrix(X0, Y0, method='diagonal'):
    """
    Calcule la matrice de rescaling D (equation 6 de Robin et al. 2019).

    D adapte l'amplitude des vecteurs d'evolution v_ik (calibres sur X0->X1)
    a l'espace des observations Y0.

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
      1. Trouver la cellule i via assign_cells (O(d), rapide).
      2. Retrouver la position de i dans idx_X via searchsorted.
      3. Tirer aleatoirement une cellule destination j selon la loi
         conditionnelle gamma_nz[pos, :].
      4. Tirer uniformement un point dans la cellule j -> Z_l.

    Retourne
    --------
    Z : array (n, d) - X corrige dont la distribution approche celle de Y
    """
    d = X.shape[1]

    gamma_nz, edges, centers, p_X, p_Y, idx_X, idx_Y = compute_OT_plan(X, Y, bin_size)

    # Assigner chaque point de X a sa cellule (O(d) par point)
    cell_indices = assign_cells(X, edges)   # shape (n,), -1 si hors grille

    Z = np.zeros_like(X)

    for l in range(len(X)):
        i = cell_indices[l]   # index complet de la cellule de X[l]

        # Position de i dans idx_X (idx_X est trie par np.where)
        pos = np.searchsorted(idx_X, i)

        if i >= 0 and pos < len(idx_X) and idx_X[pos] == i:
            probs = gamma_nz[pos, :]       # loi conditionnelle sur n_nz_Y cellules
            total = probs.sum()

            if total < 1e-10:
                Z[l] = X[l]
                continue

            probs = probs / total          # renormalisation de securite

            # Tirage de la cellule destination parmi les cellules non vides de Y
            j_nz = np.random.choice(len(idx_Y), p=probs)
            j = idx_Y[j_nz]               # index complet dans centers

            # Point uniforme dans la cellule j
            Z[l] = centers[j] + np.random.uniform(-bin_size / 2, bin_size / 2, size=d)

        else:
            # Hors grille ou cellule vide : conserver le point
            Z[l] = X[l]

    return Z


# ==============================================================================
# ALGORITHME dOTC (cas non-stationnaire) - Algorithm 2 du papier
# avec modification : nearest neighbor au lieu du plan gamma(Y0 -> X0)
# ==============================================================================

def dOTC_simplified(X0, X1, Y0, bin_size=0.2, rescaling='diagonal'):
    """
    Dynamical Optimal Transport Correction (version simplifiee).

    Implemente l'algorithme dOTC de Robin et al. (2019) avec la modification :
    nearest neighbor pour associer chaque y0 a une cellule de X0
    (au lieu du plan de transport gamma entre Y0 et X0).

    Schema (Figure 2 du papier) :

        X0  --phi-->  X1
        |                  | OTC
        NN                 v
        v
        Y0  --phi_tilde-->  Y1 (estime)

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
    n_y = Y0.shape[0]   # peut differer de n

    # =========================================================================
    # ETAPE 1 : Plan phi entre X0 et X1
    # =========================================================================
    gamma_nz, edges_phi, centers_phi, p_X0_phi, p_X1_phi, idx_X0, idx_X1 = \
        compute_OT_plan(X0, X1, bin_size)

    # =========================================================================
    # ETAPE 2 : Vecteurs d'evolution v_ik
    #
    # v_dict[i] = liste de (v_ik, w_ik) pour toutes les cellules k de X1
    # vers lesquelles la cellule i de X0 transporte.
    # Cle = index COMPLET i (dans centers_phi), coherent avec assign_cells.
    # =========================================================================
    v_dict = {}

    for a, i in enumerate(idx_X0):
        bs = np.where(gamma_nz[a, :] > 1e-10)[0]   # cellules X1 non nulles

        if len(bs) == 0:
            continue

        transports = []
        for b in bs:
            k = idx_X1[b]
            v_ik = centers_phi[k] - centers_phi[i]       # vecteur de deplacement
            w_ik = gamma_nz[a, b] / p_X0_phi[i]          # poids conditionnel
            transports.append((v_ik, w_ik))

        v_dict[i] = transports

    # =========================================================================
    # ETAPE 3 : Matrice de rescaling D
    # =========================================================================
    D = compute_rescaling_matrix(X0, Y0, method=rescaling)

    # =========================================================================
    # ETAPE 4 : Estimation de Y1
    #
    # a) Nearest neighbor par blocs : pour chaque y0, trouver le x0 le plus
    #    proche. Calcul par blocs pour eviter un tableau (n_y x n) en memoire.
    #
    # b) Assigner les x0_nn a leur cellule via assign_cells (O(d), rapide).
    #
    # c) Calculer le deplacement moyen pondere et deplacer y0.
    # =========================================================================
    Y1_estimated = np.zeros((n_y, d))

    # a) Nearest neighbor par blocs
    block_size = 500
    nn_indices = np.zeros(n_y, dtype=int)

    for start in range(0, n_y, block_size):
        end = min(start + block_size, n_y)
        block = Y0[start:end]                             # (taille_bloc, d)
        dists = np.linalg.norm(
            block[:, np.newaxis, :] - X0[np.newaxis, :, :],
            axis=2
        )                                                 # (taille_bloc, n)
        nn_indices[start:end] = np.argmin(dists, axis=1)

    # b) Cellules des x0 nearest neighbors (O(d) par point)
    x0_nns = X0[nn_indices]                               # (n_y, d)
    cell_indices_nn = assign_cells(x0_nns, edges_phi)     # (n_y,)

    # c) Deplacement
    for l in range(n_y):
        i = cell_indices_nn[l]   # index complet de la cellule du x0 le plus proche

        if i >= 0 and i in v_dict:
            # Deplacement moyen pondere = esperance de v sous loi conditionnelle phi_i
            v_mean = np.zeros(d)
            for v_ik, w_ik in v_dict[i]:
                v_mean += w_ik * v_ik

            Y1_estimated[l] = Y0[l] + D @ v_mean

        else:
            # Cellule hors grille ou sans transport : pas de deplacement
            Y1_estimated[l] = Y0[l]

    # =========================================================================
    # ETAPE 5 : Correction de X1 via OTC(X1, Y1_estimated)
    # =========================================================================
    Z1 = OTC(X1, Y1_estimated, bin_size)

    return Y1_estimated, Z1


# ==============================================================================
# WRAPPER DATAFRAME
# ==============================================================================

def dOTC_df(X0, X1, Y0, bin_size=0.2, rescaling='diagonal'):
    """
    Wrapper de dOTC_simplified acceptant et retournant des DataFrames pandas.

    Les noms de colonnes de X0 sont conserves dans les sorties.

    Parametres
    ----------
    X0, X1, Y0 : DataFrames pandas
    bin_size    : float
    rescaling   : 'diagonal' ou 'cholesky'

    Retourne
    --------
    Y1_estimated : DataFrame
    Z1           : DataFrame
    """
    cols = X0.columns
    Y1_est, Z1 = dOTC_simplified(
        X0.values, X1.values, Y0.values, bin_size, rescaling
    )
    return pd.DataFrame(Y1_est, columns=cols), pd.DataFrame(Z1, columns=cols)

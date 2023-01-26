import numpy as np
import math
from scipy.spatial.distance import cdist

def eucl_dist(x, y):
    """
    Usage
    -----
    L2-norm between point x and y
    Parameters
    ----------
    param x : numpy_array
    param y : numpy_array
    Returns
    -------
    dist : float
           L2-norm between x and y
    """
    dist = np.linalg.norm(x - y)
    return dist

def eucl_dist_traj(t1, t2):
    """
    Usage
    -----
    Pairwise L2-norm between point of trajectories t1 and t2
    Parameters
    ----------
    param t1 : len(t1)x2 numpy_array
    param t2 : len(t1)x2 numpy_array
    Returns
    -------
    dist : float
           L2-norm between x and y
    """
    mdist = cdist(t1, t2, 'euclidean')
    print('\nmdist = ', mdist)
    return mdist

def point_to_seg(p, s1, s2, dps1, dps2, ds):
    """
    Usage
    -----
    Point to segment distance between point p and segment delimited by s1 and s2
    Parameters
    ----------
    param p : 1x2 numpy_array
    param s1 : 1x2 numpy_array
    param s2 : 1x2 numpy_array
    dps1 : euclidean distance between p and s1
    dps2 : euclidean distance between p and s2
    dps : euclidean distance between s1 and s2
    Returns
    -------
    dpl: float
         Point to segment distance between p and s
    """
    px = p[0]
    py = p[1]
    p1x = s1[0]
    p1y = s1[1]
    p2x = s2[0]
    p2y = s2[1]
    if p1x == p2x and p1y == p2y:
        dpl = dps1
    else:
        segl = ds
        x_diff = p2x - p1x
        y_diff = p2y - p1y
        u1 = (((px - p1x) * x_diff) + ((py - p1y) * y_diff))
        u = u1 / (segl * segl)

        if (u < 0.00001) or (u > 1):
            # closest point does not fall within the line segment, take the shorter distance to an endpoint
            dpl = min(dps1, dps2)
        else:
            # Intersecting point is on the line, use the formula
            ix = p1x + u * x_diff
            iy = p1y + u * y_diff
            dpl = eucl_dist(p, np.array([ix, iy]))

    return dpl

def point_to_trajectory(p, t, mdist_p, t_dist, l_t):
    """
    Usage
    -----
    Point-to-trajectory distance between point p and the trajectory t.
    The Point to trajectory distance is the minimum of point-to-segment distance between p and all segment s of t
    Parameters
    ----------
    param p: 1x2 numpy_array
    param t : l_tx2 numpy_array
    param mdist_p : l_t x 1 numpy array, distances from point p to points of trajectory t
    param t_dist : l_t x 1 numpy array, distances from consecutives points in trajectory t
    param l_t: int lenght of t
    Returns
    -------
    dpt : float,
          Point-to-trajectory distance between p and trajectory t
    """
    dpt = min(
        [point_to_seg(p, t[it], t[it + 1], mdist_p[it], mdist_p[it + 1], t_dist[it]) for it in range(l_t - 1)])
    return dpt

def discret_frechet(t0, t1):
    """
    Usage
    -----
    Compute the discret frechet distance between trajectories P and Q
    Parameters
    ----------
    param t0 : px2 numpy_array, Trajectory t0
    param t1 : qx2 numpy_array, Trajectory t1
    Returns
    -------
    frech : float, the discret frechet distance between trajectories t0 and t1
    """
    n0 = len(t0)
    n1 = len(t1)
    C = np.zeros((n0 + 1, n1 + 1))
    C[1:, 0] = float('inf')
    C[0, 1:] = float('inf')
    for i in np.arange(n0) + 1:
        for j in np.arange(n1) + 1:
            C[i, j] = max(eucl_dist(t0[i - 1], t1[j - 1]), min(C[i, j - 1], C[i - 1, j - 1], C[i - 1, j]))
    dtw = C[n0, n1]
    return dtw

def e_dtw(t0, t1):
    """
    Usage
    -----
    The Dynamic-Time Warping distance between trajectory t0 and t1.
    Parameters
    ----------
    param t0 : len(t0)x2 numpy_array
    param t1 : len(t1)x2 numpy_array
    Returns
    -------
    dtw : float
          The Dynamic-Time Warping distance between trajectory t0 and t1
    """

    n0 = len(t0)
    n1 = len(t1)
    C = np.zeros((n0 + 1, n1 + 1))
    C[1:, 0] = float('inf')
    C[0, 1:] = float('inf')
    for i in np.arange(n0) + 1:
        for j in np.arange(n1) + 1:
            C[i, j] = eucl_dist(t0[i - 1], t1[j - 1]) + min(C[i, j - 1], C[i - 1, j - 1], C[i - 1, j])
    dtw = C[n0, n1]
    return dtw

def e_edr(t0, t1, eps):
    """
    Usage
    -----
    The Edit Distance on Real sequence between trajectory t0 and t1.
    Parameters
    ----------
    param t0 : len(t0)x2 numpy_array
    param t1 : len(t1)x2 numpy_array
    eps : float
    Returns
    -------
    edr : float
           The Longuest-Common-Subsequence distance between trajectory t0 and t1
    """
    n0 = len(t0)
    n1 = len(t1)
    # An (m+1) times (n+1) matrix
    C = [[0] * (n1 + 1) for _ in range(n0 + 1)]
    for i in range(1, n0 + 1):
        for j in range(1, n1 + 1):
            if eucl_dist(t0[i - 1], t1[j - 1]) < eps:
                subcost = 0
            else:
                subcost = 1
            C[i][j] = min(C[i][j - 1] + 1, C[i - 1][j] + 1, C[i - 1][j - 1] + subcost)
    edr = float(C[n0][n1]) / max([n0, n1])
    return edr

def e_erp(t0, t1, g):
    """
    Usage
    -----
    The Edit distance with Real Penalty between trajectory t0 and t1.
    Parameters
    ----------
    param t0 : len(t0)x2 numpy_array
    param t1 : len(t1)x2 numpy_array
    Returns
    -------
    dtw : float
          The Dynamic-Time Warping distance between trajectory t0 and t1
    """

    n0 = len(t0)
    n1 = len(t1)
    C = np.zeros((n0 + 1, n1 + 1))

    gt0_dist = [abs(eucl_dist(g, x)) for x in t0]
    gt1_dist = [abs(eucl_dist(g, x)) for x in t1]
    mdist = eucl_dist_traj(t0, t1)

    C[1:, 0] = sum(gt0_dist)
    C[0, 1:] = sum(gt1_dist)
    for i in np.arange(n0) + 1:
        for j in np.arange(n1) + 1:
            derp0 = C[i - 1, j] + gt0_dist[i-1]
            derp1 = C[i, j - 1] + gt1_dist[j-1]
            derp01 = C[i - 1, j - 1] + mdist[i-1, j-1]
            C[i, j] = min(derp0, derp1, derp01)
    erp = C[n0, n1]
    return erp

def e_directed_hausdorff(t1, t2, mdist, l_t1, l_t2, t2_dist):
    """
    Usage
    -----
    directed hausdorff distance from trajectory t1 to trajectory t2.
    Parameters
    ----------
    Parameters
    ----------
    param t1 :  l_t1 x 2 numpy_array
    param t2 :  l_t2 x 2 numpy_array
    mdist : len(t1) x len(t2) numpy array, pairwise distance between points of trajectories t1 and t2
    param l_t1: int, length of t1
    param l_t2: int, length of t2
    param t2_dist:  l_t1 x 1 numpy_array,  distances between consecutive points in t2
    Returns
    -------
    dh : float, directed hausdorff from trajectory t1 to trajectory t2
    """
    dh = max([point_to_trajectory(t1[i1], t2, mdist[i1], t2_dist, l_t2) for i1 in range(l_t1)])
    return dh


def e_hausdorff(t1, t2):
    """
    Usage
    -----
    hausdorff distance between trajectories t1 and t2.
    Parameters
    ----------
    param t1 :  len(t1)x2 numpy_array
    param t2 :  len(t2)x2 numpy_array
    Returns
    -------
    h : float, hausdorff from trajectories t1 and t2
    """
    mdist = eucl_dist_traj(t1, t2)
    l_t1 = len(t1)
    l_t2 = len(t2)
    t1_dist = [eucl_dist(t1[it1], t1[it1 + 1]) for it1 in range(l_t1 - 1)]
    t2_dist = [eucl_dist(t2[it2], t2[it2 + 1]) for it2 in range(l_t2 - 1)]

    h = max(e_directed_hausdorff(t1, t2, mdist, l_t1, l_t2, t2_dist),
            e_directed_hausdorff(t2, t1, mdist.T, l_t2, l_t1, t1_dist))
    return 
    
def e_lcss(t0, t1, eps):
    """
    Usage
    -----
    The Longuest-Common-Subsequence distance between trajectory t0 and t1.
    Parameters
    ----------
    param t0 : len(t0)x2 numpy_array
    param t1 : len(t1)x2 numpy_array
    eps : float
    Returns
    -------
    lcss : float
           The Longuest-Common-Subsequence distance between trajectory t0 and t1
    """
    n0 = len(t0)
    n1 = len(t1)
    # An (m+1) times (n+1) matrix
    C = [[0] * (n1 + 1) for _ in range(n0 + 1)]
    for i in range(1, n0 + 1):
        for j in range(1, n1 + 1):
            if eucl_dist(t0[i - 1], t1[j - 1]) < eps:
                C[i][j] = C[i - 1][j - 1] + 1
            else:
                C[i][j] = max(C[i][j - 1], C[i - 1][j])
    lcss = 1 - float(C[n0][n1]) / min([n0, n1])
    return lcss

def ordered_mixed_distance(si, ei, sj, ej, siei, sjej, siei_norm_2, sjej_norm_2):
    PI = math.pi
    HPI = math.pi / 2

    siei_norm = math.sqrt(siei_norm_2)
    sjej_norm = math.sqrt(sjej_norm_2)
    sisj = sj - si
    siej = ej - si

    u1 = (sisj[0] * siei[0] + sisj[1] * siei[1]) / siei_norm_2
    u2 = (siej[0] * siei[0] + siej[1] * siei[1]) / siei_norm_2

    ps = si + u1 * siei
    pe = si + u2 * siei

    cos_theta = max(-1, min(1, (sjej[0] * siei[0] + sjej[1] * siei[1]) / (siei_norm * sjej_norm)))
    theta = math.acos(cos_theta)

    # perpendicular distance
    lpe1 = eucl_dist(sj, ps)
    lpe2 = eucl_dist(ej, pe)
    if lpe1 == 0 and lpe2 == 0:
        dped = 0
    else:
        dped = (lpe1 * lpe1 + lpe2 * lpe2) / (lpe1 + lpe2)

    # parallel_distance
    lpa1 = min(eucl_dist(si, ps), eucl_dist(ei, ps))
    lpa2 = min(eucl_dist(si, pe), eucl_dist(ei, pe))
    dpad = min(lpa1, lpa2)

    # angle_distance
    if 0 <= theta < HPI:
        dad = sjej_norm * math.sin(theta)
    elif HPI <= theta <= PI:
        dad = sjej_norm
    else:
        raise ValueError("WRONG THETA")

    fdist = (dped + dpad + dad) / 3

    return fdist


def mixed_distance(si, ei, sj, ej):
    siei = ei - si
    sjej = ej - sj

    siei_norm_2 = (siei[0] * siei[0]) + (siei[1] * siei[1])
    sjej_norm_2 = (sjej[0] * sjej[0]) + (sjej[1] * sjej[1])

    if sjej_norm_2 > siei_norm_2:
        md = ordered_mixed_distance(sj, ej, si, ei, sjej, siei, sjej_norm_2, siei_norm_2)
    else:
        md = ordered_mixed_distance(si, ei, sj, ej, siei, sjej, siei_norm_2, sjej_norm_2)

    return md


def segments_distance(traj_0, traj_1):
    n0 = len(traj_0)
    n1 = len(traj_1)
    M = np.zeros((n0 - 1, n1 - 1))
    for i in range(n0 - 1):
        for j in range(n1 - 1):
            M[i, j] = mixed_distance(traj_0[i], traj_0[i + 1], traj_1[j], traj_1[j + 1])
    return M

def owd_grid_brut(traj_cell_1, traj_cell_2):
    """
    Usage
    -----
    The owd-distance of trajectory t2 from trajectory t1
    Parameters
    ----------
    param traj_cell_1 :  len(t1)x2 numpy_array
    param traj_cell_2 :  len(t2)x2 numpy_array
    Returns
    -------
    owd : float
           owd-distance of trajectory t2 from trajectory t1
    """
    D = 0
    n = len(traj_cell_1)
    for p1 in traj_cell_1:
        d = [np.linalg.norm(p1 - x) for x in traj_cell_2]
        D += min(d)
    owd = D / n
    return owd

def e_spd(t1, t2, mdist, l_t1, l_t2, t2_dist):
    """
    Usage
    -----
    The spd-distance of trajectory t2 from trajectory t1
    The spd-distance is the sum of the all the point-to-trajectory distance of points of t1 from trajectory t2
    Parameters
    ----------
    param t1 :  l_t1 x 2 numpy_array
    param t2 :  l_t2 x 2 numpy_array
    mdist : len(t1) x len(t2) numpy array, pairwise distance between points of trajectories t1 and t2
    param l_t1: int, length of t1
    param l_t2: int, length of t2
    param t2_dist:  l_t1 x 1 numpy_array,  distances between consecutive points in t2
    Returns
    -------
    spd : float
           spd-distance of trajectory t2 from trajectory t1
    """

    spd = sum([point_to_trajectory(t1[i1], t2, mdist[i1], t2_dist, l_t2) for i1 in range(l_t1)]) / l_t1
    return spd


def e_sspd(t1, t2):
    """
    Usage
    -----
    The sspd-distance between trajectories t1 and t2.
    The sspd-distance isjthe mean of the spd-distance between of t1 from t2 and the spd-distance of t2 from t1.
    Parameters
    ----------
    param t1 :  len(t1)x2 numpy_array
    param t2 :  len(t2)x2 numpy_array
    Returns
    -------
    sspd : float
            sspd-distance of trajectory t2 from trajectory t1
    """
    mdist = eucl_dist_traj(t1, t2)
    l_t1 = len(t1)
    l_t2 = len(t2)
    t1_dist = [eucl_dist(t1[it1], t1[it1 + 1]) for it1 in range(l_t1 - 1)]
    t2_dist = [eucl_dist(t2[it2], t2[it2 + 1]) for it2 in range(l_t2 - 1)]

    sspd = (e_spd(t1, t2, mdist, l_t1, l_t2, t2_dist) + e_spd(t2, t1, mdist.T, l_t2, l_t1, t1_dist)) / 2
    return sspd



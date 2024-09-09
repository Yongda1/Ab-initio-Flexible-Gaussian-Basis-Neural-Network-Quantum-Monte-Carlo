import numpy as np
import scipy


def get_rot(nconf, naip):
    """
    :parameter int nconf: number of configurations
    :parameter int naip: number of auxiliary integration points
    :returns: the integration weights, and the positions of the rotated electron e
    :rtype:  ((naip,) array, (nconf, naip, 3) array)
    """

    if nconf > 0:  # get around a bug(?) when there are zero configurations.
        rot = scipy.spatial.transform.Rotation.random(nconf).as_matrix()
    else:
        rot = np.zeros((0, 3, 3))
    quadrature_grid = generate_quadrature_grids()
    print("quadrature_grid", quadrature_grid)
    print("rot", rot)
    if naip not in quadrature_grid.keys():
        raise ValueError(f"Possible AIPs are one of {quadrature_grid.keys()}")
    points, weights = quadrature_grid[naip]
    rot_vec = np.einsum("jkl,ik->jil", rot, points)
    print("points", points)
    print("weights", weights)
    return weights, rot_vec


def generate_quadrature_grids():
    """
    Generate quadrature grids from Mitas, Shirley, and Ceperley J. Chem. Phys. 95, 3467 (1991)
        https://doi.org/10.1063/1.460849
    All the grids in the Mitas paper are hard-coded here.
    Returns a dictionary whose keys are naip (number of auxiliary points) and whose values are tuples of arrays (points, weights)
    """
    # Generate in Cartesian grids for octahedral symmetry
    octpts = np.mgrid[-1:2, -1:2, -1:2].reshape(3, -1).T
    nonzero_count = np.count_nonzero(octpts, axis=1)
    OA = octpts[nonzero_count == 1]
    OB = octpts[nonzero_count == 2] / np.sqrt(2)
    OC = octpts[nonzero_count == 3] / np.sqrt(3)
    d1 = OC * np.sqrt(3 / 11)
    d1[:, 2] *= 3
    OD = np.concatenate([np.roll(d1, i, axis=1) for i in range(3)])
    OAB = np.concatenate([OA, OB], axis=0)
    OABC = np.concatenate([OAB, OC], axis=0)
    OABCD = np.concatenate([OABC, OD], axis=0)

    # Generate in spherical grids for octahedral symmetry
    def sphere(t_, p_):
        s = np.sin(t_)
        return s * np.cos(p_), s * np.sin(p_), np.cos(t_)

    b_1 = np.arctan(2)
    c_1 = np.arccos((2 + 5**0.5) / (15 + 6 * 5**0.5) ** 0.5)
    c_2 = np.arccos(1 / (15 + 6 * 5**0.5) ** 0.5)
    theta, phi = {}, {}
    theta["A"] = np.array([0, np.pi])
    phi["A"] = np.zeros(2)
    k = np.arange(10)
    theta["B"] = np.tile([b_1, np.pi - b_1], 5)
    phi["B"] = k * np.pi / 5
    c_th1 = np.tile([np.pi - c_1, c_1], 5)
    c_th2 = np.tile([np.pi - c_2, c_2], 5)
    theta["C"] = np.concatenate([c_th1, c_th2])
    phi["C"] = np.tile(k * np.pi / 5, 2)
    I = {g: np.transpose(sphere(theta[g], phi[g])) for g in "ABC"}
    IAB = np.concatenate([I["A"], I["B"]], axis=0)
    IABC = np.concatenate([IAB, I["C"]], axis=0)

    lens = {}
    lens["O"] = [len(x) for x in [OA, OB, OC, OD]]
    lens["I"] = [len(I[s]) for s in "ABC"]

    def repeat(s, *args):
        return np.concatenate([np.repeat(w, l) for w, l in zip(args, lens[s])])

    qgrid = {}
    qgrid[6] = (OA, repeat("O", 1 / 6))
    qgrid[18] = (OAB, repeat("O", 1 / 30, 1 / 15))
    qgrid[26] = (OABC, repeat("O", 1 / 21, 4 / 105, 27 / 840))
    qgrid[50] = (OABCD, repeat("O", 4 / 315, 64 / 2835, 27 / 1280, 14641 / 725760))
    qgrid[12] = (IAB, repeat("I", 1 / 12, 1 / 12))
    qgrid[32] = (IABC, repeat("I", 5 / 168, 5 / 168, 27 / 840))

    return qgrid

output = get_rot(4, 6)
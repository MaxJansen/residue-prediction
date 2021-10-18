import sys
import numpy as np
import torch
import torch.nn.functional as F
from Bio.PDB import *


### Copied from KeOps source code ##############################################
def grid_cluster(x, size):
    r"""Simplistic clustering algorithm which distributes points into cubic bins.
    Args:
        x ((M,D) Tensor): List of points :math:`x_i \in \mathbb{R}^D`.
        size (float or (D,) Tensor): Dimensions of the cubic cells ("voxels").
    Returns:
        (M,) IntTensor:
        Vector of integer **labels**. Two points ``x[i]`` and ``x[j]`` are
        in the same cluster if and only if ``labels[i] == labels[j]``.
        Labels are sorted in a compact range :math:`[0,C)`,
        where :math:`C` is the number of non-empty cubic cells.
    Example:
        >>> x = torch.Tensor([ [0.], [.1], [.9], [.05], [.5] ])  # points in the unit interval
        >>> labels = grid_cluster(x, .2)  # bins of size .2
        >>> print( labels )
        tensor([0, 0, 2, 0, 1], dtype=torch.int32)
    """
    with torch.no_grad():
        # Quantize the points' positions
        if x.shape[1] == 1:
            weights = torch.IntTensor(
                [1],
            ).to(x.device)
        elif x.shape[1] == 2:
            weights = torch.IntTensor(
                [2 ** 10, 1],
            ).to(x.device)
        elif x.shape[1] == 3:
            weights = torch.IntTensor([2 ** 20, 2 ** 10, 1]).to(x.device)
        else:
            raise NotImplementedError()
        x_ = (x / size).floor().int()
        x_ *= weights
        lab = x_.sum(1)  # labels
        lab = lab - lab.min()

        # Replace arbitrary labels with unique identifiers in a compact arange
        u_lab = torch.unique(lab).sort()[0]
        N_lab = len(u_lab)
        foo = torch.empty(u_lab.max() + 1, dtype=torch.int32, device=x.device)
        foo[u_lab] = torch.arange(N_lab, dtype=torch.int32, device=x.device)
        lab = foo[lab]

    return lab
################################################################################


def subsample(x, batch=None, scale=1.0):
    """Subsamples the point cloud using a grid (cubic) clustering scheme.

    The function returns one average sample per cell, as described in Fig. 3.e)
    of the paper.

    Args:
        x (Tensor): (N,3) point cloud.
        batch (integer Tensor, optional): (N,) batch vector, as in PyTorch_geometric.
            Defaults to None.
        scale (float, optional): side length of the cubic grid cells. Defaults to 1 (Angstrom).

    Returns:
        (M,3): sub-sampled point cloud, with M <= N.
    """

    if batch is None:  # Single protein case:
        if True:  # Use a fast scatter_add_ implementation
            labels = grid_cluster(x, scale).long()
            C = labels.max() + 1

            # We append a "1" to the input vectors, in order to
            # compute both the numerator and denominator of the "average"
            #  fraction in one pass through the data.
            x_1 = torch.cat((x, torch.ones_like(x[:, :1])), dim=1)
            D = x_1.shape[1]
            points = torch.zeros_like(x_1[:C])
            points.scatter_add_(0, labels[:, None].repeat(1, D), x_1)
            return (points[:, :-1] / points[:, -1:]).contiguous()

        else:  # Older implementation;
            points = scatter(points * weights[:, None], labels, dim=0)
            weights = scatter(weights, labels, dim=0)
            points = points / weights[:, None]

    else:  # We process proteins using a for loop.
        # This is probably sub-optimal, but I don't really know
        # how to do more elegantly (this type of computation is
        # not super well supported by PyTorch).
        batch_size = torch.max(batch).item() + 1  # Typically, =32
        points, batches = [], []
        for b in range(batch_size):
            p = subsample(x[batch == b], scale=scale)
            points.append(p)
            batches.append(b * torch.ones_like(batch[: len(p)]))

    return torch.cat(points, dim=0), torch.cat(batches, dim=0)


def ranges_slices(batch):
    """Helper function for the diagonal ranges function."""
    Ns = batch.bincount()
    indices = Ns.cumsum(0)
    ranges = torch.cat((0 * indices[:1], indices))
    ranges = (
        torch.stack((ranges[:-1], ranges[1:])).t().int().contiguous().to(batch.device)
    )
    slices = (1 + torch.arange(len(Ns))).int().to(batch.device)

    return ranges, slices


def diagonal_ranges(batch_x=None, batch_y=None):
    """Encodes the block-diagonal structure associated to a batch vector."""

    if batch_x is None and batch_y is None:
        return None

    ranges_x, slices_x = ranges_slices(batch_x)
    ranges_y, slices_y = ranges_slices(batch_y)

    return ranges_x, slices_x, ranges_y, ranges_y, slices_y, ranges_x


def soft_distances(x, y, batch_x, batch_y, smoothness=0.01, atomtypes=None):
    """Computes a soft distance function to the atom centers of a protein.

    Implements Eq. (1) of the paper in a fast and numerically stable way.

    Args:
        x (Tensor): (N,3) atom centers.
        y (Tensor): (M,3) sampling locations.
        batch_x (integer Tensor): (N,) batch vector for x, as in PyTorch_geometric.
        batch_y (integer Tensor): (M,) batch vector for y, as in PyTorch_geometric.
        smoothness (float, optional): atom radii if atom types are not provided. Defaults to .01.
        atomtypes (integer Tensor, optional): (N,6) one-hot encoding of the atom chemical types. Defaults to None.

    Returns:
        Tensor: (M,) values of the soft distance function on the points `y`.
    """
    # Build the (N, M, 1) symbolic matrix of squared distances:
    x_i = x[:, None, :]  # (N, 1, 3) atoms
    y_j = y[None, :, :]  # (1, M, 3) sampling points
    D_ij = ((x_i - y_j) ** 2).sum(-1, keepdim=True)  # (N, M, 1) squared distances

    # Use a block-diagonal sparsity mask to support heterogeneous batch processing:
    D_ij.ranges = diagonal_ranges(batch_x, batch_y)

    if atomtypes is not None:
        # Turn the one-hot encoding "atomtypes" into a vector of diameters "smoothness_i":
        # (N, 6)  -> (N, 1, 1)  (There are 6 atom types)
        atomic_radii = torch.tensor(
            [170, 110, 152, 155, 180, 190], device=x.device
        )
        atomic_radii = atomic_radii / atomic_radii.min()
        atomtype_radii = atomtypes * atomic_radii[None, :]  # n_atoms, n_atomtypes
        # smoothness = atomtypes @ atomic_radii  # (N, 6) @ (6,) = (N,)
        smoothness = torch.sum(
            smoothness * atomtype_radii, dim=1, keepdim=False
        )  # n_atoms, 1
        smoothness_i = smoothness[:, None, None]

        # Compute an estimation of the mean smoothness in a neighborhood
        # of each sampling point:
        # density = (-D_ij.sqrt()).exp().sum(0).view(-1)  # (M,) local density of atoms
        # smooth = (smoothness_i * (-D_ij.sqrt()).exp()).sum(0).view(-1)  # (M,)
        # mean_smoothness = smooth / density  # (M,)

        # soft_dists = -mean_smoothness * (
        #    (-D_ij.sqrt() / smoothness_i).logsumexp(dim=0)
        # ).view(-1)
        mean_smoothness = (-D_ij.sqrt()).exp().sum(0)
        mean_smoothness_j = mean_smoothness[None, :, :]
        mean_smoothness = (
            smoothness_i * (-D_ij.sqrt()).exp() / mean_smoothness_j
        )  # n_atoms, n_points, 1
        mean_smoothness = mean_smoothness.sum(0).view(-1)
        soft_dists = -mean_smoothness * (
            (-D_ij.sqrt() / smoothness_i).logsumexp(dim=0)
        ).view(-1)

    else:
        soft_dists = -smoothness * ((-D_ij.sqrt() / smoothness).logsumexp(dim=0)).view(
            -1
        )

    return soft_dists


def atoms_to_points_normals(
    atoms,
    batch,
    distance=1.05,
    smoothness=0.5,
    resolution=1.0,
    nits=4,
    atomtypes=None,
    sup_sampling=20,
    variance=0.1,
):
    """Turns a collection of atoms into an oriented point cloud.

    Sampling algorithm for protein surfaces, described in Fig. 3 of the paper.

    Args:
        atoms (Tensor): (N,3) coordinates of the atom centers `a_k`.
        batch (integer Tensor): (N,) batch vector, as in PyTorch_geometric.
        distance (float, optional): value of the level set to sample from
            the smooth distance function. Defaults to 1.05.
        smoothness (float, optional): radii of the atoms, if atom types are
            not provided. Defaults to 0.5.
        resolution (float, optional): side length of the cubic cells in
            the final sub-sampling pass. Defaults to 1.0.
        nits (int, optional): number of iterations . Defaults to 4.
        atomtypes (Tensor, optional): (N,6) one-hot encoding of the atom
            chemical types. Defaults to None.

    Returns:
        (Tensor): (M,3) coordinates for the surface points `x_i`.
        (Tensor): (M,3) unit normals `n_i`.
        (integer Tensor): (M,) batch vector, as in PyTorch_geometric.
    """
    # a) Parameters for the soft distance function and its level set:
    T = distance

    N, D = atoms.shape
    B = sup_sampling  # Sup-sampling ratio

    # Batch vectors:
    batch_atoms = batch
    batch_z = batch[:, None].repeat(1, B).view(N * B)

    # b) Draw N*B points at random in the neighborhood of our atoms
    z = atoms[:, None, :] + 10 * T * torch.randn(N, B, D).type_as(atoms)
    z = z.view(-1, D)  # (N*B, D)

    # We don't want to backprop through a full network here!
    atoms = atoms.detach().contiguous()
    z = z.detach().contiguous()

    # N.B.: Test mode disables the autograd engine: we must switch it on explicitely.
    with torch.enable_grad():
        if z.is_leaf:
            z.requires_grad = True

        # c) Iterative loop: gradient descent along the potential
        # ".5 * (dist - T)^2" with respect to the positions z of our samples
        for it in range(nits):
            dists = soft_distances(
                atoms,
                z,
                batch_atoms,
                batch_z,
                smoothness=smoothness,
                atomtypes=atomtypes,
            )
            Loss = ((dists - T) ** 2).sum()
            g = torch.autograd.grad(Loss, z)[0]
            z.data -= 0.5 * g

        # d) Only keep the points which are reasonably close to the level set:
        dists = soft_distances(
            atoms, z, batch_atoms, batch_z, smoothness=smoothness, atomtypes=atomtypes
        )
        margin = (dists - T).abs()
        mask = margin < variance * T

        # d') And remove the points that are trapped *inside* the protein:
        zz = z.detach()
        zz.requires_grad = True
        for it in range(nits):
            dists = soft_distances(
                atoms,
                zz,
                batch_atoms,
                batch_z,
                smoothness=smoothness,
                atomtypes=atomtypes,
            )
            Loss = (1.0 * dists).sum()
            g = torch.autograd.grad(Loss, zz)[0]
            normals = F.normalize(g, p=2, dim=-1)  # (N, 3)
            zz = zz + 1.0 * T * normals

        dists = soft_distances(
            atoms, zz, batch_atoms, batch_z, smoothness=smoothness, atomtypes=atomtypes
        )
        mask = mask & (dists > 1.5 * T)

        z = z[mask].contiguous().detach()
        batch_z = batch_z[mask].contiguous().detach()

        # e) Subsample the point cloud:
        points, batch_points = subsample(z, batch_z, scale=resolution)

        # f) Compute the normals on this smaller point cloud:
        p = points.detach()
        p.requires_grad = True
        dists = soft_distances(
            atoms,
            p,
            batch_atoms,
            batch_points,
            smoothness=smoothness,
            atomtypes=atomtypes,
        )
        Loss = (1.0 * dists).sum()
        g = torch.autograd.grad(Loss, p)[0]
        normals = F.normalize(g, p=2, dim=-1)  # (N, 3)

    return points.detach(), normals.detach(), batch_points.detach()


ele2num = {"C": 0, "H": 1, "O": 2, "N": 3, "S": 4, "SE": 5}


def load_structure_np(fname, center):
    """Loads a .ply mesh to return a point cloud and connectivity."""
    # Load the data
    parser = PDBParser()
    structure = parser.get_structure("structure", fname)
    atoms = structure.get_atoms()

    coords = []
    types = []
    for atom in atoms:
        coords.append(atom.get_coord())
        types.append(ele2num[atom.element])

    coords = np.stack(coords)
    types_array = np.zeros((len(types), len(ele2num)))
    for i, t in enumerate(types):
        types_array[i, t] = 1.0

    # Normalize the coordinates, as specified by the user:
    if center:
        coords = coords - np.mean(coords, axis=0, keepdims=True)

    return {"xyz": coords, "types": types_array}


if __name__ == "__main__":
    # --------------------------------------------------------------------------
    # Minimal working example
    # --------------------------------------------------------------------------
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt

    # Load protein, extract coordinates from PDB file
    file = sys.argv[1] + ".pdb"
    protein_np = load_structure_np(file, True)
    atoms = torch.tensor(protein_np['xyz'][::5, :])
    atomtypes = torch.tensor(protein_np['types'][::5, :])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    coords = protein_np['xyz'][::5, :]
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    ax.scatter(x, y, z, c='r', marker='o')

    # Use dMaSIF procedure to extract surface points and normals
    points, normals, batch_points = \
        atoms_to_points_normals(atoms, torch.zeros(len(atoms), dtype=int),
                                atomtypes=atomtypes)
    print(points.shape)
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b', marker='o')
    plt.show()
    exit()

    # Use dMaSIF procedure to get per-point features
    # (types and distances of x closest atoms)
    pass
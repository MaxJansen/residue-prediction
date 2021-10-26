import torch.nn as nn
import torch.utils.data


class EGCN(nn.Module):
    """
    Implements the Equivariant Graph Convolutional Layer defined by
    Equations (3)-(6) in:
    Satorras, Victor Garcia, Emiel Hoogeboom, and Max Welling.
    "E(n) equivariant graph neural networks."
    arXiv preprint arXiv:2102.09844 (2021).
    :param feat_dim: Number of features per node
    :param coord_dim: Spatial dimension
    :param edge_dim: Number of features per edge
    :param msg_dim: Size of the edge embedding (message)
    :param feat_out_dim: Size of the updated feature embedding
    :param update_coord: Determines whether Eq. (4), the coordinate update,
    will be computed or not
    :param infer_edges: Compute soft estimation of edge values according to
    Section 3.3 in the paper
    """
    def __init__(self, feat_dim, coord_dim=3, edge_dim=0, msg_dim=32,
                 feat_out_dim=None, update_coord=True, infer_edges=False):
        super(EGCN, self).__init__()

        self.nf = feat_dim
        self.n_dim = coord_dim
        self.nf_e = edge_dim
        self.nf_m = msg_dim
        self.nf_out = feat_dim if feat_out_dim is None else feat_out_dim

        # Edge operation
        hidden_dim = int(msg_dim / 2)  # arbitrary choice
        self.phi_e = nn.Sequential(
            nn.Conv2d(2 * feat_dim + 1 + edge_dim, hidden_dim, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.2),
            nn.Conv2d(hidden_dim, msg_dim, 1),
            nn.BatchNorm2d(msg_dim)
        )

        # Weight function for coordinate update
        hidden_dim = 2 * msg_dim
        self.phi_x = nn.Sequential(
            nn.Conv2d(msg_dim, hidden_dim, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.2),
            nn.Conv2d(hidden_dim, 1, 1),
            nn.BatchNorm2d(1)
        ) if update_coord else None

        # Node operation
        hidden_dim = 2 * (feat_dim + msg_dim)
        self.phi_h = nn.Sequential(
            nn.Conv1d(feat_dim + msg_dim, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.2),
            nn.Conv1d(hidden_dim, self.nf_out, 1),
            nn.BatchNorm1d(self.nf_out)
        )

        # Soft edge estimation
        self.phi_inf = nn.Sequential(
            nn.Conv2d(msg_dim, 1, 1),
            nn.Sigmoid()
        ) if infer_edges else None

    def forward(self, x, h, a=None, adj_mat=None):
        """
        Forward pass
        :param x: Coordinate embedding (n_batch, n_dim, n_nodes)
        :param h: Feature embeddings (n_batch, n_channel, n_nodes)
        :param a: Edge attributes (n_batch, n_channel, n_nodes, n_nodes)
        :param adj_mat: Adjacency matrix (n_batch, n_nodes, n_nodes)
        :return: Updated coordinates, updated node embedding
        """
        n_nodes = x.size(-1)

        # (n_batch, n_dim, n_nodes, n_nodes)
        dx = x.unsqueeze(3) - x.unsqueeze(2)

        # Equation (3): Edge embedding m_ij
        h_i = h.unsqueeze(3).repeat(1, 1, 1, n_nodes)
        h_j = h.unsqueeze(2).repeat(1, 1, n_nodes, 1)
        # (n_batch, n_input, n_nodes, n_nodes)
        phi_e_input = torch.cat((h_i, h_j,
                                 torch.sum(dx**2, dim=1, keepdim=True)), dim=1)
        if self.nf_e > 0:
            phi_e_input = torch.cat((phi_e_input, a), dim=1)
        # (n_batch, nf_msg, n_nodes, n_nodes)
        m_ij = self.phi_e(phi_e_input)

        # Equation (4): Coordinate update
        if self.phi_x is not None:
            coord_weights = self.phi_x(m_ij).squeeze(1)
            # j =/= i is fulfilled because dx(i,i) = 0
            x = x + \
                torch.einsum('bdij,bij->bdi', dx, coord_weights) / (n_nodes - 1)

        # Equation (5): Aggregated message m_i
        if self.phi_inf is not None:
            e_ij = self.phi_inf(m_ij)  # (n_batch, 1, n_nodes, n_nodes)
            m_ij = e_ij * m_ij
        if adj_mat is not None:
            # m_ij.masked_fill_(~adj_mat.unsqueeze(1), 0.0)
            m_ij = m_ij * adj_mat.unsqueeze(1)
        # (n_batch, nf_msg, n_nodes)
        m_i = torch.sum(m_ij, dim=-1)

        # Equation (6): Feature update
        h = self.phi_h(torch.cat((h, m_i), dim=1))

        return x, h


class EGNNdiscriminator(nn.Module):
    """
    Inspired by the network presented in Appendix C.3 of the EGNN paper:
    Satorras, Victor Garcia, Emiel Hoogeboom, and Max Welling.
    "E(n) equivariant graph neural networks."
    arXiv preprint arXiv:2102.09844 (2021).
    """
    def __init__(self, n_feat=5, agg_func='mean', nf=32, update_coord=False,
                 infer_edges=False, use_adj=False):
        super(EGNNdiscriminator, self).__init__()

        assert agg_func in ['sum', 'mean']
        self.agg_func = getattr(torch, agg_func)
        self.use_adj = use_adj

        self.n_feat = n_feat

        # EGCN layers
        self.embed = nn.ModuleList([
            EGCN(n_feat, 3, edge_dim=0, msg_dim=nf, feat_out_dim=n_feat,
                 update_coord=update_coord, infer_edges=infer_edges),
            EGCN(n_feat, 3, edge_dim=0, msg_dim=nf, feat_out_dim=2 * n_feat,
                 update_coord=update_coord, infer_edges=infer_edges),
            EGCN(2 * n_feat, 3, edge_dim=0, msg_dim=2 * nf, feat_out_dim=2 * n_feat,
                 update_coord=update_coord, infer_edges=infer_edges),
            EGCN(2 * n_feat, 3, edge_dim=0, msg_dim=2 * nf, feat_out_dim=4 * n_feat,
                 update_coord=update_coord, infer_edges=infer_edges),
            EGCN(4 * n_feat, 3, edge_dim=0, msg_dim=4 * nf, feat_out_dim=4 * n_feat,
                 update_coord=update_coord, infer_edges=infer_edges),
            EGCN(4 * n_feat, 3, edge_dim=0, msg_dim=4 * nf, feat_out_dim=8 * n_feat,
                 update_coord=update_coord, infer_edges=infer_edges)
        ])

        # input shape: [nBatch, nFeat, nPoints]
        self.node_wise_mlp = nn.Sequential(
            # output of the EGNN is forwarded node-wise through two MLP layers
            # nn.Conv1d(3 + 4 * n_feat, 128, 1),
            nn.Conv1d(8 * n_feat, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, 1),
            # nn.BatchNorm1d(256),
        )

        # operations applied to the averaged embedding in order to produce the
        # output value
        self.out_mlp = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Retrieve coordinates, features, and edges
        # x: [nb, ncoord+nfeat, npoint]
        # coordinates: [nb, 3, npoint]
        # features: [nb, nfeat, npoint]
        # adj_mat: [nb, npoint, npoint]
        coord, feat = x[:, :3, :], x[:, 3:3+self.n_feat, :]
        adj_mat = x[:, 3+self.n_feat:, :] if self.use_adj else None

        for layer in self.embed:
            coord, feat = layer(coord, feat, adj_mat=adj_mat)
        # x = torch.cat((coord, feat), dim=1)
        # x = self.node_wise_mlp(x)
        x = self.node_wise_mlp(feat)

        # use symmetry function to achieve permutation invariance
        x = self.agg_func(x, dim=-1)

        x = self.out_mlp(x)
        return x.squeeze(dim=1)


if __name__ == "__main__":
    model = EGNNdiscriminator(nf=32, update_coord=False, infer_edges=True,
                              use_adj=True)
    n_param_tot = sum(p.numel() for p in model.parameters())
    print(f'{type(model).__name__} has {n_param_tot} parameters.')

    test_input = torch.rand((16, 8, 200))  # 8 = 3 coord + 5 feat
    edges = torch.block_diag(*[torch.ones(10, 10) for _ in range(20)]).unsqueeze(0).repeat(16, 1, 1)
    test_input = torch.cat((test_input, edges), dim=1)
    output = model(test_input)

    print(output)

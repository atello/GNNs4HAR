import logging
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU
from torch_geometric.nn import BatchNorm, LayerNorm, ChebConv, GraphNorm
from torch_geometric.nn import GATConv, GATv2Conv, TransformerConv
from torch_geometric.nn import GraphConv, GINEConv
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.utils import to_dense_batch

from utils import corrupt_graph

logging.basicConfig(level=logging.NOTSET)

EPS = 1e-15


class GdmGNN(torch.nn.Module):
    def __init__(self, net_params):
        super(GdmGNN, self).__init__()

        self.sigma = nn.Parameter(torch.rand(1))  # Learnable Gaussian Kernel parameter
        self.gdmLinear = Linear(net_params["input_dim"], net_params["input_dim"])
        self.GNN = GraphConvNet(net_params=net_params)

    def forward(self, x, edge_index, edge_weight, batch):
        batch_size = batch.max().item() + 1

        x = self.gdmLinear(x)
        i = 0
        total_nodes = 0
        xs = []
        edges = []
        weights = []
        batch_idxs = []
        # for adj in x0.reshape(-1, num_nodes, f_dim):
        for win in to_dense_batch(x, batch)[0]:
            # similarity_matrix = utils.gaussian_kernel(win)
            similarity_matrix = self.compute_gaussian_kernel(win)

            edge_index = torch.stack([e for e in torch.nonzero(similarity_matrix) if e[0] != e[1]])
            edge_weight = similarity_matrix[edge_index[:, 0], edge_index[:, 1]]

            # # unique nodes in edge_index to remove isolated nodes
            # unique_nodes = edge_index.unique()
            # nodes_dict = {key.item(): i for i, key in enumerate(unique_nodes)}
            #
            # win = win[unique_nodes]
            # edge_index = torch.tensor([[nodes_dict[e[0].item()] + (i * total_nodes),
            #                             nodes_dict[e[1].item()] + (i * total_nodes)]
            #                            for e in zip(edge_index[:, 0], edge_index[:, 1])], dtype=torch.long)
            # total_nodes += len(win)
            # assert len(edge_index.unique()) == len(win), "Isolated Nodes"

            edge_index = torch.tensor([[e[0].item() + (i * len(win)), e[1].item() + (i * len(win))]
                                       for e in zip(edge_index[:, 0], edge_index[:, 1])], dtype=torch.long)

            # xs.append(win)
            edges.append(edge_index)
            weights.append(edge_weight)
            # batch_idxs.append(torch.full((1, len(win)), i))

            i = i + 1

        # x2 = torch.cat(xs, dim=0).to(x.device)
        edge_index = torch.cat(edges, dim=0).t().contiguous().to(x.device)
        edge_weight = torch.cat(weights, dim=0).to(x.device)
        # batch2 = torch.cat(batch_idxs, dim=1).to(x.device)

        x = self.GNN(x, edge_index, edge_weight, batch)
        return x

    def compute_gaussian_kernel(self, x):
        distance_matrix = torch.sum((x[:, None, :] - x[None, :, :]) ** 2, dim=-1)
        adj_matrix = torch.exp(-distance_matrix / (2 * self.sigma ** 2))
        return adj_matrix


class MPoolGNN(torch.nn.Module):
    def __init__(self, net_params):
        super(MPoolGNN, self).__init__()
        # torch.manual_seed(42)
        self.net_params = net_params

        # self.bn = BatchNorm(net_params["input_dim"])
        self.convs = torch.nn.ModuleList()
        self.convs.append(GraphConv(net_params["input_dim"], net_params["hidden_dim"], aggr=net_params["aggr"]))

        for _ in range(net_params["num_layers"] - 1):
            self.convs.append(GraphConv(net_params["hidden_dim"],
                                        net_params["hidden_dim"],
                                        aggr=net_params["aggr"]))

        if self.net_params["batch_norm"]:
            self.norms = torch.nn.ModuleList()

            for _ in range(net_params["num_layers"] - 1):
                self.norms.append(BatchNorm(net_params["hidden_dim"]))

        self.lin1 = Linear(net_params["hidden_dim"] * 3, net_params["hidden_dim"])
        self.lin2 = Linear(net_params["hidden_dim"], net_params["out_dim"])

    def forward(self, x, edge_index, edge_weight, batch):

        # x = self.bn(x)
        for i in range(self.net_params["num_layers"] - 1):
            x = self.convs[i](x, edge_index, edge_weight)

            if self.net_params["batch_norm"]:
                x = self.norms[i](x)

            x = x.relu()
            dropout = self.net_params["conv_dropout"]
            x = F.dropout(x, p=dropout, training=self.training)

        x = self.convs[self.net_params["num_layers"] - 1](x, edge_index, edge_weight)

        sumpool = global_add_pool(x, batch)  # [batch_size, hidden_channels]
        meanpool = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
        maxpool = global_max_pool(x, batch)  # [batch_size, hidden_channels]

        x = torch.cat((sumpool, meanpool, maxpool), dim=1)

        # 3. Apply a final classifier
        x = self.lin1(x)
        x = x.relu()
        x = F.dropout(x, p=self.net_params["classifier_dropout"], training=self.training)
        x = self.lin2(x)

        if self.net_params["is_explainer"]:
            x = F.softmax(x, dim=-1)

        return x


class GraphConvNet(torch.nn.Module):
    def __init__(self, net_params):
        super(GraphConvNet, self).__init__()
        # torch.manual_seed(42)
        self.net_params = net_params

        # self.bn = BatchNorm(net_params["input_dim"])
        self.convs = torch.nn.ModuleList()
        self.convs.append(GraphConv(net_params["input_dim"], net_params["hidden_dim"], aggr=net_params["aggr"]))

        for _ in range(net_params["num_layers"] - 1):
            self.convs.append(GraphConv(net_params["hidden_dim"],
                                        net_params["hidden_dim"],
                                        aggr=net_params["aggr"]))

        if self.net_params["batch_norm"]:
            self.norms = torch.nn.ModuleList()

            for _ in range(net_params["num_layers"] - 1):
                self.norms.append(BatchNorm(net_params["hidden_dim"]))

        self.lin1 = Linear(net_params["hidden_dim"], net_params["hidden_dim"])
        self.lin2 = Linear(net_params["hidden_dim"], net_params["out_dim"])

    def forward(self, x, edge_index, edge_weight, batch):

        # x = self.bn(x)
        for i in range(self.net_params["num_layers"] - 1):
            x = self.convs[i](x, edge_index, edge_weight)

            if self.net_params["batch_norm"]:
                x = self.norms[i](x)

            x = x.relu()
            dropout = self.net_params["conv_dropout"]
            x = F.dropout(x, p=dropout, training=self.training)

        x = self.convs[self.net_params["num_layers"] - 1](x, edge_index, edge_weight)

        if self.net_params["global_pooling"] == "add":
            x = global_add_pool(x, batch)  # [batch_size, hidden_channels]
        elif self.net_params["global_pooling"] == "mean":
            x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
        elif self.net_params["global_pooling"] == "max":
            x = global_max_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = self.lin1(x)
        x = x.relu()
        x = F.dropout(x, p=self.net_params["classifier_dropout"], training=self.training)
        x = self.lin2(x)

        if self.net_params["is_explainer"]:
            x = F.softmax(x, dim=-1)

        return x


class GATNet(torch.nn.Module):
    def __init__(self, input_dim, hc, heads, out_channels):
        super(GATNet, self).__init__()

        self.bn = BatchNorm(input_dim)
        self.conv1 = GATConv(input_dim, hc, heads, edge_dim=1, dropout=0.6)
        self.conv2 = GATConv(hc * heads, hc, heads, edge_dim=1, dropout=0.2)
        self.conv3 = GATConv(hc * heads, hc, heads, edge_dim=1, dropout=0.2)

        self.lin1 = Linear(hc * heads, 2 * hc)
        self.lin2 = Linear(2 * hc, out_channels)

    def forward(self, x, edge_index, edge_weights, batch):
        x = self.bn(x)
        x = self.conv1(x, edge_index, edge_weights)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_weights)
        x = F.relu(x)
        x = self.conv3(x, edge_index, edge_weights)

        x = global_max_pool(x, batch)

        x = self.lin1(x)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return x


class GAT2Net(torch.nn.Module):
    def __init__(self, params):
        super(GAT2Net, self).__init__()
        # torch.manual_seed(42)
        self.input_dim = params["input_dim"]
        self.hidden_dim = params["hidden_dim"]
        self.out_dim = params["out_dim"]
        self.heads = params["heads"]
        self.add_batch_norm = params["batch_norm"]
        self.aggr = params["aggr"]
        self.global_pooling = params["global_pooling"]
        self.conv_dropout = params["conv_dropout"]
        self.classifier_dropout = params["classifier_dropout"]

        if self.add_batch_norm:
            self.bn1 = BatchNorm(self.input_dim)

        self.conv1 = GATv2Conv(self.input_dim, self.hidden_dim, self.heads, edge_dim=1,
                               dropout=self.conv_dropout)
        self.conv2 = GATv2Conv(self.hidden_dim * self.heads, self.hidden_dim, self.heads, edge_dim=1,
                               dropout=self.conv_dropout)
        self.conv3 = GATv2Conv(self.hidden_dim * self.heads, self.hidden_dim, self.heads, edge_dim=1,
                               dropout=self.conv_dropout, concat=False, aggr=self.aggr)

        self.lin1 = Linear(self.hidden_dim, self.out_dim)

    def forward(self, x, edge_index, edge_weight, batch):

        if self.add_batch_norm:
            x = self.bn1(x)
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.conv3(x, edge_index, edge_weight)

        if self.global_pooling == "add":
            x = global_add_pool(x, batch)  # [batch_size, hidden_channels]
        elif self.global_pooling == "mean":
            x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
        elif self.global_pooling == "max":
            x = global_max_pool(x, batch)  # [batch_size, hidden_channels]

        x = F.dropout(x, p=self.classifier_dropout, training=self.training)
        x = self.lin1(x)
        return x


class TransfNet(torch.nn.Module):
    def __init__(self, input_dim, hc, heads, out_channels):
        super(TransfNet, self).__init__()

        self.conv1 = TransformerConv(input_dim, hc, heads, edge_dim=1)
        self.conv2 = TransformerConv(hc * heads, hc, heads, edge_dim=1)
        self.conv3 = TransformerConv(hc * heads, hc, heads, edge_dim=1)

        self.lin1 = Linear(hc * heads, out_channels)

    def forward(self, x, edge_index, edge_weights, batch, return_attention_weights=False):
        x = self.conv1(x, edge_index, edge_weights.reshape(-1, 1))
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_weights.reshape(-1, 1))
        x = F.relu(x)
        x = self.conv3(x, edge_index, edge_weights.reshape(-1, 1))

        x = global_max_pool(x, batch)

        x = self.lin1(x)
        return x


class GINENet(torch.nn.Module):
    def __init__(self, net_params):
        super(GINENet, self).__init__()
        self.net_params = net_params

        # self.mlp = MLP(in_channels=net_params["input_dim"],
        #                hidden_channels=2 * net_params["hidden_dim"],
        #                out_channels=net_params["hidden_dim"],
        #                num_layers=3)
        nn = Sequential(Linear(net_params["input_dim"], 2 * net_params["input_dim"]), ReLU(),
                        Linear(2 * net_params["input_dim"], net_params["hidden_dim"]))
        self.conv1 = GINEConv(nn, train_eps=True)
        # self.conv2 = GINEConv(self.mlp)

        self.lin1 = Linear(net_params["hidden_dim"], net_params["out_dim"])

    def forward(self, x, edge_index, edge_weights, batch):
        x = self.conv1(x, edge_index, edge_weights)
        # x = F.relu(x)
        # x = self.conv2(x, edge_index, edge_weights)

        x = global_max_pool(x, batch)
        x = F.dropout(x, p=self.net_params["classifier_dropout"], training=self.training)
        x = self.lin1(x)
        return x


# ******************* Residual connections *******************


class GResBlock(torch.nn.Module):
    def __init__(self, in_dim, out_dim, hc, heads=2):
        super(GResBlock, self).__init__()

        self.conv1 = GATv2Conv(in_dim, hc, heads, concat=True)
        self.lynrm1 = LayerNorm(hc * heads)
        self.conv2 = GATv2Conv(hc * heads, out_dim, 1, concat=False)
        self.lynrm2 = LayerNorm(out_dim)

    def forward(self, x, edge_index):
        x_0 = torch.clone(x)
        x = self.conv1(x, edge_index)
        x = self.lynrm1(x)
        x = F.leaky_relu(x)
        x = self.conv2(x, edge_index) + x_0
        x = self.lynrm2(x)
        x = F.leaky_relu(x)
        return x


class GATResClassifier(torch.nn.Module):
    def __init__(self, input_dim, out_dim, num_channels, global_pooling,
                 num_blocks=3, heads=2, dropout=0.5):
        super(GATResClassifier, self).__init__()

        self.global_pooling = global_pooling
        self.dropout = dropout
        self.heads = heads

        nc = num_channels
        self.num_blocks = num_blocks

        self.lin0 = Linear(input_dim, nc)
        self.blocks = nn.ModuleList()

        for _ in range(self.num_blocks):
            block = GResBlock(nc, nc, nc, heads=self.heads)
            self.blocks.append(block)

        self.lin1 = Linear(nc, nc)
        self.lin2 = Linear(nc, out_dim)

    def forward(self, x, edge_index, edge_weight, batch):
        x = self.lin0(x)

        for i in range(self.num_blocks):
            x = self.blocks[i](x, edge_index)

        if self.global_pooling == "add":
            readout = global_add_pool(x, batch)  # [batch_size, hidden_channels]
        elif self.global_pooling == "mean":
            readout = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
        elif self.global_pooling == "max":
            readout = global_max_pool(x, batch)  # [batch_size, hidden_channels]

        x = self.lin1(readout)
        x = x.relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        return x


# ******************* Contrastive Learning *******************

class GAT2Enconder(torch.nn.Module):
    def __init__(self, params):
        super(GAT2Enconder, self).__init__()
        # torch.manual_seed(42)
        self.input_dim = params["input_dim"]
        self.hidden_dim = params["hidden_dim"]
        self.out_dim = params["out_dim"]
        self.heads = params["heads"]
        self.add_batch_norm = params["batch_norm"]
        self.aggr = params["aggr"]
        self.global_pooling = params["global_pooling"]
        self.conv_dropout = params["conv_dropout"]
        self.classifier_dropout = params["classifier_dropout"]

        if self.add_batch_norm:
            self.bn1 = BatchNorm(self.input_dim)

        self.conv1 = GATv2Conv(self.input_dim, self.hidden_dim, self.heads, edge_dim=1,
                               dropout=self.conv_dropout)
        self.conv2 = GATv2Conv(self.hidden_dim * self.heads, self.hidden_dim, self.heads, edge_dim=1,
                               dropout=self.conv_dropout)
        self.conv3 = GATv2Conv(self.hidden_dim * self.heads, self.hidden_dim, self.heads, edge_dim=1,
                               dropout=self.conv_dropout, concat=False, aggr=self.aggr)

        self.lin1 = Linear(self.hidden_dim, self.out_dim)

    def forward(self, x, edge_index, edge_weight, batch):

        if self.add_batch_norm:
            x = self.bn1(x)
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.conv3(x, edge_index, edge_weight)

        if self.global_pooling == "add":
            readout = global_add_pool(x, batch)  # [batch_size, hidden_channels]
        elif self.global_pooling == "mean":
            readout = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
        elif self.global_pooling == "max":
            readout = global_max_pool(x, batch)  # [batch_size, hidden_channels]

        # x = F.dropout(x, p=self.classifier_dropout, training=self.training)
        # x = self.lin1(x)
        return readout


class GraphConvEncoder(torch.nn.Module):
    def __init__(self, net_params):
        super(GraphConvEncoder, self).__init__()
        # torch.manual_seed(42)
        self.net_params = net_params

        self.convs = torch.nn.ModuleList()
        self.convs.append(GraphConv(net_params["input_dim"], net_params["hidden_dim"], aggr=net_params["aggr"]))

        for _ in range(net_params["num_layers"] - 1):
            self.convs.append(GraphConv(net_params["hidden_dim"],
                                        net_params["hidden_dim"],
                                        aggr=net_params["aggr"]))

        if self.net_params["batch_norm"]:
            self.norms = torch.nn.ModuleList()

            for _ in range(net_params["num_layers"] - 1):
                self.norms.append(BatchNorm(net_params["hidden_dim"]))

    def forward(self, x, edge_index, edge_weight, batch):

        for i in range(self.net_params["num_layers"] - 1):
            x = self.convs[i](x, edge_index, edge_weight)

            if self.net_params["batch_norm"]:
                x = self.norms[i](x)

            x = x.relu()
            dropout = self.net_params["conv_dropout"]
            x = F.dropout(x, p=dropout, training=self.training)

        x = self.convs[self.net_params["num_layers"] - 1](x, edge_index, edge_weight)

        if self.net_params["global_pooling"] == "add":
            readout = global_add_pool(x, batch)  # [batch_size, hidden_channels]
        elif self.net_params["global_pooling"] == "mean":
            readout = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
        elif self.net_params["global_pooling"] == "max":
            readout = global_max_pool(x, batch)  # [batch_size, hidden_channels]

        return readout  # x, readout


class HARGMIdual(torch.nn.Module):
    def __init__(self, global_encoder, win_encoder, hidden_channels, out_dim, dropout,
                 corruption="all", is_explainer=False, pre_training=False):
        super(HARGMIdual, self).__init__()
        self.global_encoder = global_encoder
        self.win_encoder = win_encoder
        self.dropout = dropout
        self.corruption = corruption
        self.is_explainer = is_explainer
        self.pre_training = pre_training

        # self.temperature = 0.05

        self.lin1 = Linear(2 * hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, out_dim)

        # self.head_pos1 = Linear(hidden_channels, 2048)
        # self.head_pos2 = Linear(hidden_channels, hidden_channels * 2)
        # self.head_neg = Linear(hidden_channels, hidden_channels * 2)

    def forward(self,
                x_s, edge_index_s, edge_weight_s, x_s_batch,
                x_t, edge_index_t, edge_weight_t, x_t_batch):
        # contrastive views global adj. matrix
        readout_pos_global = self.global_encoder(x_s, edge_index_s, edge_weight_s, x_s_batch)

        # contrastive views windows adj. matrix
        readout_pos_win = self.win_encoder(x_t, edge_index_t, edge_weight_t, x_t_batch)
        x_t_corrupted, edge_index_t_corrupted = corrupt_graph(deepcopy(x_t), deepcopy(edge_index_t), self.corruption)
        readout_neg_win = self.win_encoder(x_t_corrupted, edge_index_t_corrupted, edge_weight_t, x_t_batch)

        # head_pos1 = self.head_pos1(readout_pos_global)
        # head_pos2 = self.head_pos1(readout_pos_win)
        # head_neg = self.head_pos1(readout_neg_win)
        # contrastive_views = (head_pos1, head_pos2, head_neg)
        contrastive_views = (readout_pos_global, readout_pos_win, readout_neg_win)

        graph_emb = torch.cat((readout_pos_global, readout_pos_win), dim=1)

        x = self.lin1(graph_emb)
        x = x.relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)

        ret = x
        if self.training or self.pre_training:
            ret = (ret, contrastive_views)
        elif self.is_explainer:
            ret = F.log_softmax(x, dim=-1)

        return ret


class HARGMIsingle(torch.nn.Module):
    def __init__(self, encoder, hidden_channels, out_dim, dropout,
                 corruption="all", is_explainer=False):
        super(HARGMIsingle, self).__init__()
        self.encoder = encoder
        self.dropout = dropout
        self.corruption = corruption
        self.is_explainer = is_explainer

        self.temperature = 0.05

        self.lin1 = Linear(2 * hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, out_dim)

    def forward(self, x_s, edge_index_s, edge_weight_s, x_s_batch):
        # contrastive views global adj. matrix
        readout_pos1 = self.encoder(x_s, edge_index_s, edge_weight_s, x_s_batch)

        x_s_pos = x_s + 0.1 * torch.randn_like(x_s)
        readout_pos2 = self.encoder(x_s_pos, edge_index_s, edge_weight_s, x_s_batch)

        x_s_corrupted, edge_index_s_corrupted = corrupt_graph(deepcopy(x_s), deepcopy(edge_index_s), self.corruption)
        readout_neg = self.encoder(x_s_corrupted, edge_index_s_corrupted, edge_weight_s, x_s_batch)

        contrastive_views = (readout_pos1, readout_pos2, readout_neg)

        graph_emb = torch.cat((readout_pos1, readout_pos2), dim=1)

        x = self.lin1(graph_emb)
        x = x.relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)

        ret = x
        if self.training:
            ret = (ret, contrastive_views)
        elif self.is_explainer:
            ret = F.log_softmax(x, dim=-1)

        return ret

# ********************************* ResGCNN - Yan Yan  ****************************************


class ResGCNN_pamap2(nn.Module):
    def __init__(self, device):
        super(ResGCNN_pamap2, self).__init__()
        self.device = device
        self.conv1 = ChebConv(512, 256, 2)
        self.bn1 = GraphNorm(256)
        self.conv2 = ChebConv(256, 512, 3)
        self.bn2 = GraphNorm(512)
        self.conv3 = ChebConv(512, 256, 3)
        self.bn3 = GraphNorm(256)
        self.conv4 = ChebConv(256, 512, 2)

        self.conv5 = ChebConv(512, 256, 2)
        self.bn5 = GraphNorm(256)
        self.conv6 = ChebConv(256, 512, 3)
        self.bn6 = GraphNorm(512)
        self.conv7 = ChebConv(512, 256, 3)
        self.bn7 = GraphNorm(256)
        self.conv8 = ChebConv(256, 512, 2)

        self.conv9 = ChebConv(512, 256, 2)
        self.bn9 = GraphNorm(256)
        self.conv10 = ChebConv(256, 512, 3)
        self.bn10 = GraphNorm(512)
        self.conv11 = ChebConv(512, 256, 3)
        self.bn11 = GraphNorm(256)
        self.conv12 = ChebConv(256, 512, 2)

        self.conv13 = ChebConv(512, 256, 2)
        self.bn13 = GraphNorm(256)
        self.conv14 = ChebConv(256, 512, 3)
        self.bn14 = GraphNorm(512)
        self.conv15 = ChebConv(512, 256, 3)
        self.bn15 = GraphNorm(256)
        self.conv16 = ChebConv(256, 512, 2)

        self.linear1 = torch.nn.Linear(512, 64)
        self.linear2 = torch.nn.Linear(64, 12)   #10分类

    def forward(self, x, edge_index, edge_weight, batch):
        # x, edge_index = data.x, data.edge_index
        x1 = self.conv1(x, edge_index)
        x1 = self.bn1(x1)
        x1 = F.leaky_relu(x1, negative_slope=0.2)
        x2 = self.conv2(x1, edge_index)
        x2 = self.bn2(x2)
        x2 = F.leaky_relu(x2, negative_slope=0.2)
        x3 = self.conv3(x2, edge_index)
        x3 = self.bn3(x3)
        x3 = F.leaky_relu(x3, negative_slope=0.2)
        x4 = self.conv4(x3, edge_index)
        x4 += x
        x4 = F.relu(x4)

        x5 = self.conv5(x4, edge_index)
        x5 = self.bn5(x5)
        x5 = F.leaky_relu(x5, negative_slope=0.2)
        x6 = self.conv6(x5, edge_index)
        x6 = self.bn6(x6)
        x6 = F.leaky_relu(x6, negative_slope=0.2)
        x7 = self.conv7(x6, edge_index)
        x7 = self.bn7(x7)
        x7 = F.leaky_relu(x7, negative_slope=0.2)
        x8 = self.conv8(x7, edge_index)
        x8 += x
        x8 = F.relu(x8)

        x9 = self.conv9(x8, edge_index)
        x9 = self.bn9(x9)
        x9 = F.leaky_relu(x9, negative_slope=0.2)
        x10 = self.conv10(x9, edge_index)
        x10 = self.bn10(x10)
        x10 = F.leaky_relu(x10, negative_slope=0.2)
        x11 = self.conv11(x10, edge_index)
        x11 = self.bn11(x11)
        x11 = F.leaky_relu(x11, negative_slope=0.2)
        x12 = self.conv12(x11, edge_index)
        x12 += x
        x12 = F.relu(x12)

        x13 = self.conv13(x12, edge_index)
        x13 = self.bn13(x13)
        x13 = F.leaky_relu(x13, negative_slope=0.2)
        x14 = self.conv14(x13, edge_index)
        x14 = self.bn14(x14)
        x14 = F.leaky_relu(x14, negative_slope=0.2)
        x15 = self.conv15(x14, edge_index)
        x15 = self.bn15(x15)
        x15 = F.leaky_relu(x15, negative_slope=0.2)
        x16 = self.conv16(x15, edge_index)
        x16 += x
        x16 = F.relu(x16)

        out = global_mean_pool(x16, batch)  # 平均池化
        out = self.linear1(out)
        out = F.tanh(out)
        out = self.linear2(out)
        return out


class ResGCNN_mhealth(nn.Module):
    def __init__(self, device):
        super(ResGCNN_mhealth, self).__init__()
        self.device = device
        self.conv1 = ChebConv(128, 256, 2)
        self.bn1 = GraphNorm(256)
        self.conv2 = ChebConv(256, 512, 3)
        self.bn2 = GraphNorm(512)
        self.conv3 = ChebConv(512, 256, 3)
        self.bn3 = GraphNorm(256)
        self.conv4 = ChebConv(256, 128, 2)

        self.conv5 = ChebConv(128, 256, 2)
        self.bn5 = GraphNorm(256)
        self.conv6 = ChebConv(256, 512, 3)
        self.bn6 = GraphNorm(512)
        self.conv7 = ChebConv(512, 256, 3)
        self.bn7 = GraphNorm(256)
        self.conv8 = ChebConv(256, 128, 2)

        self.conv9 = ChebConv(128, 256, 2)
        self.bn9 = GraphNorm(256)
        self.conv10 = ChebConv(256, 512, 3)
        self.bn10 = GraphNorm(512)
        self.conv11 = ChebConv(512, 256, 3)
        self.bn11 = GraphNorm(256)
        self.conv12 = ChebConv(256, 128, 2)

        self.conv13 = ChebConv(128, 256, 2)
        self.bn13 = GraphNorm(256)
        self.conv14 = ChebConv(256, 512, 3)
        self.bn14 = GraphNorm(512)
        self.conv15 = ChebConv(512, 256, 3)
        self.bn15 = GraphNorm(256)
        self.conv16 = ChebConv(256, 128, 2)

        self.linear1 = torch.nn.Linear(128, 64)
        self.linear2 = torch.nn.Linear(64, 12)

    def forward(self, x, edge_index, edge_weight, batch):
        # x, edge_index = data.x, data.edge_index
        x1 = self.conv1(x, edge_index)
        x1 = self.bn1(x1)
        x1 = F.leaky_relu(x1, negative_slope=0.2)
        x2 = self.conv2(x1, edge_index)
        x2 = self.bn2(x2)
        x2 = F.leaky_relu(x2, negative_slope=0.2)
        x3 = self.conv3(x2, edge_index)
        x3 = self.bn3(x3)
        x3 = F.leaky_relu(x3, negative_slope=0.2)
        x4 = self.conv4(x3, edge_index)
        x4 += x
        x4 = F.relu(x4)

        x5 = self.conv5(x4, edge_index)
        x5 = self.bn5(x5)
        x5 = F.leaky_relu(x5, negative_slope=0.2)
        x6 = self.conv6(x5, edge_index)
        x6 = self.bn6(x6)
        x6 = F.leaky_relu(x6, negative_slope=0.2)
        x7 = self.conv7(x6, edge_index)
        x7 = self.bn7(x7)
        x7 = F.leaky_relu(x7, negative_slope=0.2)
        x8 = self.conv8(x7, edge_index)
        x8 += x
        x8 = F.relu(x8)

        x9 = self.conv9(x8, edge_index)
        x9 = self.bn9(x9)
        x9 = F.leaky_relu(x9, negative_slope=0.2)
        x10 = self.conv10(x9, edge_index)
        x10 = self.bn10(x10)
        x10 = F.leaky_relu(x10, negative_slope=0.2)
        x11 = self.conv11(x10, edge_index)
        x11 = self.bn11(x11)
        x11 = F.leaky_relu(x11, negative_slope=0.2)
        x12 = self.conv12(x11, edge_index)
        x12 += x
        x12 = F.relu(x12)

        x13 = self.conv13(x12, edge_index)
        x13 = self.bn13(x13)
        x13 = F.leaky_relu(x13, negative_slope=0.2)
        x14 = self.conv14(x13, edge_index)
        x14 = self.bn14(x14)
        x14 = F.leaky_relu(x14, negative_slope=0.2)
        x15 = self.conv15(x14, edge_index)
        x15 = self.bn15(x15)
        x15 = F.leaky_relu(x15, negative_slope=0.2)
        x16 = self.conv16(x15, edge_index)
        x16 += x
        x16 = F.relu(x16)

        out = global_mean_pool(x16, batch)  # 平均池化
        out = self.linear1(out)
        out = F.tanh(out)
        out = self.linear2(out)
        return out

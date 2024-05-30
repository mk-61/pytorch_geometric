# An implementation of unsupervised bipartite GraphSAGE using the Alibaba
# Taobao dataset.
import os.path as osp

import dgl.graphbolt as gb
import torch
import torch.nn.functional as F
import tqdm
from sklearn.metrics import roc_auc_score
from torch.nn import Embedding, Linear

import torch_geometric as pyg
import torch_geometric.transforms as T
from torch_geometric.datasets import Taobao
from torch_geometric.nn import SAGEConv
from torch_geometric.utils.convert import to_scipy_sparse_matrix

############################################33


class LinkPredictionTask(gb.Task):
    def __init__(self, data: pyg.data.HeteroData):
        print('Computing data splits...')
        train_data, valid_data, test_data = T.RandomLinkSplit(
            num_val=0.1,
            num_test=0.1,
            neg_sampling_ratio=1.0,
            add_negative_train_samples=False,
            edge_types=[('user', 'to', 'item')],
            rev_edge_types=[('item', 'rev_to', 'user')],
        )(data)
        self._train_set = self.make_set(train_data)
        self._valid_set = self.make_set(valid_data)
        self._test_set = self.make_set(test_data)
        print('Done!')

    @staticmethod
    def make_set(data: pyg.data.HeteroData):
        return gb.ItemSetDict({
            'user:to:item': gb.ItemSet((data['user', 'item'].edge_index.T,),
                                       names=('node_pairs',)),
            'item:rev_to:user': gb.ItemSet((data['item', 'user'].edge_index.T,),
                                           names=('node_pairs',)),
        })

    @property
    def train_set(self):
        return self._train_set

    @property
    def validation_set(self):
        return self._valid_set

    @property
    def test_set(self):
        return self._train_set


def create_sampling_graph(data: pyg.data.HeteroData) -> gb.SamplingGraph:
    ntype_offset = torch.cat(
        (torch.LongTensor([0]),
         torch.LongTensor([data[ntype].num_nodes for ntype in data.node_types]).cumsum(0)))
    edge_index_list = []
    for src, et, dst in data.edge_types:
        src_i, dst_i = (data.node_types.index(role) for role in (src, dst))
        edge_index = data[src, et, dst].edge_index
        edge_index_list.append(torch.stack((edge_index[0] + ntype_offset[src_i],
                                            edge_index[1] + ntype_offset[dst_i])))
    csc = to_scipy_sparse_matrix(torch.cat(edge_index_list, dim=1)).tocsc()
    type_per_edge = torch.cat([torch.full((edge_index.size(1),), i)
                               for i, t in enumerate(edge_index_list)])
    node_type_to_id = {t:i for i, t in enumerate(data.node_types)}
    edge_type_to_id = {':'.join(t):i for i, t in enumerate(data.edge_types)}
    return gb.fused_csc_sampling_graph(torch.from_numpy(csc.indptr).to(torch.long),
                                       torch.from_numpy(csc.indices).to(torch.long),
                                       ntype_offset,
                                       type_per_edge=type_per_edge,
                                       node_type_to_id=node_type_to_id,
                                       edge_type_to_id=edge_type_to_id)


def preprocess_taobao(data: pyg.data.HeteroData) -> pyg.data.HeteroData:
    data['user'].x = torch.arange(0, data['user'].num_nodes)
    data['item'].x = torch.arange(0, data['item'].num_nodes)

    # Only consider user<>item relationships for simplicity:
    del data['category']
    del data['item', 'category']
    del data['user', 'item'].time
    del data['user', 'item'].behavior

    for src, et, dst in data.edge_types:
        edge_index = data[src, et, dst].edge_index
        print(f'Deduplicating {edge_index.size(1)} {(src, et, dst)} edges...')
        edge_index = edge_index.to(device).unique(dim=1).to('cpu')
        print(f'Found {edge_index.size(1)} unique')
        data[src, et, dst].edge_index = edge_index

    # Compute sparsified item<>item relationships through users:
    print('Computing item<>item relationships...')
    mat = to_scipy_sparse_matrix(data['user', 'item'].edge_index).tocsr()
    mat = mat[:data['user'].num_nodes, :data['item'].num_nodes]
    comat = mat.T @ mat
    comat.setdiag(0)
    comat = comat >= 3.
    comat = comat.tocoo()
    row = torch.from_numpy(comat.row).to(torch.long)
    col = torch.from_numpy(comat.col).to(torch.long)
    item_to_item_edge_index = torch.stack([row, col], dim=0)

    # Add the generated item<>item relationships for high-order information:
    data['item', 'item'].edge_index = item_to_item_edge_index
    return data


class TaobaoDataset(gb.Dataset):
    def __init__(self):
        super().__init__()
        data = Taobao(osp.join(osp.dirname(osp.realpath(__file__)), '../../data/Taobao'),
                      pre_transform=preprocess_taobao, transform=T.ToUndirected())[0]
        self._tasks = [LinkPredictionTask(data)]
        self._graph = create_sampling_graph(data)

    @property
    def tasks(self):
        return self._tasks

    @property
    def graph(self):
        return self._graph


############################################33

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
dataset = TaobaoDataset()
train_set = dataset.tasks[0].train_set
valid_set = dataset.tasks[0].validation_set
test_set = dataset.tasks[0].test_set


def create_dataloader(itemset):
    is_train = itemset is train_set
    datapipe = gb.ItemSampler(itemset, batch_size=2048, shuffle=is_train, drop_last=is_train)
    # datapipe = datapipe.copy_to(device)
    if is_train:
        datapipe = datapipe.sample_uniform_negative(dataset.graph, negative_ratio=1)
        datapipe = datapipe.sample_neighbor(dataset.graph, fanouts=[8, 4])
        datapipe = datapipe.transform(gb.exclude_seed_edges)
    else:
        datapipe = datapipe.sample_neighbor(dataset.graph, fanouts=[8, 4])
    return gb.DataLoader(datapipe, num_workers=0)

train_loader = create_dataloader(train_set)
valid_loader = create_dataloader(valid_set)
test_loader = create_dataloader(test_set)


class ItemGNNEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(-1, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        return self.lin(x)


class UserGNNEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), hidden_channels)
        self.conv3 = SAGEConv((-1, -1), hidden_channels)
        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        item_x = self.conv1(
            x_dict['item'],
            edge_index_dict[('item', 'to', 'item')],
        ).relu()

        user_x = self.conv2(
            (x_dict['item'], x_dict['user']),
            edge_index_dict[('item', 'rev_to', 'user')],
        ).relu()

        user_x = self.conv3(
            (item_x, user_x),
            edge_index_dict[('item', 'rev_to', 'user')],
        ).relu()

        return self.lin(user_x)


class EdgeDecoder(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.lin1 = Linear(2 * hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, 1)

    def forward(self, z_src, z_dst, edge_label_index):
        row, col = edge_label_index
        z = torch.cat([z_src[row], z_dst[col]], dim=-1)

        z = self.lin1(z).relu()
        z = self.lin2(z)
        return z.view(-1)


class Model(torch.nn.Module):
    def __init__(self, num_users, num_items, hidden_channels, out_channels):
        super().__init__()
        self.user_emb = Embedding(num_users, hidden_channels, device=device)
        self.item_emb = Embedding(num_items, hidden_channels, device=device)
        self.item_encoder = ItemGNNEncoder(hidden_channels, out_channels)
        self.user_encoder = UserGNNEncoder(hidden_channels, out_channels)
        self.decoder = EdgeDecoder(out_channels)

    def forward(self, x_dict, edge_index_dict, edge_label_index):
        z_dict = {}
        x_dict['user'] = self.user_emb(x_dict['user'])
        x_dict['item'] = self.item_emb(x_dict['item'])
        z_dict['item'] = self.item_encoder(
            x_dict['item'],
            edge_index_dict[('item', 'to', 'item')],
        )
        z_dict['user'] = self.user_encoder(x_dict, edge_index_dict)

        return self.decoder(z_dict['user'], z_dict['item'], edge_label_index)


model = Model(
    num_users=dataset.graph.num_nodes['user'],
    num_items=dataset.graph.num_nodes['item'],
    hidden_channels=64,
    out_channels=64,
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def train():
    model.train()

    total_loss = total_examples = 0
    # for batch in tqdm.tqdm(train_loader):
    print('Start iterating')
    for batch in train_loader:
        import IPython; IPython.embed()
        batch = batch.to(device)
        optimizer.zero_grad()

        pred = model(
            batch.x_dict,
            batch.edge_index_dict,
            batch['user', 'item'].edge_label_index,
        )
        loss = F.binary_cross_entropy_with_logits(
            pred, batch['user', 'item'].edge_label)

        loss.backward()
        optimizer.step()
        total_loss += float(loss)
        total_examples += pred.numel()

    return total_loss / total_examples


@torch.no_grad()
def test(loader):
    model.eval()

    preds, targets = [], []
    for batch in tqdm.tqdm(loader):
        batch = batch.to(device)

        pred = model(
            batch.x_dict,
            batch.edge_index_dict,
            batch['user', 'item'].edge_label_index,
        ).sigmoid().view(-1).cpu()
        target = batch['user', 'item'].edge_label.long().cpu()

        preds.append(pred)
        targets.append(target)

    pred = torch.cat(preds, dim=0).numpy()
    target = torch.cat(targets, dim=0).numpy()

    return roc_auc_score(target, pred)


for epoch in range(1, 21):
    loss = train()
    val_auc = test(val_loader)
    test_auc = test(test_loader)

    print(f'Epoch: {epoch:02d}, Loss: {loss:4f}, Val: {val_auc:.4f}, '
          f'Test: {test_auc:.4f}')

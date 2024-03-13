import os
from typing import Optional

import torch
import torch.nn.functional as F
from ogb.nodeproppred import PygNodePropPredDataset

from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import GCNConv

from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = PygNodePropPredDataset(name='ogbn-papers100M')
split_idx = dataset.get_idx_split()


def get_num_workers() -> int:
    try:
        return len(os.sched_getaffinity(0)) // 2
    except Exception:
        return os.cpu_count() // 2


kwargs = dict(
    data=dataset[0],
    num_neighbors=[50, 50],
    batch_size=128,
    num_workers=get_num_workers(),
)
train_loader = NeighborLoader(input_nodes=split_idx['train'], shuffle=True,
                              **kwargs)
val_loader = NeighborLoader(input_nodes=split_idx['valid'], **kwargs)
test_loader = NeighborLoader(input_nodes=split_idx['test'], **kwargs)


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x


model = GCN(dataset.num_features, 64, dataset.num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0005)


def train(desc):
    model.train()
    total_loss = 0
    total_correct = 0
    total = split_idx['train'].size(0)

    pbar = tqdm(desc=desc, total=total)

    for i, batch in enumerate(train_loader):
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index)[:batch.batch_size]
        y = batch.y[:batch.batch_size].view(-1).to(torch.long)
        loss = F.cross_entropy(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_correct += out.argmax(dim=-1).eq(y).sum().item()

        pbar.update(batch.batch_size)

    pbar.close()

    avg_loss = total_loss / len(train_loader)
    avg_accuracy = total_correct / total
    return avg_loss, avg_accuracy


@torch.no_grad()
def test(loader: NeighborLoader, val_steps: Optional[int] = None):
    model.eval()

    total_correct = total_examples = 0
    for i, batch in enumerate(loader):
        if val_steps is not None and i >= val_steps:
            break

        batch = batch.to(device)
        out = model(batch.x, batch.edge_index)[:batch.batch_size]
        pred = out.argmax(dim=-1)
        y = batch.y[:batch.batch_size].view(-1).to(torch.long)

        total_correct += int((pred == y).sum())
        total_examples += y.size(0)

    return total_correct / total_examples


for epoch in range(1, 4):
    train_loss, train_acc = train(f'Epoch {epoch} train')
    val_acc = test(val_loader, val_steps=100)
    print(
        f"Epoch {epoch}, Train Loss: {train_loss:.4f}, "
        f"Train Accuracy: {train_acc:.4f}, "
        f"Valid Accuracy: {val_acc:.4f}"
    )

test_acc = test(test_loader)
print(f'Test Acc: {test_acc:.4f}')

import dgl.graphbolt as gb
import torch
import torch.nn.functional as F
import torchmetrics.functional as MF
from torch_geometric.nn import GCNConv
from tqdm import tqdm


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


def create_dataloader(dataset_set, graph, feature, batch_size, fanout, device, job):
    # Initialize an ItemSampler to sample mini-batches from the dataset.
    datapipe = gb.ItemSampler(
        dataset_set,
        batch_size=batch_size,
        shuffle=(job == "train"),
        drop_last=(job == "train"),
    )
    # Sample neighbors for each node in the mini-batch.
    datapipe = datapipe.sample_neighbor(graph, fanout)
    # Copy the data to the specified device.
    datapipe = datapipe.copy_to(device=device, extra_attrs=["input_nodes"])
    # Fetch node features for the sampled subgraph.
    datapipe = datapipe.fetch_feature(feature, node_feature_keys=["feat"])
    # Create and return a DataLoader to handle data loading.
    dataloader = gb.DataLoader(datapipe, num_workers=0)

    return dataloader


def train(model, dataloader, optimizer, desc, set_size):
    model.train()
    total_loss = 0
    total_correct = 0
    num_batches = 0

    pbar = tqdm(desc=desc, total=set_size)

    for minibatch in dataloader:
        pyg_data = minibatch.to_pyg_data()

        optimizer.zero_grad()
        out = model(pyg_data.x, pyg_data.edge_index)[:pyg_data.y.shape[0]]
        y = pyg_data.y
        loss = F.cross_entropy(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_correct += out.argmax(dim=-1).eq(y).sum().item()
        num_batches += 1

        pbar.update(pyg_data.batch_size)

    pbar.close()

    avg_loss = total_loss / num_batches
    avg_accuracy = total_correct / set_size
    return avg_loss, avg_accuracy


@torch.no_grad()
def evaluate(model, dataloader, num_classes, desc, set_size):
    model.eval()
    y_hats = []
    ys = []

    pbar = tqdm(desc=desc, total=set_size)

    for minibatch in dataloader:
        pyg_data = minibatch.to_pyg_data()
        out = model(pyg_data.x, pyg_data.edge_index)[:pyg_data.y.shape[0]]
        y_hats.append(out)
        ys.append(pyg_data.y)

        pbar.update(pyg_data.batch_size)

    pbar.close()

    return MF.accuracy(
        torch.cat(y_hats),
        torch.cat(ys),
        task="multiclass",
        num_classes=num_classes,
    )


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = gb.BuiltinDataset('ogbn-papers100M').load()
feature = dataset.feature.pin_memory_()
train_set = dataset.tasks[0].train_set
valid_set = dataset.tasks[0].validation_set
test_set = dataset.tasks[0].test_set
num_classes = dataset.tasks[0].metadata["num_classes"]

train_dataloader = create_dataloader(
    train_set,
    dataset.graph,
    feature,
    128,
    [50, 50],
    device,
    job="train",
)
valid_dataloader = create_dataloader(
    valid_set,
    dataset.graph,
    feature,
    512,
    [-1],
    device,
    job="valid",
)
test_dataloader = create_dataloader(
    test_set,
    dataset.graph,
    feature,
    512,
    [-1],
    device,
    job="test",
)
in_channels = feature.size("node", None, "feat")[0]
hidden_channels = 64
model = GCN(in_channels, hidden_channels, num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0005)
for epoch in range(4):
    train_loss, train_accuracy = train(model, train_dataloader, optimizer,
            f'Epoch {epoch} train', len(train_set))
    valid_accuracy = evaluate(model, valid_dataloader, num_classes,
            f'Epoch {epoch} valid', len(valid_set))
    print(
        f"Epoch {epoch}, Train Loss: {train_loss:.4f}, "
        f"Train Accuracy: {train_accuracy:.4f}, "
        f"Valid Accuracy: {valid_accuracy:.4f}"
    )
test_accuracy = evaluate(model, test_dataloader, num_classes, 'Test', len(test_set))
print(f"Test Accuracy: {test_accuracy:.4f}")

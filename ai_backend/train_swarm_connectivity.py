"""Connectivity-prediction training: 3 methods mirroring the Graph & Network
Analysis coursework's own structure (`D:\\ΠΜΣ\\Γράφοι και Ανάλυση Δικτύων\\
εξαμηνιαια εργασια\\GNA-ΤΣΙΛΙΜΠΩΚΟΣ-25118\\GNA.ipynb`), applied to
swarm_network_generator.py's synthetic connectivity-fragmentation task
instead of that coursework's ENZYMES molecule-classification task:

  Method A: hand-engineered graph features (networkx, mirrors that
            notebook's extract_graph_features(), same feature families:
            degree stats, density, clustering/transitivity, components,
            LCC diameter/avg-path, assortativity, betweenness/closeness
            centrality) + this task's own physical aggregates -> MLP.
  Method B: biased-random-walk Node2Vec (mirrors node2vec_biased_walks()/
            get_graph_embedding(), same hand-rolled walk + gensim
            Word2Vec skip-gram, not the node2vec package) -> pooled
            graph embedding -> MLP.
  Method C: GCN / GIN graph classification (mirrors GCN_Classifier/
            GIN_Classifier exactly: BatchNorm1d + dropout stack, global
            pooling), using this task's own 6-dim node features (position,
            velocity, uncertainty, elapsed loss time) instead of ENZYMES'
            node labels.

Same multi-seed rigor as every other experiment in this project
(confidence_interval()/set_global_seed() reused from train_px4_synthetic.py).

Usage:
  python ai_backend/train_swarm_connectivity.py
"""
import os
import pickle
import random
import sys

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_px4_synthetic import set_global_seed, confidence_interval

_DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'runs', 'swarm_connectivity',
                           'swarm_scenarios.pkl')
_OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'runs', 'swarm_connectivity')
_FIG_DIR = os.path.join(_OUT_DIR, 'figures')

N_SEEDS = 3


def _load_split(splits, name):
    scenarios, labels = splits[name]
    graphs = [nx.from_numpy_array(adj) for _, adj in scenarios]
    feats = [f for f, _ in scenarios]
    return graphs, feats, labels


# --------------------------------------------------------------------------
# Method A: hand-engineered graph features (mirrors extract_graph_features())
# --------------------------------------------------------------------------
def extract_graph_features(G: nx.Graph, node_feats: np.ndarray) -> np.ndarray:
    """15 topological features, same families as the coursework's own
    extract_graph_features(), plus 6 physically-meaningful aggregates this
    task's node features actually carry (uncertainty/velocity/elapsed
    loss time) that a pure adjacency-matrix routine has no way to see."""
    G = G.copy()
    G.remove_edges_from(nx.selfloop_edges(G))
    n = G.number_of_nodes()
    m = G.number_of_edges()

    degrees = [d for _, d in G.degree()]
    avg_degree = float(np.mean(degrees)) if n > 0 else 0.0
    std_degree = float(np.std(degrees)) if n > 0 else 0.0
    max_degree = float(max(degrees)) if n > 0 else 0.0
    density = nx.density(G)
    avg_clustering = nx.average_clustering(G) if n > 2 else 0.0
    transitivity = nx.transitivity(G)

    components = list(nx.connected_components(G))
    num_comp = len(components)
    largest_cc = max(components, key=len) if components else set()
    G_lcc = G.subgraph(largest_cc).copy()
    if len(G_lcc) > 1 and nx.is_connected(G_lcc):
        try:
            lcc_diameter = nx.diameter(G_lcc)
            lcc_avg_sp = nx.average_shortest_path_length(G_lcc)
        except Exception:
            lcc_diameter, lcc_avg_sp = 0.0, 0.0
    else:
        lcc_diameter, lcc_avg_sp = 0.0, 0.0
    lcc_frac = len(largest_cc) / n if n > 0 else 0.0

    try:
        assortativity = nx.degree_assortativity_coefficient(G) if m > 1 else 0.0
    except Exception:
        assortativity = 0.0
    if np.isnan(assortativity):
        assortativity = 0.0

    if n > 2:
        bc = nx.betweenness_centrality(G, normalized=True)
        avg_betweenness = float(np.mean(list(bc.values())))
        cc_ = nx.closeness_centrality(G)
        avg_closeness = float(np.mean(list(cc_.values())))
        # Base station (node 0) is the operationally relevant node -- its
        # own centrality, not just the graph average, is directly relevant
        # to "can drones relay back to base."
        base_betweenness = float(bc.get(0, 0.0))
        base_closeness = float(cc_.get(0, 0.0))
    else:
        avg_betweenness = avg_closeness = base_betweenness = base_closeness = 0.0

    topo = np.array([
        n, m, avg_degree, std_degree, max_degree, density, avg_clustering,
        transitivity, num_comp, lcc_diameter, lcc_avg_sp, lcc_frac,
        assortativity, avg_betweenness, avg_closeness, base_betweenness, base_closeness,
    ], dtype=np.float32)

    drone_feats = node_feats[1:]  # exclude the base station row
    physical = np.array([
        drone_feats[:, 4].mean(), drone_feats[:, 4].std(),   # uncertainty mean/std
        drone_feats[:, 5].mean(), drone_feats[:, 5].std(),   # elapsed-loss-frac mean/std
        np.linalg.norm(drone_feats[:, 2:4], axis=1).mean(),  # mean speed
        np.linalg.norm(drone_feats[:, 2:4], axis=1).std(),   # speed std
    ], dtype=np.float32)

    return np.concatenate([topo, physical])


def run_method_a(train, val, test, seed, return_model=False):
    from sklearn.neural_network import MLPClassifier
    set_global_seed(seed)
    Gtr, Ftr, ytr = train
    Gva, Fva, yva = val
    Gte, Fte, yte = test

    Xtr = np.array([extract_graph_features(g, f) for g, f in zip(Gtr, Ftr)])
    Xva = np.array([extract_graph_features(g, f) for g, f in zip(Gva, Fva)])
    Xte = np.array([extract_graph_features(g, f) for g, f in zip(Gte, Fte)])

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(Xtr)
    Xtr, Xva, Xte = scaler.transform(Xtr), scaler.transform(Xva), scaler.transform(Xte)

    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3, max_iter=500,
                         early_stopping=True, random_state=seed)
    mlp.fit(Xtr, ytr)
    from sklearn.metrics import f1_score, confusion_matrix
    val_f1 = f1_score(yva, mlp.predict(Xva), average='macro')
    test_pred = mlp.predict(Xte)
    test_f1 = f1_score(yte, test_pred, average='macro')
    cm = confusion_matrix(yte, test_pred)
    if return_model:
        return val_f1, test_f1, cm, (scaler, mlp)
    return val_f1, test_f1, cm


# --------------------------------------------------------------------------
# Method B: Node2Vec (mirrors node2vec_biased_walks / get_graph_embedding)
# --------------------------------------------------------------------------
def node2vec_biased_walks(G, num_walks=5, walk_length=15, p=1.0, q=1.0, rng=None):
    rng = rng or random
    nodes = list(G.nodes())
    walks = []
    for _ in range(num_walks):
        rng.shuffle(nodes)
        for start in nodes:
            walk = [start]
            while len(walk) < walk_length:
                cur = walk[-1]
                neighbors = list(G.neighbors(cur))
                if not neighbors:
                    break
                if len(walk) == 1:
                    walk.append(rng.choice(neighbors))
                else:
                    prev = walk[-2]
                    probs = []
                    for nxt in neighbors:
                        if nxt == prev:
                            probs.append(1.0 / p)
                        elif G.has_edge(prev, nxt):
                            probs.append(1.0)
                        else:
                            probs.append(1.0 / q)
                    total = sum(probs)
                    probs = [pr / total for pr in probs]
                    nxt = np.random.choice(neighbors, p=probs)
                    walk.append(nxt)
            walks.append([str(v) for v in walk])
    return walks


def get_graph_embedding(G, embedding_dim=32, num_walks=5, walk_length=15,
                         p=1.0, q=1.0, window=5, seed=0, pooling='mean_max'):
    from gensim.models import Word2Vec
    G = G.copy()
    G.remove_edges_from(nx.selfloop_edges(G))
    dim = embedding_dim * 2 if pooling == 'mean_max' else embedding_dim
    if G.number_of_nodes() < 2 or G.number_of_edges() < 1:
        return np.zeros(dim, dtype=np.float32)

    walks = node2vec_biased_walks(G, num_walks=num_walks, walk_length=walk_length, p=p, q=q)
    model = Word2Vec(sentences=walks, vector_size=embedding_dim, window=window,
                      min_count=0, sg=1, hs=0, negative=5, workers=1, seed=seed, epochs=10)
    node_ids = [str(n) for n in G.nodes()]
    Z = np.array([model.wv[nid] for nid in node_ids])
    if pooling == 'mean':
        return Z.mean(axis=0)
    if pooling == 'max':
        return Z.max(axis=0)
    return np.concatenate([Z.mean(axis=0), Z.max(axis=0)])


def run_method_b(train, val, test, seed):
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import f1_score, confusion_matrix
    set_global_seed(seed)
    Gtr, _, ytr = train
    Gva, _, yva = val
    Gte, _, yte = test

    Xtr = np.array([get_graph_embedding(g, seed=seed) for g in Gtr])
    Xva = np.array([get_graph_embedding(g, seed=seed) for g in Gva])
    Xte = np.array([get_graph_embedding(g, seed=seed) for g in Gte])

    scaler = StandardScaler().fit(Xtr)
    Xtr, Xva, Xte = scaler.transform(Xtr), scaler.transform(Xva), scaler.transform(Xte)

    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3, max_iter=500,
                         early_stopping=True, random_state=seed)
    mlp.fit(Xtr, ytr)
    val_f1 = f1_score(yva, mlp.predict(Xva), average='macro')
    test_pred = mlp.predict(Xte)
    test_f1 = f1_score(yte, test_pred, average='macro')
    cm = confusion_matrix(yte, test_pred)
    return val_f1, test_f1, cm


# --------------------------------------------------------------------------
# Method C: GCN / GIN (mirrors GCN_Classifier / GIN_Classifier)
# --------------------------------------------------------------------------
def _build_pyg_dataset(graphs, feats, labels):
    import torch
    from torch_geometric.data import Data
    data_list = []
    for G, f, y in zip(graphs, feats, labels):
        edge_index = torch.tensor(list(G.edges()), dtype=torch.long).t().contiguous()
        if edge_index.numel() == 0:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
        else:
            edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)  # undirected
        x = torch.tensor(f, dtype=torch.float32)
        data_list.append(Data(x=x, edge_index=edge_index, y=torch.tensor([y], dtype=torch.long)))
    return data_list


def _make_gnn_classes():
    import torch
    from torch.nn import ModuleList, BatchNorm1d, Linear, Sequential, ReLU
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv, GINConv, global_mean_pool, global_add_pool

    class GCN_Classifier(torch.nn.Module):
        def __init__(self, in_channels, hidden_channels, num_layers, num_classes, dropout=0.5):
            super().__init__()
            self.convs = ModuleList()
            self.bns = ModuleList()
            self.dropout = dropout
            for i in range(num_layers):
                ic = in_channels if i == 0 else hidden_channels
                self.convs.append(GCNConv(ic, hidden_channels))
                self.bns.append(BatchNorm1d(hidden_channels))
            self.lin1 = Linear(hidden_channels, hidden_channels)
            self.lin2 = Linear(hidden_channels, num_classes)

        def forward(self, x, edge_index, batch):
            for conv, bn in zip(self.convs, self.bns):
                x = conv(x, edge_index)
                x = bn(x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            x = global_mean_pool(x, batch)
            x = F.relu(self.lin1(x))
            x = F.dropout(x, p=self.dropout, training=self.training)
            return self.lin2(x)

    class GIN_Classifier(torch.nn.Module):
        def __init__(self, in_channels, hidden_channels, num_layers, num_classes, dropout=0.5):
            super().__init__()
            self.convs = ModuleList()
            self.bns = ModuleList()
            self.dropout = dropout
            for i in range(num_layers):
                ic = in_channels if i == 0 else hidden_channels
                mlp = Sequential(Linear(ic, hidden_channels), BatchNorm1d(hidden_channels), ReLU(),
                                  Linear(hidden_channels, hidden_channels), ReLU())
                self.convs.append(GINConv(mlp, train_eps=True))
                self.bns.append(BatchNorm1d(hidden_channels))
            self.lin1 = Linear(num_layers * hidden_channels, hidden_channels)
            self.lin2 = Linear(hidden_channels, num_classes)

        def forward(self, x, edge_index, batch):
            xs = []
            for conv, bn in zip(self.convs, self.bns):
                x = conv(x, edge_index)
                x = bn(x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                xs.append(global_add_pool(x, batch))
            x = torch.cat(xs, dim=1)
            x = F.relu(self.lin1(x))
            x = F.dropout(x, p=self.dropout, training=self.training)
            return self.lin2(x)

    return GCN_Classifier, GIN_Classifier


def run_method_c(train, val, test, seed, arch='GCN', hidden=64, layers=2, dropout=0.5, epochs=60):
    import torch
    from torch_geometric.loader import DataLoader
    from sklearn.metrics import f1_score, confusion_matrix
    set_global_seed(seed)
    torch.manual_seed(seed)

    Gtr, Ftr, ytr = train
    Gva, Fva, yva = val
    Gte, Fte, yte = test
    train_data = _build_pyg_dataset(Gtr, Ftr, ytr)
    val_data = _build_pyg_dataset(Gva, Fva, yva)
    test_data = _build_pyg_dataset(Gte, Fte, yte)

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

    GCN_Classifier, GIN_Classifier = _make_gnn_classes()
    cls = GCN_Classifier if arch == 'GCN' else GIN_Classifier
    model = cls(in_channels=Ftr[0].shape[1], hidden_channels=hidden, num_layers=layers,
                num_classes=2, dropout=dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    def _eval(loader):
        model.eval()
        preds, ys = [], []
        with torch.no_grad():
            for batch in loader:
                out = model(batch.x, batch.edge_index, batch.batch)
                preds.append(out.argmax(dim=1))
                ys.append(batch.y)
        preds = torch.cat(preds).numpy()
        ys = torch.cat(ys).numpy()
        return f1_score(ys, preds, average='macro'), preds, ys

    best_val_f1, best_state = -1.0, None
    for _ in range(epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
        val_f1, _, _ = _eval(val_loader)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    test_f1, test_pred, test_y = _eval(test_loader)
    cm = confusion_matrix(test_y, test_pred)
    return best_val_f1, test_f1, cm


# --------------------------------------------------------------------------
def _plot_comparison(results, out_path):
    from plot_style import COLOR_ACCENT_BLUE, COLOR_ACCENT_ORANGE, COLOR_ACCENT_GREEN, COLOR_ACCENT_PURPLE
    labels = list(results.keys())
    means = [results[k][0] for k in labels]
    los = [results[k][1] for k in labels]
    his = [results[k][2] for k in labels]
    errs = [[m - lo for m, lo in zip(means, los)], [hi - m for m, hi in zip(means, his)]]

    # Fixed categorical order, never cycled (dataviz convention) -- one hue
    # per method family, GCN/GIN sharing the "Method C" story get adjacent
    # but visually distinct hues rather than repeating Method A's blue.
    fixed_colors = [COLOR_ACCENT_BLUE, COLOR_ACCENT_ORANGE, COLOR_ACCENT_GREEN, COLOR_ACCENT_PURPLE]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.bar(labels, means, yerr=errs, capsize=4, color=fixed_colors[:len(labels)])
    ax.set_ylabel('Test Macro-F1')
    ax.set_title('Swarm connectivity-fragmentation prediction: method comparison')
    ax.set_ylim(0, 1.0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, color='#e0e0e0', linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, facecolor='white')
    plt.close(fig)
    print(f"Saved -> {out_path}")


def _plot_confusion(cm, title, out_path):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap='Blues')
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                     color='white' if cm[i, j] > cm.max() / 2 else 'black')
    ax.set_xticks([0, 1]); ax.set_xticklabels(['stays connected', 'fragments'])
    ax.set_yticks([0, 1]); ax.set_yticklabels(['stays connected', 'fragments'])
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    os.makedirs(_FIG_DIR, exist_ok=True)

    with open(_DATA_PATH, 'rb') as f:
        splits = pickle.load(f)
    train = _load_split(splits, 'train')
    val = _load_split(splits, 'val')
    test = _load_split(splits, 'test')
    print(f"Loaded: train={len(train[2])}, val={len(val[2])}, test={len(test[2])}")

    results = {}
    cms = {}
    best_method_a = (-1.0, None)   # (val_f1, (scaler, mlp)) -- the live swarm demo loads this back

    for label, fn, kwargs in [
        ('Method A\n(graph features)', run_method_a, {}),
        ('Method B\n(Node2Vec)', run_method_b, {}),
        ('Method C\n(GCN)', run_method_c, {'arch': 'GCN'}),
        ('Method C\n(GIN)', run_method_c, {'arch': 'GIN'}),
    ]:
        test_f1s = []
        last_cm = None
        print(f"\n{'=' * 60}\n{label.replace(chr(10), ' ')}\n{'=' * 60}")
        for seed in range(N_SEEDS):
            if fn is run_method_a:
                val_f1, test_f1, cm, model_artifacts = fn(train, val, test, seed, return_model=True)
                if val_f1 > best_method_a[0]:
                    best_method_a = (val_f1, model_artifacts)
            else:
                val_f1, test_f1, cm = fn(train, val, test, seed, **kwargs) if kwargs else fn(train, val, test, seed)
            print(f"  seed {seed}: val_f1={val_f1:.3f}  test_f1={test_f1:.3f}")
            test_f1s.append(test_f1)
            last_cm = cm
        mean, lo, hi = confidence_interval(test_f1s)
        print(f"  -> mean test Macro-F1 = {mean:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")
        results[label] = (mean, lo, hi, test_f1s)
        cms[label] = last_cm
        slug = label.replace(chr(10), ' ').replace('(', '').replace(')', '').replace(' ', '_')
        _plot_confusion(last_cm, f"{label.replace(chr(10), ' ')} (seed 2)",
                         os.path.join(_FIG_DIR, f"confusion_{slug}.png"))

    import json
    with open(os.path.join(_OUT_DIR, 'comparison.json'), 'w') as f:
        json.dump({k: {'mean': v[0], 'ci_lo': v[1], 'ci_hi': v[2], 'per_seed': v[3]}
                   for k, v in results.items()}, f, indent=2)

    _plot_comparison({k: v[:3] for k, v in results.items()},
                      os.path.join(_FIG_DIR, 'method_comparison.png'))

    # Method A won on Macro-F1 and needs no torch/gensim at inference time --
    # persisted for live_swarm_demo.py to load directly for real-time
    # fragmentation-risk prediction, best-of-3-seeds by validation Macro-F1.
    if best_method_a[1] is not None:
        model_out = os.path.join(_OUT_DIR, 'method_a_model.pkl')
        with open(model_out, 'wb') as f:
            pickle.dump({'scaler': best_method_a[1][0], 'mlp': best_method_a[1][1],
                         'val_f1': best_method_a[0]}, f)
        print(f"Saved -> {model_out} (val_f1={best_method_a[0]:.3f})")

    print("\n" + "=" * 60)
    print("Summary (test Macro-F1, mean [95% CI]):")
    for k, (mean, lo, hi, _) in results.items():
        print(f"  {k.replace(chr(10), ' '):<28} {mean:.3f}  [{lo:.3f}, {hi:.3f}]")

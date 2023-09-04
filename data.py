import torch
import random
import numpy as np
import networkx as nx
from torch.utils.data import DataLoader

def generate_input(n, p, seed) -> tuple[list[tuple[int, int]], tuple[int, int], bool]:
    graph = nx.erdos_renyi_graph(n, p, seed)
    random.seed(seed)
    start, end = random.sample(range(n), 2)
    return (graph.edges(), (start, end), nx.has_path(graph, start, end))

def graph_to_sequence(graph):
    edges, (start, end), has_path = graph
    sequence = []
    for (a, b) in edges:
        sequence.append(a+1)
        sequence.append(b+1)
        sequence.append(0)
    sequence.append(start+1)
    sequence.append(end+1)
    return (sequence, int(has_path))

def measure_overlap(train_graphs, eval_graphs):
    train_sequences = [str(graph_to_sequence(graph)[0]) for graph in train_graphs]
    eval_sequences = [str(graph_to_sequence(graph)[0]) for graph in eval_graphs]
    return len(set(train_sequences).intersection(set(eval_sequences))) / len(set(eval_sequences))
    
def collate_fn(graphs, device="cuda"):
    xs = []
    ys = []
    for graph in graphs:
        x, y = graph_to_sequence(graph)
        xs.append(x)
        ys.append(y)
    padding_value = 200
    padded_xs = torch.nn.utils.rnn.pad_sequence([torch.tensor(x) for x in xs], batch_first=True, padding_value=padding_value)
    attention_mask = (padded_xs != padding_value).type(torch.long)
    result = {"input_ids": padded_xs, "attention_mask": attention_mask, "labels": torch.tensor(ys)}
    if device == "cuda":
        result = {k: v.cuda() for k, v in result.items()}
    return result
    
def make_dataloaders(args):
    train_graphs, eval_graphs = [], []
    for seed in range(args.n_train):
        train_graphs.append(generate_input(args.n_nodes, args.p_edge, seed))
    for seed in range(args.n_train, args.n_train+args.n_eval):
        eval_graphs.append(generate_input(args.n_nodes, args.p_edge, seed))
    print(f"train/eval overlap: {measure_overlap(train_graphs, eval_graphs)}")

    train_dataloader = DataLoader(train_graphs, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn)
    eval_dataloader = DataLoader(eval_graphs, batch_size=args.eval_batch_size, shuffle=True, collate_fn=collate_fn)
    return train_dataloader, eval_dataloader
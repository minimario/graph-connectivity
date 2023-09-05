import json
import os
from tqdm import tqdm
import argparse
from data import make_dataloaders, make_eval_dataloader, make_all_pairs_dataloader 
import torch
from transformers import BertConfig, BertForSequenceClassification
from transformers import get_scheduler
import wandb

def calc_accuracy(preds, labels, graph_ids, strict=True):
    if not strict:
        accuracy = (preds == labels).type(torch.float).mean()
        return accuracy.item()
    else:
        n, _, _ = graph_ids[0]
        samples_per_graph = n*(n-1)//2
        num_graphs = len(graph_ids) // samples_per_graph
        accuracy = (preds == labels)
        count = 0
        for i in range(num_graphs):
            if all(accuracy[i*samples_per_graph:(i+1)*samples_per_graph]):
                count += 1
        return count / num_graphs

def eval(model, eval_dataloader, save_dir=None):
    model.eval()
    all_preds, all_labels, all_graph_ids = [], [], []
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(eval_dataloader):
            outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])
            graph_ids = batch["graph_ids"]
            total_loss += outputs.loss
            all_graph_ids += graph_ids
            all_preds.append(torch.argmax(outputs.logits, dim=1))
            all_labels.append(batch["labels"])

    all_preds, all_labels = torch.hstack(all_preds), torch.hstack(all_labels)
    nonstrict_accuracy = calc_accuracy(all_preds, all_labels, all_graph_ids, strict=False)
    strict_accuracy = calc_accuracy(all_preds, all_labels, all_graph_ids, strict=True)
    loss = total_loss.item() / len(eval_dataloader)
    # wandb.log({"eval_loss": loss})
    # wandb.log({"eval_accuracy_strict": strict_accuracy})
    # wandb.log({"eval_accuracy_nonstrict": nonstrict_accuracy})
    print(f"eval loss: {loss} | eval_accuracy (nonstrict): {nonstrict_accuracy} | eval_accuracy (strict): {strict_accuracy}")
    if save_dir:
        save_data = {
            "preds": all_preds.tolist(),
            "labels": all_labels.tolist(),
            "graph_ids": all_graph_ids,
            "nonstrict_accuracy": nonstrict_accuracy,
            "strict_accuracy": strict_accuracy
        }
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        save_path = os.path.join(args.save_dir, "results.json")
        print(f"saving results to {save_path}")
        json.dump(save_data, open(save_path, "w"))
    return loss, nonstrict_accuracy, strict_accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_size", type=int, default=250)
    parser.add_argument("--max_length", type=int, default=500)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--eval_batch_size", type=int, default=512)
    parser.add_argument("--n_eval", type=int, default=1000)
    parser.add_argument("--n_nodes", type=int, default=20)
    parser.add_argument("--p_edge", type=float, default=0.16)
    parser.add_argument("--load_dir", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default="eval/outputs")
    args = parser.parse_args()

    # initialize wandb
    # wandb.init(project="graph-connectivity", entity="codegen", config=args)

    # load the data
    eval_dataloader = make_all_pairs_dataloader(args) 

    # load the model
    if args.load_dir == None:
        model_config = BertConfig(vocab_size=args.vocab_size, max_position_embeddings=args.max_length, num_labels=2)
        model = BertForSequenceClassification(model_config)
    else:
        model = BertForSequenceClassification.from_pretrained(args.load_dir)

    print(f"Loading from {args.load_dir}", flush=True)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {total_params}")
    model.to(args.device)

    eval(model, eval_dataloader, save_dir=args.save_dir)
import os
from tqdm import tqdm
import argparse
from data import make_dataloaders, make_eval_dataloader
import torch
from transformers import BertConfig, BertForSequenceClassification
from transformers import get_scheduler
import wandb

def eval(model, eval_dataloader):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0
    with torch.no_grad():
        for batch in eval_dataloader:
            outputs = model(**batch)
            total_loss += outputs.loss
            all_preds.append(torch.argmax(outputs.logits, dim=1))
            all_labels.append(batch["labels"])

    all_preds, all_labels = torch.hstack(all_preds), torch.hstack(all_labels)
    accuracy = (all_preds == all_labels).type(torch.float).mean()
    loss = total_loss / len(eval_dataloader)
    # wandb.log({"eval_loss": loss.item()})
    # wandb.log({"eval_accuracy": accuracy.item()})
    print(f"eval loss: {loss.item()} | eval_accuracy: {accuracy.item()}")
    return loss.item(), accuracy.item()

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
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    args = parser.parse_args()

    # # initialize wandb
    # wandb.init(project="graph-connectivity", entity="codegen", config=args)

    # load the data
    eval_dataloader = make_eval_dataloader(args)

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

    eval(model, eval_dataloader)
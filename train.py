from tqdm import tqdm
import argparse
from data import make_dataloaders
import torch
from transformers import BertConfig, BertForSequenceClassification
from transformers import get_scheduler
import wandb

def initialize_optimizer(model, args):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    return optimizer

def eval(model, eval_dataloader):
    model.eval()
    with torch.no_grad():
        total_loss = 0
        total_accuracy = 0
        for batch in eval_dataloader:
            outputs = model(**batch)
            total_loss += outputs.loss

            # now calculate accuracy
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            labels = batch["labels"]
            accuracy = (preds == labels).type(torch.float).mean()
            total_accuracy += accuracy
            
        loss = total_loss / len(eval_dataloader)
        accuracy = total_accuracy / len(eval_dataloader)

        wandb.log({"eval_loss": loss.item()})
        wandb.log({"eval_accuracy": accuracy.item()})
    return loss.item(), accuracy.item()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_size", type=int, default=250)
    parser.add_argument("--max_length", type=int, default=500)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--n_train", type=int, default=100000)
    parser.add_argument("--n_eval", type=int, default=1000)
    parser.add_argument("--n_nodes", type=int, default=10)
    parser.add_argument("--p_edge", type=float, default=0.16)
    parser.add_argument("--eval_freq", type=float, default=100)
    parser.add_argument("--log_freq", type=float, default=100)
    args = parser.parse_args()

    # initialize wandb
    wandb.init(project="graph-connectivity", entity="codegen")

    # load the data
    train_dataloader, test_dataloader = make_dataloaders(args)
    num_training_steps = len(train_dataloader) * args.num_epochs

    # load the model
    model_config = BertConfig(vocab_size=args.vocab_size, max_position_embeddings=args.max_length, num_labels=2)
    model = BertForSequenceClassification(model_config)
    model.to(args.device)

    # load the optimizer and scheduler
    optimizer = initialize_optimizer(model, args)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    # train
    progress_bar = tqdm(range(num_training_steps))
    model.train()
    for epoch in range(args.num_epochs):
        for step, batch in enumerate(train_dataloader):
            # calculate loss
            outputs = model(**batch)
            loss = outputs.loss

            # calculate accuracy
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            labels = batch["labels"]
            accuracy = (preds == labels).type(torch.float).mean()

            # log
            wandb.log({"train_loss": loss.item()})
            wandb.log({"train_accuracy": accuracy.item()})
            
            # update
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

            if step % args.eval_freq == 0:
                loss, accuracy = eval(model, test_dataloader)
                print(f"eval loss: {loss} | accuracy: {accuracy}")
                model.train()
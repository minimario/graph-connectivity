import os
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

def eval(model, eval_dataloader, global_step):
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

        wandb.log({"eval_loss": loss.item()}, step=global_step)
        wandb.log({"eval_accuracy": accuracy.item()}, step=global_step)
    return loss.item(), accuracy.item()

def save_model(model, optimizer, lr_scheduler, global_step, save_dir) -> None:
    model.save_pretrained(os.path.join(save_dir, f"ckpt_{global_step}"))
    torch.save(optimizer.state_dict(), os.path.join(save_dir, f"ckpt_{global_step}", "optimizer.pt"))
    torch.save(lr_scheduler.state_dict(), os.path.join(args.save_dir, f"ckpt_{global_step}", "scheduler.pt"))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_size", type=int, default=250)
    parser.add_argument("--max_length", type=int, default=500)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--train_batch_size", type=int, default=512)
    parser.add_argument("--eval_batch_size", type=int, default=512)
    parser.add_argument("--n_train", type=int, default=100000)
    parser.add_argument("--n_eval", type=int, default=1000)
    parser.add_argument("--n_nodes", type=int, default=10)
    parser.add_argument("--p_edge", type=float, default=0.16)
    parser.add_argument("--eval_freq", type=float, default=500)
    parser.add_argument("--log_freq", type=float, default=100)
    parser.add_argument("--save_freq", type=float, default=500)
    parser.add_argument("--load_dir", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default="checkpoints")

    args = parser.parse_args()

    # initialize wandb
    wandb.init(project="graph-connectivity", entity="codegen", config=args)

    # load the data
    train_dataloader, test_dataloader = make_dataloaders(args)
    num_training_steps = len(train_dataloader) * args.num_epochs

    # load the model
    if args.load_dir == None:
        model_config = BertConfig(vocab_size=args.vocab_size, max_position_embeddings=args.max_length, num_labels=2)
        model = BertForSequenceClassification(model_config)
    else:
        model = BertForSequenceClassification.from_pretrained(args.load_dir)

    print(f"Loading from {args.load_dir}", flush=True)
    model.load_state_dict
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {total_params}")
    model.to(args.device)

    # load the optimizer and scheduler
    optimizer = initialize_optimizer(model, args)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )
    if args.load_dir:
        print("Reloading optimizer and scheduler")
        optimizer.load_state_dict(torch.load(os.path.join(args.load_dir, "optimizer.pt")))
        lr_scheduler.load_state_dict(torch.load(os.path.join(args.load_dir, "scheduler.pt")))
        print(f"Skipping steps up to {args.load_dir.split('_')[-1]}")

    # train
    progress_bar = tqdm(range(num_training_steps))
    model.train()
    global_step = 0
    for epoch in range(args.num_epochs):
        for step, batch in enumerate(train_dataloader):
            global_step += 1
            if args.load_dir and global_step <= int(args.load_dir.split("_")[-1]):
                progress_bar.update(1)
                continue

            # calculate loss
            outputs = model(**batch)
            loss = outputs.loss

            # calculate accuracy
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            labels = batch["labels"]
            accuracy = (preds == labels).type(torch.float).mean()

            # log
            wandb.log({"train_loss": loss.item()}, step=global_step)
            wandb.log({"train_accuracy": accuracy.item()}, step=global_step)
            
            # update
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

            if global_step % args.eval_freq == 0:
                loss, accuracy = eval(model, test_dataloader, global_step)
                print(f"step {global_step} | eval loss: {loss} | accuracy: {accuracy}", flush=True)
                model.train()
            
            if global_step % args.save_freq == 0 and args.save_dir != None:
                save_model(model, optimizer, lr_scheduler, global_step, args.save_dir)

    if args.save_dir != None:
        save_model(model, optimizer, lr_scheduler, global_step, args.save_dir)

                
                
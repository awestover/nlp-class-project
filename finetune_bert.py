"""
accelerate launch --mixed_precision bf16 finetune_bert.py \
--model_direction rtl \
--warmup_steps 500 \
--learning_rate 5e-5 \
--per_device_train_batch_size 128 \
--per_device_eval_batch_size 128 \
--output_dir checkpoints/bert_base_rtl/ \
--eval_steps 899 \
--block_size 128 \
--num_train_epochs 4 \
--weight_decay 1e-4

accelerate launch --mixed_precision bf16 finetune_bert.py \
--model_direction ltr \
--warmup_steps 500 \
--learning_rate 5e-5 \
--per_device_train_batch_size 128 \
--per_device_eval_batch_size 128 \
--output_dir checkpoints/bert_base_ltr/ \
--eval_steps 899 \
--block_size 128 \
--num_train_epochs 4 \
--weight_decay 1e-4
"""

import argparse
import math
import os

import accelerate
import torch
import transformers
import wandb
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from utils import preprocess_datasets, convert_to_torch_dataset, add_attn_hooks, causal_loss_wrapper


def parse_args():
    """
    Re-using HuggingFace arguments when possible (most of the help strings are directly copied).
    https://github.com/huggingface/transformers/blob/7bbc62474391aff64f63fcc064c975752d1fa4de/examples/pytorch/language-modeling/run_clm.py#L75
    """
    parser = argparse.ArgumentParser()

    # Model
    parser.add_argument("--model_direction", type=str, required=True, choices=["ltr", "rtl"],
                        help="Whether to train a left-to-right or right-to-left LM.")
    parser.add_argument("--model_name_or_path", type=str, default="bert-base-uncased",
                        help="Checkpoint to initialize weights from.")  # TODO: option for training from scratch w/ conf

    # Data
    parser.add_argument("--dataset_name", type=str, default="Salesforce/wikitext",
                        help="The name of the dataset to use (via the datasets library).")
    parser.add_argument("--dataset_config_name", type=str, default="wikitext-103-v1",
                        help="The configuration name of the dataset to use (via the datasets library).")
    # TODO: block_size, train on shorter seqs?
    parser.add_argument(
        "--block_size",
        type=int,
        help="Optional input sequence length after tokenization. "
             "The training dataset will be truncated in block of this size for training. "
             "Default to the model max input length for single sentence inputs (take into account special tokens)."
    )

    # Training
    parser.add_argument("--output_dir", type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, required=True)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--logging_steps", type=int, default=1,
                        help="Number of update steps between two logs.")
    parser.add_argument("--eval_steps", type=int, default=20000,
                        help="Number of update steps between two logs.")
    parser.add_argument("--dataloader_num_workers", type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()

    accelerator = accelerate.Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, log_with="wandb", project_dir=args.output_dir)
    # Will `add_attn_hooks` to `model` later
    model = transformers.AutoModelForMaskedLM.from_pretrained(args.model_name_or_path, attn_implementation="sdpa")
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name_or_path)

    # Data
    raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name)
    block_size = args.block_size if args.block_size is not None else model.config.max_position_embeddings
    model.config.max_position_embeddings = block_size

    processed_datasets = preprocess_datasets(raw_datasets, tokenizer, block_size)
    for split, hf_dataset in processed_datasets.items():
        processed_datasets[split] = convert_to_torch_dataset(hf_dataset)

    train_loader = DataLoader(processed_datasets["train"], batch_size=args.per_device_train_batch_size, shuffle=True)
    val_loader = DataLoader(processed_datasets["validation"], batch_size=args.per_device_eval_batch_size)
    # test_loader = DataLoader(processed_datasets["test"], batch_size=args.per_device_eval_batch_size)
    model, train_loader, val_loader = accelerator.prepare(model, train_loader, val_loader)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    lr_scheduler = transformers.get_scheduler(
        name=transformers.SchedulerType.COSINE,
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps * accelerator.num_processes,
        num_training_steps=args.num_train_epochs * math.ceil(len(train_loader) / args.gradient_accumulation_steps),
    )
    loss_fn = causal_loss_wrapper(args.model_direction)

    add_attn_hooks(model, args.model_direction)
    model.train()
    optimizer.zero_grad()

    accelerator.init_trackers(project_name="NLP-Class-Project", config=vars(args),
                              init_kwargs={"wandb": {"entity": "frostbyte"}})

    global_step = 0  # unaccumulated steps
    past_losses = []
    for epoch in tqdm(range(args.num_train_epochs), position=0, leave=True, desc="Epoch"):
        for batch in tqdm(train_loader, position=1, leave=False, desc="Train Iteration"):
            with accelerator.accumulate(model):
                labels = batch.pop("labels")
                outputs = model(**batch)
                loss = loss_fn(outputs.logits, labels)
                accelerator.backward(loss)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            past_losses.append(loss.item())
            if (global_step + 1) % args.logging_steps == 1:
                avg_train_loss = torch.tensor(past_losses).mean().item()  # Assuming 1 GPU
                accelerator.log({
                    "train_loss": avg_train_loss,
                    "learning_rate": lr_scheduler.get_last_lr()[0],
                })
                past_losses.clear()

            if (global_step + 1) % args.eval_steps == 0:
                val_loss_sum = val_examples = 0
                model.eval()
                for val_batch in tqdm(val_loader, position=2, leave=False, desc="Val Iteration"):
                    labels = val_batch.pop("labels")
                    with torch.no_grad():
                        outputs = model(**val_batch)

                    loss = loss_fn(outputs.logits, labels)

                    batch_size = labels.size(0)
                    val_loss_sum += loss.item() * batch_size
                    val_examples += batch_size

                accelerator.log({"val_loss": val_loss_sum / val_examples},
                                log_kwargs={"wandb": {"commit": False}})
                model.train()

            global_step += 1

    model.save_pretrained(os.path.join(args.output_dir, f"epoch_{epoch}_checkpt"))


if __name__ == "__main__":
    main()

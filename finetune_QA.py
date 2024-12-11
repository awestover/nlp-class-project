"""
accelerate launch --mixed_precision bf16 finetune_QA.py \
--model_direction rtl \
--checkpoint_path /home/sipb/nlp-class-project/checkpoints/distilbert_base_rtl/epoch_3_checkpt \
--tokenizer_name distilbert/distilbert-base-uncased \
--warmup_steps 100 \
--learning_rate 1e-5 \
--per_device_train_batch_size 128 \
--per_device_eval_batch_size 128 \
--output_dir checkpoints/qa_distilbert_base_rtl/ \
--eval_steps 38 \
--block_size 128 \
--num_train_epochs 50 \
--weight_decay 1e-4


accelerate launch --mixed_precision bf16 finetune_QA.py \
--model_direction ltr \
--checkpoint_path /home/sipb/nlp-class-project/checkpoints/distilbert_base_ltr/epoch_3_checkpt \
--tokenizer_name distilbert/distilbert-base-uncased \
--warmup_steps 100 \
--learning_rate 1e-5 \
--per_device_train_batch_size 128 \
--per_device_eval_batch_size 128 \
--output_dir checkpoints/qa_distilbert_base_ltr/ \
--eval_steps 38 \
--block_size 128 \
--num_train_epochs 50 \
--weight_decay 1e-4

accelerate launch --mixed_precision bf16 finetune_QA.py \
--model_direction ltr \
--checkpoint_path /home/sipb/nlp-class-project/checkpoints/distilbert_base_ltr/epoch_3_checkpt \
--tokenizer_name distilbert/distilbert-base-uncased \
--warmup_steps 100 \
--learning_rate 1e-5 \
--per_device_train_batch_size 128 \
--per_device_eval_batch_size 128 \
--output_dir checkpoints/qa_distilbert_base_ltr_overfit/ \
--eval_steps 50 \
--block_size 128 \
--num_train_epochs 1000 \
--weight_decay 0
"""



import argparse
import math
import os
from collections import defaultdict

import accelerate
import torch
import transformers
import wandb
from datasets import load_dataset 
from torch.utils.data import Dataset, DataLoader
from transformers.data.data_collator import default_data_collator
from tqdm.auto import tqdm

from utils import preprocess_datasets, convert_to_torch_dataset, add_attn_hooks, causal_loss_wrapper

#### HERE WE do the dataset stuff
class DatasetAQ(Dataset):
    def __init__(self, qa_pairs, text_direction, tokenizer):
        self.qa_pairs = qa_pairs
        self.text_direction = text_direction
        self.tokenizer = tokenizer 
    
    def __getitem__(self, idx):
        question, answer = self.qa_pairs[idx]
        sentence = torch.cat([question, answer], dim=0) if self.text_direction.lower() == "rtl" else torch.cat([answer, question], dim=0)

        # TODO: length
        num_to_pad = self.tokenizer.model_max_length - sentence.size(0)
        assert num_to_pad >= 0, (sentence.size(), self.tokenizer.model_max_length)

        if num_to_pad > 0:
            pad_tokens = torch.full((num_to_pad,), self.tokenizer.pad_token_id, dtype=sentence.dtype)
            pad_labels = torch.full((num_to_pad,), -100, dtype=sentence.dtype)

            if self.text_direction.lower() == "rtl":
                input_ids = torch.cat([pad_tokens, sentence], dim=0)
                labels = torch.cat([pad_labels, sentence], dim=0)
                attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
                attention_mask[:num_to_pad] = 0
            else:
                input_ids = torch.cat([sentence, pad_tokens], dim=0)
                labels = torch.cat([sentence, pad_labels], dim=0)
                attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
                attention_mask[-num_to_pad:] = 0
                
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }
    
    def __len__(self):
        return len(self.qa_pairs)
            
####



def parse_args():
    """
    Re-using HuggingFace arguments when possible (most of the help strings are directly copied).
    https://github.com/huggingface/transformers/blob/7bbc62474391aff64f63fcc064c975752d1fa4de/examples/pytorch/language-modeling/run_clm.py#L75
    """
    parser = argparse.ArgumentParser()

    # Model
    parser.add_argument("--model_direction", type=str, required=True, choices=["ltr", "rtl"],
                        help="Whether to train a left-to-right or right-to-left LM.")
    parser.add_argument("--checkpoint_path", type=str,
                        help="Path to load model weights from.")

    # Data
    parser.add_argument("--tokenizer_name", type=str,
                    help="Name of tokenizer to load.")
    parser.add_argument("--dataset_name", type=str, default="truthfulqa/truthful_qa",
                        help="The name of the dataset to use (via the datasets library).")
    parser.add_argument("--dataset_config_name", type=str, default="generation",
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
    parser.add_argument("--train_from_scratch", action="store_true")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, required=True)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--scheduler", type=str, default="cosine")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--logging_steps", type=int, default=1,
                        help="Number of update steps between two logs.")
    parser.add_argument("--eval_steps", type=int, default=20000,
                        help="Number of update steps between two logs.")
    parser.add_argument("--dataloader_num_workers", type=int, default=8)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    transformers.set_seed(42)

    accelerator = accelerate.Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, log_with="wandb", project_dir=args.output_dir)
    # Will `add_attn_hooks` to `model` later

    # Load model weights in both cases, but re-initialize if training from scratch
    model = transformers.AutoModelForMaskedLM.from_pretrained(args.checkpoint_path, attn_implementation="sdpa", ignore_mismatched_sizes=True)
    if args.train_from_scratch:
        model.apply(model._init_weights)
        model.tie_weights()  # probably not applicable

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.tokenizer_name)

    # Data
    raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name)
    block_size = args.block_size if args.block_size is not None else model.config.max_position_embeddings
    model.config.max_position_embeddings = block_size
    tokenizer.model_max_length = block_size

    # QA-specific code
    all_data = raw_datasets["validation"]
    transformers.set_seed(42)
    train_val_split = all_data.train_test_split(test_size=0.2, shuffle=True)
    val_test_split = train_val_split['test'].train_test_split(test_size=0.5, shuffle=False)
    train_dataset = train_val_split['train']
    val_dataset = val_test_split['train']
    test_dataset = val_test_split['test']
    
    qa_pairs = defaultdict(list)
    for data_name, dataset in zip(["test","train","val"], [train_dataset, test_dataset, val_dataset]):
        for row in dataset:
            tokenized_question = tokenizer("Question: "+ row["question"], return_tensors="pt")["input_ids"].squeeze(0)
            for ans_type in ["correct_answers", "incorrect_answers"]:
                for answer in row[ans_type]:
                    # the [:, 1:] thing is to remove CLS token
                    qa_pairs[data_name].append((tokenized_question, tokenizer(f"Answer: {answer}", return_tensors="pt")["input_ids"].squeeze(0)[1:]))

    train_dataset = DatasetAQ(qa_pairs["train"], args.model_direction, tokenizer)
    test_dataset = DatasetAQ(qa_pairs["test"], args.model_direction, tokenizer)
    val_dataset = DatasetAQ(qa_pairs["val"], args.model_direction, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=args.per_device_train_batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.per_device_eval_batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=args.per_device_eval_batch_size)

    model, train_loader, test_loader, val_loader = accelerator.prepare(model, train_loader, test_loader, val_loader)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    lr_scheduler = transformers.get_scheduler(
        name=transformers.SchedulerType.COSINE,
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps * accelerator.num_processes,
        # num_training_steps=args.num_train_epochs * math.ceil(len(train_loader) / args.gradient_accumulation_steps),
        num_training_steps=args.num_train_epochs * len(train_loader),
    )

    lr_scheduler = accelerator.prepare(lr_scheduler)  # testing if this fixes learning rate

    loss_fn = causal_loss_wrapper(args.model_direction)

    add_attn_hooks(model, args.model_direction)
    model.train()
    optimizer.zero_grad()

    wandb.require("core")
    accelerator.init_trackers(
        project_name="NLP-Class-Project",
        config=vars(args) | {"model_parameters": sum(p.numel() for p in model.parameters())},
        init_kwargs={"wandb": {"entity": "frostbyte"}}
    )

    global_step = 0  # unaccumulated steps
    past_losses = []
    best_val_loss = float("inf")
    best_checkpt_path = os.path.join(args.output_dir, f"best_checkpt")

    for epoch in tqdm(range(args.num_train_epochs), position=0, leave=True, desc="Epoch"):
        for step, batch in enumerate(tqdm(train_loader, position=1, leave=False, desc="Train Iteration")):
            with accelerator.accumulate(model):
                labels = batch.pop("labels")
                outputs = model(**batch)
                loss = loss_fn(outputs.logits, labels)
                accelerator.backward(loss)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            past_losses.append(loss.item())
            if (global_step + 1) % args.logging_steps == 0:
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

                val_loss = val_loss_sum / val_examples
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    model.save_pretrained(best_checkpt_path)

                accelerator.log({"val_loss": val_loss_sum / val_examples},
                                log_kwargs={"wandb": {"commit": False}})
                model.train()

            if ((step + 1) % args.gradient_accumulation_steps == 0) or step == (len(train_loader) - 1):
                global_step += 1

    # model.save_pretrained(os.path.join(args.output_dir, f"epoch_{epoch}_checkpt"))

    # testing
    model.from_pretrained(best_checkpt_path)
    model.eval()
    with torch.no_grad():
        test_loss_sum = test_examples = 0
        for test_batch in tqdm(test_loader):
            labels = test_batch.pop("labels")
            outputs = model(**test_batch)

            loss = loss_fn(outputs.logits, labels)

            batch_size = labels.size(0)
            test_loss_sum += loss.item() * batch_size
            test_examples += batch_size

        accelerator.log({"test_loss": test_loss_sum / test_examples})


if __name__ == "__main__":
    main()

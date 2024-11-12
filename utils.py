from itertools import chain

import torch
import torch.nn as nn
import transformers
from datasets import DatasetDict
from transformers.models.bert.modeling_bert import BERT_SELF_ATTENTION_CLASSES
from transformers.models.distilbert.modeling_distilbert import DISTILBERT_ATTENTION_CLASSES


BERT_ATTENTIONS = tuple(BERT_SELF_ATTENTION_CLASSES.values())
DISTILBERT_ATTENTIONS = tuple(DISTILBERT_ATTENTION_CLASSES.values())
IMPLEMENTED_ATTENTIONS = tuple(BERT_ATTENTIONS + DISTILBERT_ATTENTIONS)


def ltr_mask(seq_len: int) -> torch.Tensor:
    mask = torch.ones((seq_len, seq_len), dtype=torch.bool)
    return torch.tril(mask)


def rtl_mask(seq_len: int) -> torch.Tensor:
    return ltr_mask(seq_len).T


def add_attn_hooks(model: transformers.PreTrainedModel, model_direction: str) -> None:
    """
    Forces bidirectional `model` into a unidirectional one based on `model_direction`.
    Adds hooks to `model`'s self-attention blocks, in-place.

    Args:
        model: only implemented for BERT models right now
        model_direction: one of "ltr" or "rtl"
    """
    assert model_direction.lower() in ("ltr", "rtl")
    mask_func = ltr_mask if model_direction.lower() == "ltr" else rtl_mask
    model.register_buffer("attention_mask", mask_func(model.config.max_position_embeddings).to(model.device))

    def get_attention_mask(seq_len: int) -> torch.Tensor:
        """
        Returns `model.attention_mask` if `seq_len` is the max length, generate new attention mask otherwise.
        """
        # During training, we should always be padding to max length, so we can always use `model.attention_mask`.
        if seq_len != model.config.max_position_embeddings:
            assert not torch.is_grad_enabled()
            return ltr_mask(seq_len).to(model.device)  # TODO: should this be mask_func?
            # TODO: should we just have a different function to "prepare" model for inference?
        else:
            return model.attention_mask

    def attn_hook(attn_module: nn.Module, args: tuple, kwargs: dict):
        """
        Uses nonlocal `model.attention_mask` to save memory.
        """
        if isinstance(attn_module, BERT_ATTENTIONS):
            """
            Assuming https://github.com/huggingface/transformers/blob/33868a057c02f0368ba63bd1edb746be38fe3d90/src/transformers/models/bert/modeling_bert.py#L515
            so no `kwargs` and `attention_mask` is second positional arg.
            """
            assert not kwargs

            args = list(args)
            seq_len = args[0].size(1)
            args[1] = get_attention_mask(seq_len)
            args = tuple(args)
        elif isinstance(attn_module, DISTILBERT_ATTENTIONS):
            """
            Assuming https://github.com/huggingface/transformers/blob/33eef992503689ba1af98090e26d3e98865b2a9b/src/transformers/models/distilbert/modeling_distilbert.py#L481
            so "mask" in `kwargs`.
            """
            assert not args and "mask" in kwargs and "query" in kwargs, f"{args=} {kwargs=}"
            seq_len = kwargs["query"].size(1)
            kwargs["mask"] = get_attention_mask(seq_len)
        else:
            raise NotImplementedError(f"{attn_module=}")

        return args, kwargs

    for name, module in model.named_modules():
        if isinstance(module, IMPLEMENTED_ATTENTIONS):
            module._forward_pre_hooks.clear()  # in case we run multiple times
            module.register_forward_pre_hook(attn_hook, with_kwargs=True)


def causal_loss_wrapper(model_direction: str):
    ce_loss = torch.nn.CrossEntropyLoss()

    def loss_fn(logits, labels):
        if model_direction.lower() == "ltr":
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
        elif model_direction.lower() == "rtl":
            shift_logits = logits[..., 1:, :].contiguous()
            shift_labels = labels[..., :-1].contiguous()
        else:
            raise NotImplementedError(f"{model_direction=}")

        # Flatten the tokens
        return ce_loss(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

    return loss_fn


def preprocess_datasets(
    raw_datasets: DatasetDict,
    tokenizer: transformers.PreTrainedTokenizer,
    block_size: int
) -> DatasetDict:
    """
    Preprocess datasets.
    Closely follows https://github.com/huggingface/transformers/blob/7bbc62474391aff64f63fcc064c975752d1fa4de/examples/pytorch/language-modeling/run_clm.py#L449

    `raw_datasets` is the output of `load_datasets()`, expected to always have a "train" split
    """
    column_names = list(raw_datasets["train"].features)
    text_column_name = "text" if "text" in column_names else column_names[0]
    tokenized_datasets = raw_datasets.map(
        lambda examples: tokenizer(examples[text_column_name]),
        batched=True,
        num_proc=8,
        remove_columns=column_names,
        desc="Running tokenizer on dataset",
    )

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
        # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
    # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
    # to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/process#map

    # # with training_args.main_process_first(desc="grouping texts together"):
    return tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=8,
        # load_from_cache_file=not data_args.overwrite_cache,
        desc=f"Grouping texts in chunks of {block_size}",
    )


def convert_to_torch_dataset(hf_dataset):
    """ Convert HuggingFace Dataset into PyTorch Dataset """
    return hf_dataset.with_format("torch")

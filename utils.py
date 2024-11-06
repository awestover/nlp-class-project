import torch
import torch.nn as nn
import transformers


def ltr_mask(seq_len: int) -> torch.Tensor:
    mask = torch.ones((seq_len, seq_len), dtype=torch.bool)
    return torch.tril(mask, diagonal=-1)


def rtl_mask(seq_len: int) -> torch.Tensor:
    return ltr_mask(seq_len).T


def add_attn_hooks(model: transformers.BertModel, text_direction: str) -> None:
    """
    Forces bidirectional `model` into a unidirectional one based on `direction`.
    Adds hooks to `model`'s self-attention blocks, in-place.

    Args:
        model: only implemented for BERT models right now
        text_direction: one of "ltr" or "rtl"
    """
    assert text_direction.lower() in ("ltr", "rtl")
    mask_func = ltr_mask if text_direction.lower() == "ltr" else rtl_mask
    model.register_buffer("attn_mask", mask_func(model.config.max_position_embeddings).to(model.device))

    def attn_hook(attn_module: nn.Module, args: tuple, kwargs: dict):
        """
        Assuming https://github.com/huggingface/transformers/blob/33868a057c02f0368ba63bd1edb746be38fe3d90/src/transformers/models/bert/modeling_bert.py#L515
        so no `kwargs` and `attention_mask` is second positional arg.

        Uses nonlocal `model.attn_mask` to save memory.
        """
        assert not kwargs

        args = list(args)
        assert args[1].size()[-2:] == model.attn_mask.size(), f"{args[1].size()=} {model.attn_mask.size()=}"
        args[1] = model.attn_mask
        return tuple(args), kwargs

    for name, module in model.named_modules():
        if isinstance(module, transformers.models.bert.modeling_bert.BertSelfAttention):
            module._forward_hooks.clear()  # in case we run multiple times
            module.register_forward_pre_hook(attn_hook, with_kwargs=True)

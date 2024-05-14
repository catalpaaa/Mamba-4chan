# Modified from https://github.com/state-spaces/mamba/blob/main/mamba_ssm/utils/generation.py <3

# Copyright (c) 2023, Albert Gu, Tri Dao.

from dataclasses import dataclass, field
from typing import Callable, Optional

import torch
from torch import Tensor
from transformers.generation import (
    GreedySearchDecoderOnlyOutput,
    SampleDecoderOnlyOutput,
    TextStreamer,
)


@dataclass
class InferenceParams:
    """Inference parameters that are passed to the main model in order
    to efficienly calculate and store the context during inference."""

    max_seqlen: int
    max_batch_size: int
    seqlen_offset: int = 0
    batch_size_offset: int = 0
    key_value_memory_dict: dict = field(default_factory=dict)
    lengths_per_sample: Optional[Tensor] = None

    def reset(self, max_seqlen, max_batch_size):
        self.max_seqlen = max_seqlen
        self.max_batch_size = max_batch_size
        self.seqlen_offset = 0
        if self.lengths_per_sample is not None:
            self.lengths_per_sample.zero_()


def modify_logits_for_min_p_filtering(logits, min_p):
    """Set the logits for none min_p values to -inf. Done in-place."""
    if min_p <= 0.0 or min_p >= 1.0:
        return
    indices_to_remove = logits < min_p
    logits.masked_fill_(indices_to_remove, float("-Inf"))


# https://github.com/NVIDIA/Megatron-LM/blob/0bb597b42c53355a567aba2a1357cc34b9d99ddd/megatron/text_generation/sampling.py
# https://github.com/huggingface/transformers/blob/a44985b41cfa2de48a5e1de7f1f93b7483da25d1/src/transformers/generation/logits_process.py#L231
def modify_logits_for_top_k_filtering(logits, top_k):
    """Set the logits for none top-k values to -inf. Done in-place."""
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits.masked_fill_(indices_to_remove, float("-Inf"))


# https://github.com/NVIDIA/Megatron-LM/blob/0bb597b42c53355a567aba2a1357cc34b9d99ddd/megatron/text_generation/sampling.py
# https://github.com/huggingface/transformers/blob/a44985b41cfa2de48a5e1de7f1f93b7483da25d1/src/transformers/generation/logits_process.py#L170
def modify_logits_for_top_p_filtering(logits, top_p):
    """Set the logits for none top-p values to -inf. Done in-place."""
    if top_p <= 0.0 or top_p >= 1.0:
        return
    # First sort and calculate cumulative sum of probabilities.
    sorted_logits, sorted_indices = torch.sort(logits, descending=False)
    cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
    # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
    sorted_indices_to_remove = cumulative_probs <= (1 - top_p)
    # scatter sorted tensors to original indexing
    indices_to_remove = sorted_indices_to_remove.scatter(
        1, sorted_indices, sorted_indices_to_remove
    )
    logits.masked_fill_(indices_to_remove, float("-inf"))


def modify_logit_for_repetition_penalty(
    logits, prev_output_tokens, repetition_penalty=1.0
):
    """Apply repetition penalty. See https://arxiv.org/abs/1909.05858
    logits: (batch_size, vocab_size)
    prev_output_tokens: (batch_size, seq_len)
    """
    if repetition_penalty == 1.0:
        return logits
    score = torch.gather(logits, 1, prev_output_tokens)
    # if score < 0 then repetition penalty has to be multiplied to reduce the previous token probability
    score = torch.where(
        score < 0, score * repetition_penalty, score / repetition_penalty
    )
    logits.scatter_(1, prev_output_tokens, score)
    return logits


def sample(logits, top_k=1, top_p=0.0, min_p=0.0, temperature=1.0):
    """Sample from top-k logits.
    Arguments:
        logits: Tensor of shape (batch_size, vocab_size)
    """
    if top_k == 1:  # Short-circuit for greedy decoding
        return logits.argmax(dim=-1)
    else:
        if top_p > 0.0:
            assert top_p <= 1.0, "top-p should be in (0, 1]."
        if top_k > 0:
            top_k = min(top_k, logits.size(-1))  # Safety check
            logits_top, indices = torch.topk(logits, top_k, dim=-1)
            if temperature != 1.0:
                logits_top /= temperature
            modify_logits_for_top_p_filtering(logits_top, top_p)
            return indices[
                torch.arange(indices.shape[0], device=indices.device),
                torch.multinomial(
                    torch.softmax(logits_top, dim=-1), num_samples=1
                ).squeeze(dim=-1),
            ]
        else:
            if min_p > 0.0:
                logits_top = logits.clone()
                max_prob = logits_top[..., 0].item()
                modify_logits_for_min_p_filtering(logits_top, min_p)
                if temperature != 1.0:
                    logits_top /= temperature
                return torch.multinomial(
                    torch.softmax(logits_top, dim=-1), num_samples=1
                ).squeeze(dim=-1)
            # Clone so that when we modify for top_p we don't change the original logits
            logits_top = logits / temperature if temperature != 1.0 else logits.clone()
            modify_logits_for_top_p_filtering(logits_top, top_p)
            return torch.multinomial(
                torch.softmax(logits_top, dim=-1), num_samples=1
            ).squeeze(dim=-1)


@torch.inference_mode()
def decode(
    input_ids,
    model,
    max_length,
    top_k=1,
    top_p=0.9,
    min_p=0.0,
    temperature=0.7,
    repetition_penalty=1.1,
    eos_token_id=0,
    streamer: Optional[TextStreamer] = None,
):
    """Decoding, either greedy or with top-k or top-p sampling.
    If top-k = 0, don't limit the number of candidates (pure sampling).
    Top-k and top-p can be used together. If top_k > 0 and top_p > 0, then top-k is applied first,
    then top-p.
    We assume that all sequences in the same batch have the same length.

    Arguments:
        input_ids: (batch, seq_len)
        max_length: int
        teacher_outputs (optional): (batch, seq_len). If provided, instead of sampling from the
            logits, the next token is taken from the teacher_outputs. Useful for testing.
    Returns: GreedySearchDecoderOnlyOutput or SampleDecoderOnlyOutput, with the following fields:
        sequences: (batch, max_length)
        scores: tuples of (batch, vocab_size)
    """
    if streamer is not None:
        streamer.put(input_ids.cpu())

    batch_size, _ = input_ids.shape
    inference_params = InferenceParams(max_seqlen=max_length, max_batch_size=batch_size)

    def get_logits(input_ids, inference_params):
        logits = model(
            input_ids,
            inference_params=inference_params,
            num_last_tokens=1,
        ).squeeze(dim=1)

        return logits

    def sample_tokens(logits):
        token = sample(
            logits, top_k=top_k, top_p=top_p, min_p=min_p, temperature=temperature
        )
        return token.unsqueeze(1)

    def should_stop(current_token, inference_params):
        if inference_params.seqlen_offset == 0:
            return False
        if eos_token_id is not None and (current_token == eos_token_id).all():
            return True
        if inference_params.seqlen_offset >= max_length - 1:
            return True
        return False

    scores, sequences = [], [input_ids]
    sequences_cat = input_ids

    while not should_stop(sequences[-1], inference_params):
        scores.append(get_logits(sequences[-1], inference_params))
        inference_params.seqlen_offset += sequences[-1].shape[1]

        if repetition_penalty == 1.0:
            sampled_tokens = sample_tokens(scores[-1])
        else:
            logits = modify_logit_for_repetition_penalty(
                scores[-1].clone(), sequences_cat, repetition_penalty
            )
            sampled_tokens = sample_tokens(logits)
            sequences_cat = torch.cat([sequences_cat, sampled_tokens], dim=1)
        sequences.append(sampled_tokens)

        if streamer is not None:
            streamer.put(sampled_tokens.cpu())

    if streamer is not None:
        streamer.end()

    output_cls = (
        GreedySearchDecoderOnlyOutput if top_k == 1 else SampleDecoderOnlyOutput
    )

    return output_cls(
        sequences=torch.cat(sequences, dim=1), scores=tuple(scores)
    ).sequences


class GenerationMixin:
    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        raise NotImplementedError

    def generate_token(
        self,
        input_ids: Tensor,
        max_length: int,
        top_k: int = 1,
        top_p: float = 0.9,
        min_p: float = 0.0,
        temperature: float = 0.7,
        repetition_penalty: float = 1.2,
        eos_token_id: int = 0,
        streamer: Optional[TextStreamer] = None,
    ):
        return decode(
            input_ids,
            self,
            max_length,
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            eos_token_id=eos_token_id,
            streamer=streamer,
        )

    def generate_text(
        self,
        tokenizer,
        input_str: str,
        max_length: int,
        top_k: int = 1,
        top_p: float = 0.9,
        min_p: float = 0.0,
        temperature: float = 0.7,
        repetition_penalty: float = 1.2,
    ):
        input_ids = tokenizer(input_str, return_tensors="pt")["input_ids"].cuda()

        pred = decode(
            input_ids,
            self,
            max_length,
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            eos_token_id=tokenizer.eos_token_id,
        )

        return tokenizer.batch_decode(pred)[0]


@dataclass
class DecodingCGCache:
    max_batch_size: int = 0
    max_seqlen: int = 0
    device = None
    dtype = None
    callables: dict = field(default_factory=dict)
    mempool = None
    inference_params: Optional[InferenceParams] = None
    run: Optional[Callable] = None

import six
import math
import torch
from torch import nn


GPT2_PADDED_INPUTS = ["input_ids", "lm_labels", "token_type_ids"]
RAG_GPU_KEYS = ["input_ids", "input_attn_mask", "input_eos", "token_type_ids", "dialog_tti", "dialog", "decoder_input_ids",
               "persona_candidates", "persona_candidates_attn_mask", "persona_candidates_g", "persona_candidates_attn_mask_g",
               "knowledge_candidates", "knowledge_candidates_attn_mask", "knowledge_candidates_g", "knowledge_candidates_attn_mask_g",
               "persona_can_idx", "persona_grounding",
               "knowledge_can_idx", "knowledge_grounding", "tot_knowledge", "tot_knowledge_eos",
               "lm_labels", "mc_token_ids", "tot_knowledge_token_ids", "edge_index", "edge_type"]

class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text.decode("utf-8", "ignore")
        elif isinstance(text, unicode):
            return text
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")


def rag_forward_args(batch):
    forward_args = {
        "input_ids": batch["input_ids"],
        "input_attn_mask": batch["input_attn_mask"],
        "decoder_input_ids": batch["decoder_input_ids"],
        "persona_input_ids": batch["persona_candidates"],
        "persona_attn_mask": batch["persona_candidates_attn_mask"],
        "knowledge_input_ids": batch["knowledge_candidates"],
        "knowledge_attn_mask": batch["knowledge_candidates_attn_mask"],
        "persona_grounding": batch["persona_grounding"],
        "knowledge_grounding": batch["knowledge_grounding"],
        "lm_labels": batch["lm_labels"],
    }
    
    return forward_args


def run_rag_inference(batch):
    forward_args = {
        "input_ids": batch["input_ids"],
        "input_attn_mask": batch["input_attn_mask"],
        "decoder_input_ids": batch["decoder_input_ids"],
        "persona_input_ids": batch["persona_candidates"],
        "persona_attn_mask": batch["persona_candidates_attn_mask"],
        "knowledge_input_ids": batch["knowledge_candidates"],
        "knowledge_attn_mask": batch["knowledge_candidates_attn_mask"],
        "persona_grounding": batch["persona_grounding"],
        "knowledge_grounding": batch["knowledge_grounding"],
        "lm_labels": batch["lm_labels"],
    }
    
    return forward_args


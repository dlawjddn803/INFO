import logging
import torch
import math
from torch import nn
from transformers import (
    RagTokenizer,
)

from data_reader import (
    RagProcessor,
    RagDataset
)
from data_reader.data_processor import CustomChatProcessor

from models.ReasoningRAGCT import (
    ReasoningRAGCT
)

from .rag_utils import rag_forward_args, RAG_GPU_KEYS


def move_to_cuda(maybe_tensor, device):
    if torch.is_tensor(maybe_tensor):
        return maybe_tensor.cuda(device)
    elif isinstance(maybe_tensor, dict):
        return {
            key: move_to_cuda(value, device)
            for key, value in maybe_tensor.items()
        }
    elif isinstance(maybe_tensor, list):
        return [move_to_cuda(x, device) for x in maybe_tensor]
    else:
        return maybe_tensor


class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


def init_logger(filename=None):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        filename=filename
    )


MODEL_CLASSES = {
    "rag-tok-ct": ReasoningRAGCT,
}

TOKENIZER_CLASSES = {
    "rag-tok-ct": RagTokenizer,
}

DATASET_CLASSES = {
    "rag-tok-ct": RagDataset,
}

DATA_PROCESSORS = {
    "pkchat": CustomChatProcessor,

}

MODEL_PROCESSORS = {
    "rag-tok-ct": RagProcessor,
}

TRAIN_ARGS = {
    "rag-tok-ct": rag_forward_args,
}

MODEL_INFERENCE = {

}

GPU_KEYS = {
    "rag-tok-ct": RAG_GPU_KEYS,
}

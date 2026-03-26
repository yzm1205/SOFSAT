from __future__ import annotations

import importlib
import json
import os
import sys
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union
import dotenv
import numpy as np
import torch
from torch import Tensor
from transformers import (
    AutoConfig,
    AutoProcessor,
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
)
from tqdm import tqdm
from math import ceil


sys.path.insert(0, "./")

from src.utils import full_path

TextInput = Union[str, Sequence[str]]

dotenv.load_dotenv(os.getenv("./models/.env"))
_hf_token = os.getenv("huggingface_token")

def _derive_model_label(model_name: Optional[str], custom_class_path: Optional[str]) -> str:
    if model_name:
        return model_name.split("/")[-1]
    if custom_class_path:
        return custom_class_path.split(".")[-1]
    return "custom_embedder"


class BaseEmbedder(ABC):
    """Contract shared by all embedding providers."""

    @abstractmethod
    def encode(self, texts: TextInput, **kwargs: Any) -> np.ndarray:  # pragma: no cover - abstract
        """Convert one or more texts into embeddings."""


class LLMEmbedder(BaseEmbedder):
    """Default Hugging Face-based embedder."""

    def __init__(
        self,
        model_name: str,
        device: Optional[Union[str, torch.device]] = None,
        *,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        max_length: int = 1024,
    ) -> None:
        self.model_name = model_name
        self.device = self._resolve_device(device)
        self.tokenizer_kwargs = tokenizer_kwargs or {}
        self.model_kwargs = model_kwargs or {}
        self.max_length = max_length

        model_dir = self._resolve_model_path(model_name)

        config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
        # self.model_type = config.architectures[0] if config.architectures else ""
        
        self.model_type = (config.architectures[0] if getattr(config, "architectures", None) else "") or ""
        self.is_multimodal = hasattr(config, "vision_config") or getattr(config, "is_multimodal", False)
        self.image_token_id = getattr(config, "image_token_index", None)
        
        # Prefer Processor for multimodal (even for text-only); else Tokenizer
        self.processor = None
        self.tokenizer = None
        if self.is_multimodal:
            self.processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True, **self.tokenizer_kwargs)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, **self.tokenizer_kwargs)


        default_model_kwargs = dict(
            trust_remote_code=True,
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map=self._device_map_for_hf(),
        )
        default_model_kwargs.update(self.model_kwargs)
        
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_dir, **default_model_kwargs
            )
        except Exception:
            self.model = AutoModel.from_pretrained(
                model_dir, **default_model_kwargs
            )
        # if self.tokenizer.pad_token is None:
        #     self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.tokenizer is not None and self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model.eval()
        if default_model_kwargs.get("device_map") in (None, "sequential") and self.device:
            self.model.to(self.device)
    
    def encode(
        self,
        texts: TextInput,
        *,
        batch_size: int = 32,
        show_progress: bool = True,
        **_: Any,
    ) -> np.ndarray:
    
        batched_texts: List[str] = [texts] if isinstance(texts, str) else list(texts)
        if not batched_texts:
            raise ValueError("texts must contain at least one element")

        embeddings: List[Tensor] = []
        iterator: Iterable[int] = range(0, len(batched_texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, 
                            total=ceil(len(batched_texts) / batch_size),
                            desc="Embedding batches")

        for start in iterator:
            chunk = batched_texts[start : start + batch_size]
            inputs = self._prepare_text_inputs(chunk)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True, use_cache=False)

            hidden = outputs.hidden_states[-1]
            attn_mask = inputs.get("attention_mask")
            if attn_mask is None:
                attn_mask = torch.ones(hidden.size()[:2], dtype=torch.long, device=hidden.device)

            mask = attn_mask.bool()
            pooled = self._mean_pool(hidden, mask)
            embeddings.append(pooled.detach().cpu())

        stacked = torch.cat(embeddings, dim=0)
        array = stacked.numpy()
        return _ensure_2d_array(array, provider=self.__class__.__name__)

    @staticmethod
    def _mean_pool(token_embeddings: Tensor, attention_mask: Tensor) -> Tensor:
        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        summed = torch.sum(token_embeddings * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counts

    @staticmethod
    def _resolve_model_path(model_name: str) -> str:
        try:
            return check_model_in_cache(model_name)
        except ValueError:
            return model_name

    @staticmethod
    def _resolve_device(device: Optional[Union[str, torch.device]]) -> torch.device:
        if device is None:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if isinstance(device, torch.device):
            return device
        return torch.device(device)

    def _device_map_for_hf(self) -> Union[str, Dict[str, int]]:
        if isinstance(self.device, torch.device):
            if self.device.type == "cuda":
                index = 0 if self.device.index is None else self.device.index
                return {"": index}
            return "cpu"
        return str(self.device)
    
    def _prepare_text_inputs(self, texts: List[str]) -> Dict[str, Tensor]:
        if self.processor is not None:
            # Multimodal checkpoint, text-only path
            return self.processor(
                text=texts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length
            )
        else:
            return self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_token_type_ids=False,
            )


class _ValidatedEmbedder(BaseEmbedder):
    def __init__(self, delegate: BaseEmbedder) -> None:
        self._delegate = delegate

    def encode(self, texts: TextInput, **kwargs: Any) -> np.ndarray:
        array = self._delegate.encode(texts, **kwargs)
        return _ensure_2d_array(array, provider=self._delegate.__class__.__name__)


def check_model_in_cache(model_name: str) -> str:
    if model_name in {"LLaMA3", "llama3"}:
        return str(full_path("/data/shared/llama3-8b/Meta-Llama-3-8B_shard_size_1GB"))

    if model_name in {"Mistral", "mistral"}:
        return str(full_path("/data/shared/mistral-7b-v03/Mistral-7B-v0.3_shard_size_1GB"))

    if model_name in {"olmo", "OLMo"}:
        return str(full_path("/data/shared/olmo/OLMo-7B_shard_size_2GB"))

    raise ValueError(f"Model '{model_name}' not found in local cache.")


def build_embedder(
    model_name: Optional[str],
    device: Optional[Union[str, torch.device]] = None,
    *,
    custom_class_path: Optional[str] = None,
    custom_kwargs: Optional[Union[str, Dict[str, Any]]] = None,
) -> BaseEmbedder:
    class_path = custom_class_path or os.getenv("CUSTOM_CLASS_PATH")
    kwargs = _load_kwargs(custom_kwargs)

    if class_path:
        embedder_cls = _import_embedder_class(class_path)
        try:
            instance = embedder_cls(**kwargs)
        except TypeError as exc:
            raise TypeError(
                f"Failed to instantiate custom embedder '{class_path}': {exc}"
            ) from exc
        if not isinstance(instance, BaseEmbedder):
            raise TypeError(
                f"Custom embedder '{class_path}' must inherit from BaseEmbedder."
            )
        return _ValidatedEmbedder(instance)

    if not model_name:
        raise ValueError(
            "model_name must be provided when no custom embedder class path is supplied."
        )

    return _ValidatedEmbedder(LLMEmbedder(model_name=model_name, device=device))


def _load_kwargs(custom_kwargs: Optional[Union[str, Dict[str, Any]]]) -> Dict[str, Any]:
    if isinstance(custom_kwargs, dict):
        return dict(custom_kwargs)

    payload = custom_kwargs or os.getenv("CUSTOM_CLASS_KWARGS")
    if payload in (None, ""):
        return {}

    if isinstance(payload, str):
        try:
            return json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "custom kwargs must be valid JSON when provided as a string"
            ) from exc

    raise TypeError("custom_kwargs must be a dict, JSON string, or None")


def _import_embedder_class(path: str):
    try:
        module_name, class_name = path.rsplit(".", 1)
    except ValueError as exc:
        raise ValueError(
            f"Invalid custom class path '{path}'. Expected 'module.ClassName'."
        ) from exc

    module = importlib.import_module(module_name)

    try:
        embedder_cls = getattr(module, class_name)
    except AttributeError as exc:
        raise ImportError(
            f"Module '{module_name}' has no attribute '{class_name}'."
        ) from exc

    if not issubclass(embedder_cls, BaseEmbedder):
        raise TypeError(
            f"Custom embedder '{path}' must inherit from BaseEmbedder."
        )

    return embedder_cls


def _ensure_2d_array(array: Any, *, provider: str) -> np.ndarray:
    if not isinstance(array, np.ndarray):
        raise TypeError(
            f"Embedder '{provider}' must return a numpy.ndarray; received {type(array)!r}."
        )

    if array.ndim != 2:
        raise ValueError(
            f"Embedder '{provider}' must return a 2D array with shape (N, D); got {array.shape}."
        )

    if not np.issubdtype(array.dtype, np.floating):
        array = array.astype(np.float32)

    return array


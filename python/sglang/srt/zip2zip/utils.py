import os
from typing import Optional, Union

import torch
from huggingface_hub import hf_hub_download
from zip2zip import Zip2ZipTokenizer, Zip2ZipConfig
from zip2zip.constants import SAFETENSORS_ENCODERS_NAME
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast


def load_encoders(path: str, torch_device: Optional[torch.device] = None) -> None:
    if os.path.isfile(os.path.join(path, SAFETENSORS_ENCODERS_NAME)):
        encoder_file = os.path.join(path, SAFETENSORS_ENCODERS_NAME)
    else:
        try:
            encoder_file = hf_hub_download(
                path,
                SAFETENSORS_ENCODERS_NAME,
            )
        except Exception as exc:
            raise ValueError(
                f"Can't find '{SAFETENSORS_ENCODERS_NAME}' at '{path}'"
            ) from exc


def get_zip2zip_tokenizer(
    zip2zip_path: str, tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]
) -> Zip2ZipTokenizer:
    config = Zip2ZipConfig.from_pretrained(zip2zip_path)

    return Zip2ZipTokenizer(config, tokenizer)

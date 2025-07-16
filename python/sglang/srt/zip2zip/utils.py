import os
from typing import Optional

import torch
from huggingface_hub import hf_hub_download
from zip2zip.constants import SAFETENSORS_ENCODERS_NAME

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
        
    

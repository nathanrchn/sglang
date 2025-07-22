import os
from typing import Optional, Union

from huggingface_hub import hf_hub_download
from zip2zip_compression import LZWCompressor
from zip2zip import Zip2ZipTokenizer, Zip2ZipConfig
from zip2zip.constants import SAFETENSORS_ENCODERS_NAME
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast


def get_zip2zip_tokenizer(
    zip2zip_path: str, tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]
) -> Zip2ZipTokenizer:
    config = Zip2ZipConfig.from_pretrained(zip2zip_path)

    tokenizer = Zip2ZipTokenizer(config, tokenizer)
    del config

    return tokenizer


def get_lzw_compressor(
    zip2zip_path: str, tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]
) -> LZWCompressor:
    config = Zip2ZipConfig.from_pretrained(zip2zip_path)
    compression_config = config.compression

    compressor = LZWCompressor(
        initial_vocab_size=compression_config.initial_vocab_size,
        max_codebook_size=compression_config.max_codebook_size,
        max_subtokens=compression_config.max_subtokens,
        pad_token_id=tokenizer.pad_token_id,
        disabled_ids=compression_config.disabled_ids,
    )
    del config

    return compressor

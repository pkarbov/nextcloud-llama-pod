import os
import enum
import uuid
import llama_cpp
import configparser
import multiprocessing

from loguru import logger
from typing import Optional, Any
from pydantic import BaseSettings, Field
from typing_extensions import Literal
#from typing_extensions import reveal_type

from llama_pod.settings import settings

class EngineSettings(BaseSettings):

    model: str = Field(
        default='', description="The path to the model to use for generating completions."
    )
    model_alias: Optional[str] = Field(
        default=None,
        description="The alias of the model to use for generating completions.",
    )
    n_ctx: int = Field(default=2048, ge=1, description="The context size.")
    n_gpu_layers: int = Field(
        default=0,
        ge=0,
        description="The number of layers to put on the GPU. The rest will be on the CPU.",
    )
    n_batch: int = Field(
        default=512, ge=1, description="The batch size to use per eval."
    )
    n_threads: int = Field(
        default=max(multiprocessing.cpu_count() // 2, 1),
        ge=1,
        description="The number of threads to use.",
    )
    f16_kv: bool = Field(default=True, description="Whether to use f16 key/value.")
    use_mlock: bool = Field(
        default=llama_cpp.llama_mlock_supported(),
        description="Use mlock.",
    )
    use_mmap: bool = Field(
        default=llama_cpp.llama_mmap_supported(),
        description="Use mmap.",
    )
    embedding: bool = Field(default=True, description="Whether to use embeddings.")
    low_vram: bool = Field(
        default=False,
        description="Whether to use less VRAM. This will reduce performance.",
    )
    last_n_tokens_size: int = Field(
        default=64,
        ge=0,
        description="Last n tokens to keep for repeat penalty calculation.",
    )
    logits_all: bool = Field(default=True, description="Whether to return logits.")
    cache: bool = Field(
        default=False,
        description="Use a cache to reduce processing times for evaluated prompts.",
    )
    cache_type: Literal["ram", "disk"] = Field(
        default="ram",
        description="The type of cache to use. Only used if cache is True.",
    )
    cache_size: int = Field(
        default=2 << 30,
        description="The size of the cache in bytes. Only used if cache is True.",
    )
    vocab_only: bool = Field(
        default=False, description="Whether to only return the vocabulary."
    )
    verbose: bool = Field(
        default=True, description="Whether to print debug information."
    )


    def __getitem__(self, key):
        return super().__getattribute__(key)

    def api_settings(exclude = []):
        """
        Static

        Initialize global LLaMa settings
        """
        global settings
        dict_cur = {}
        if settings :
            dict_cur = settings.dict(exclude=exclude)
        return [dict_cur]


    def update_settings(setting):
        """
        Static

        Initialize global LLaMa settings
        """
        global settings
        if settings :
            settings = setting
        return settings

    def init_settings():
        """
        Static

        Initialize global LLaMa engine settings
        """
        global settings
        if settings is None :
            settings = EngineSettings()
        return settings

    async def shutdown_settings():
        """
        Static

        Destroy global LLaMa settings
        """
        global settings
        settings = None
        return

settings = None

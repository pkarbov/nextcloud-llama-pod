"""
 - @copyright Copyright (c) 2023 Pavlo Karbovnyk <pkarbovn@gmail.com>
 -
 - @license AGPL-3.0-or-later
 -
 - This program is free software: you can redistribute it and/or modify
 - it under the terms of the GNU Affero General Public License as
 - published by the Free Software Foundation, either version 3 of the
 - License, or (at your option) any later version.
 -
 - This program is distributed in the hope that it will be useful,
 - but WITHOUT ANY WARRANTY; without even the implied warranty of
 - MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 - GNU Affero General Public License for more details.
 -
 - You should have received a copy of the GNU Affero General Public License
 - along with this program. If not, see <http://www.gnu.org/licenses/>.
 -
 """

import os
import io
import stat
import uuid
import psutil
import struct
import llama_cpp

from loguru import logger
from typing import Optional,List, Union, Any, Dict
from threading import RLock
from llama_pod.settings_engine import EngineSettings
#########################################################################
# Functions for API calls
#########################################################################

"""
- @ Scan directory find all models files
-   return list of models dict
"""

class EngineLLaMa:

    engine = None

    def __init__(self,settings):
        """
        Constructor

        Initialize LLaMa Engine
        """
        ################################################################################
        # setup engine
        self.llama_lock = RLock()
        self.llama = llama_cpp.Llama(
            model_path=settings.model,
            n_gpu_layers=settings.n_gpu_layers,
            f16_kv=settings.f16_kv,
            use_mlock=settings.use_mlock,
            use_mmap=settings.use_mmap,
            embedding=settings.embedding,
            logits_all=settings.logits_all,
            n_threads=settings.n_threads,
            n_batch=settings.n_batch,
            n_ctx=settings.n_ctx,
            last_n_tokens_size=settings.last_n_tokens_size,
            vocab_only=settings.vocab_only,
            verbose=settings.verbose,
        )
        ################################################################################
        # setup cache
        if settings.cache:
            if settings.cache_type == "disk":
                cache = llama_cpp.LlamaDiskCache(capacity_bytes=settings.cache_size)
            else:
                cache = llama_cpp.LlamaRAMCache(capacity_bytes=settings.cache_size)
            cache = llama_cpp.LlamaCache(capacity_bytes=settings.cache_size)
            self.llama.set_cache(cache)
        ################################################################################
        # Update global settings. LLaMa started successfully.
        logger.info('Memory used: {}%'.format(psutil.virtual_memory()[2]))
        EngineSettings.update_settings(settings)

    def __del__(self):
        with self.llama_lock:
            self.llama_lock = None
        self.llama = None
        return

    def api_activate_llama(settings:Dict[str, Any] = None):
        """
        Static

        Activate LLaMa engine
        """
        EngineLLaMa.engine = EngineLLaMa(settings)
        logger.info('EngineLLaMa::api_activate_llama')

    def api_deactivate_llama():
        """
        Static

        Initialize global LLaMa settings
        """
        EngineLLaMa.engine = None
        logger.info('Memory used: {}%'.format(psutil.virtual_memory()[2]))
        return True

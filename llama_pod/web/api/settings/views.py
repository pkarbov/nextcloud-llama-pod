import os

from loguru import logger
from typing import Annotated, Union
from fastapi import APIRouter, Header

from llama_pod.settings import settings
from llama_pod.settings_engine import EngineSettings
from llama_pod.web.api.settings.schema import ModelSettings


router = APIRouter()

@router.get("/", response_model=ModelSettings)
async def send_llama_model_settings(
    authorization: Annotated[Union[str, None], Header()] = None
) -> ModelSettings:
    """
    Sends echo back to user.

    :param incoming_message: incoming message.
    :returns: message same as the incoming.
    """
    exclude = {
        'model',
        'model_alias',
        'server',
        'port',
        'settings',
    }
    return {
        "data": EngineSettings.api_settings(exclude=exclude),
    }

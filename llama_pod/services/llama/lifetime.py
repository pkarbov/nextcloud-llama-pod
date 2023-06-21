import configparser

from fastapi import FastAPI

from llama_pod.settings import settings
from llama_pod.settings_engine import EngineSettings

def init_llama(app: FastAPI) -> None:  # pragma: no cover
    """
    Creates connection for llama.

    :param app: current fastapi application.
    """

    EngineSettings.init_settings()

    return

async def shutdown_llama(app: FastAPI) -> None:  # pragma: no cover
    """
    Closes llama connection.

    :param app: current FastAPI app.
    """
    await EngineSettings.shutdown_settings()

    return
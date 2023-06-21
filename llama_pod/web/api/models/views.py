import os

from fastapi import APIRouter, Header

from typing import Annotated, Union

from llama_pod.settings import settings
from llama_pod.services.llama.utils import directory_scan
from llama_pod.web.api.models.schema import ModelList, ModelData


router = APIRouter()


@router.get("/", response_model=ModelList)
async def send_llama_model_list(
    authorization: Annotated[Union[str, None], Header()] = None
) -> ModelList:
    """
    Sends echo back to user.

    :param incoming_message: incoming message.
    :returns: message same as the incoming.
    """
    ls_res = []
    if os.path.isdir(settings.llama_path) :
        directory_scan(settings.llama_path, ls_res)
    return {
        "data": ls_res,
    }

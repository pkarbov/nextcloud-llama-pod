import os

from loguru import logger
from fastapi import APIRouter, HTTPException, status

from llama_pod.settings_engine import EngineSettings
from llama_pod.services.llama.engine_llama import EngineLLaMa
from llama_pod.web.api.models.schema import ModelData
from llama_pod.web.api.activate.schema import SettingModelData

router = APIRouter()

@router.post("/")
async def activate_model(
    incoming_data: SettingModelData,
) -> str:
    """
    Sends echo back to user.

    :param incoming_message: incoming message.
    :returns: message same as the incoming.
    """
    #if item_id not in items:
    #    raise HTTPException(status_code=404, detail="Item not found")
    # logger.info(incoming_data)
    res = None
    exclude = lambda d, keys: {x: d[x] for x in d if x not in keys}
    if incoming_data.model.active :
        try:
            model    = ModelData.parse_obj(incoming_data.model)
            settings = EngineSettings.parse_obj(incoming_data.setting)
            settings.model = os.path.join(model.path,model.name).replace('\\','')
            # load model
            EngineLLaMa.api_activate_llama(settings)
            res = 'Activated'
        except Exception as ex:
            logger.info(str(ex))
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(ex))
    else:
        EngineLLaMa.api_deactivate_llama()
        res = 'Deactivated'

    return res


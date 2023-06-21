from typing import Optional,List, Union, Any, Dict
from pydantic import BaseModel

from llama_pod.web.api.models.schema import ModelData
from llama_pod.web.api.settings.schema import ModelSettings

class SettingModelData(BaseModel):
    """Simple message model."""
    model: ModelData
    setting: Optional[Dict[str, Any]] = None

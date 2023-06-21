from typing import List, Union, Any, Dict
from pydantic import BaseModel

# Type for: [] -> empty list | [{'key': Any, ...}, ..] -> list with dictionaries
class ModelSettings(BaseModel):
    data: List[Union[Dict[str, Any],None]]


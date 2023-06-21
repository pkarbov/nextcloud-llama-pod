from typing import List
from pydantic import BaseModel

class ModelData(BaseModel):
    """Model data."""
    id   : str
    name : str
    path : str
    size : int
    vocab: int
    embd : int
    mult : int
    head : int
    layer: int
    rot  : int
    active   : bool
    file_type: int

class ModelList(BaseModel):
    """Model list."""
    data:   List[ModelData]


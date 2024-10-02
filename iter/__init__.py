import os

from .configuration_iter import ITERConfig
from .modeling_iter2 import ITER, ITERForRelationExtraction

#use_iter_2 = os.environ.get("ITER_2", "false") == "true"
#if not use_iter_2:
#    from .modeling_iter import ITER
#    ITERForRelationExtraction = ITER

__all__ = ["ITER", "ITERForRelationExtraction", "ITERConfig"]

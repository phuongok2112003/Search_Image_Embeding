from enum import Enum
from .openai_clip import OpenAIClipModel
from .laion_clip import LaionClipModel

class ModelType(Enum):
    OPENAI_CLIP = "OpenAIClipModel"
    LAION_CLIP = "LaionClipModel"

MODEL_CLASS_MAP = {
    ModelType.OPENAI_CLIP: OpenAIClipModel,
    ModelType.LAION_CLIP: LaionClipModel,
}

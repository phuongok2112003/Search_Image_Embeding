from enum import Enum
from .openai_clip import OpenAIClipModel
from .laion_clip import LaionClipModel
from .multilingual_clip import MultilingualClipModel

class ModelType(str, Enum):
    OPENAI_CLIP = "openai_clip"
    LAION_CLIP = "laion_clip"
    MULTILINGUAL_CLIP = "multilingual_clip"

MODEL_CLASS_MAP = {
    ModelType.OPENAI_CLIP: OpenAIClipModel,
    ModelType.LAION_CLIP: LaionClipModel,
    ModelType.MULTILINGUAL_CLIP: MultilingualClipModel,
}

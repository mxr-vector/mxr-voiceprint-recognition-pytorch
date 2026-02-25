from .voiceprint_service import singleVoiceprintService
from .swallow_service import singleSwallowService
from .intent_service import singleIntentService

__version__ = "1.0.0"
__author__ = "YuanJie"
__all__ = ["singleVoiceprintService", "singleSwallowService", "singleIntentService"]
print("package {} is imported!".format(__package__))

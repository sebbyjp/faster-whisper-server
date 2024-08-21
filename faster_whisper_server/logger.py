import logging

from faster_whisper_server.audio_config import config as audio_config

# Disables all but `faster_whisper_server` logger

root_logger = logging.getLogger()
root_logger.setLevel(logging.CRITICAL)
logger = logging.getLogger(__name__)
logger.setLevel(audio_config.log_level.upper())
logging.basicConfig(format="%(asctime)s:%(levelname)s:%(name)s:%(funcName)s:%(message)s")

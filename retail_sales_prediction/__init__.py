import logging
import os

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
MODULE_NAME = os.path.basename(MODULE_DIR)
PROJECT_DIR = os.path.abspath(os.path.join(MODULE_DIR, '..'))
PROJECT_NAME = os.path.basename(PROJECT_DIR)

logger = logging.getLogger(MODULE_NAME)

# -*- coding: utf-8 -*-

import logging
import logging.config

logging.config.fileConfig("logger/logger.conf")
logger = logging.getLogger("root")
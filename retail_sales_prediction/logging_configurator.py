import logging
import sys

from retail_sales_prediction import MODULE_NAME


class LoggingConfigurator(object):
    """
    A singleton to handle setup and teardown of logging.

    Logging can be done using log file and also displays the messages on console

    """

    self_ = None

    def __init__(self, file_name=None):
        self.formatter = logging.Formatter(
            fmt="%(asctime)-15s %(levelname)s  [%(name)s] %(filename)s:%(lineno)s - %(message)s",
            datefmt='%Y-%m-%d %H:%M:%S')

        self.console_handler = logging.StreamHandler(sys.stdout)
        self.console_handler.setFormatter(self.formatter)

        if file_name is not None:
            self.file_handler = logging.FileHandler(file_name)
            self.file_handler.setFormatter(self.formatter)
        self.root_logger = logging.getLogger()
        self.root_logger.addHandler(self.console_handler)
        self.root_logger.addHandler(self.file_handler)
        self.root_logger.setLevel(logging.DEBUG)

        # py4j is very noisy
        logging.getLogger("py4j").setLevel(logging.WARN)

        self.module_logger = logging.getLogger(MODULE_NAME)
        self.module_logger.setLevel(logging.DEBUG)

    @staticmethod
    def init(file_name=None):
        if LoggingConfigurator.self_ is None:
            if file_name is not None:
                LoggingConfigurator.self_ = LoggingConfigurator(file_name)

        return LoggingConfigurator.self_

    @staticmethod
    def shutdown():
        LoggingConfigurator.self = None
        logging.shutdown()

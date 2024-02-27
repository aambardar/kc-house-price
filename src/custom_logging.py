import logging
import inspect


class CustomLogger:
    logs = []

    def __init__(self, logger_name):
        self.logger = logging.getLogger(logger_name)
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)

        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(stream_handler)

    def debug(self, message):
        self.logger.debug(f'\033[36m{message}\033[0m')  # Cyan for DEBUG

    def info(self, message):
        log_invoker = inspect.stack()[1][3]
        self.logger.info(f'\033[90m{log_invoker}\033[0m - \033[32m{message}\033[0m')  # Green for INFO

    def warning(self, message):
        self.logger.warning(f'\033[33m{message}\033[0m')  # Yellow for WARNING

    def error(self, message):
        self.logger.error(f'\033[31m{message}\033[0m')  # Red for ERROR

    def critical(self, message):
        self.logger.critical(f'\033[41m{message}\033[0m')  # Background Red for CRITICAL

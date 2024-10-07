import logging


class Logs:

  def __init__(self, level: logging._Level, message: logging.message):
    super(Logs, self).__init__()

    return logging.basicConfig(level=level, message=message, format='%(asctime)s - %(levelname)s - %(message)s')

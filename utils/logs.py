import logging


class Logs:

  def __init__(self, level: int, message: str):

    super(Logs, self).__init__()

    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')
    self.logger = logging.getLogger()
    self.logger.setLevel(level)
    self.logger.log(level, message)


  def get_logger(self):
    return self.logger

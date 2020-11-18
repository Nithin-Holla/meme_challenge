import logging

_LOG_FMT = '%(asctime)s : %(levelname)s -   %(message)s'
_DATE_FMT = '%d/%m/%Y %I:%M:%S %p'
logging.basicConfig(format=_LOG_FMT, datefmt=_DATE_FMT, level=logging.INFO)
LOGGER = logging.getLogger('TrainLogger')
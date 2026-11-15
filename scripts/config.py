"""Application configuration and shared constants."""
import logging
import warnings

DEBUG = False
LOC = "Home"


suppress_logging = True


if LOC == "Work":
    DATA_DIR = r"\\storage.erasmusmc.nl\v\vcl02\ONCO\Data\01 Onderzoeksbestanden\STAPPS\STAPPS_Main\STAPPS\Data"
    PASSWORD = r"U:\password.txt"
    DATA_STRUCTURE = r"\\storage.erasmusmc.nl\v\vcl02\ONCO\Data\01 Onderzoeksbestanden\STAPPS\STAPPS_Main\STAPPS\structure.json"
    FIGURE_FOLDER = r"\\storage.erasmusmc.nl\v\vcl02\ONCO\Data\01 Onderzoeksbestanden\STAPPS\STAPPS_Main\STAPPS\plots\figures"
elif LOC == "Home":
    import config2
    DATA_DIR = config2.DATA_DIR
    PASSWORD = config2.PASSWORD
    DATA_STRUCTURE = config2.DATA_STRUCTURE
    FIGURE_FOLDER = config2.FIGURE_FOLDER

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(levelname)-8s | "
        "%(filename)s:%(lineno)d | "
        "%(funcName)s | "
        "%(message)s"
    ))
logging.basicConfig(
    level=logging.WARNING,
    format=(
        "%(levelname)-8s | "
        "%(filename)s:%(lineno)d | "
        "%(funcName)s | "
        "%(message)s"
    ))

if suppress_logging:

    class LogFilter(logging.Filter):
        def filter(self, record):
            return record.levelno in (
                logging.INFO,
                logging.ERROR,
                logging.CRITICAL,
            )

    root = logging.getLogger()

    for handler in root.handlers:
        handler.addFilter(LogFilter())

    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=UserWarning)




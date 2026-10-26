"""Application configuration and shared constants."""
import logging

DEBUG = False
LOC = "Home"

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

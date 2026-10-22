"""Application configuration and shared constants."""
import logging

DEBUG = False
LOC = "Work"
if LOC == "Home":
    DATA_DIR = "E:\\Data\\Zip-files pilot patienten"
    PASSWORD = "E:\OneDrive\School\Technical Medicine\TM Jaar 3\Medical Oncology - STAPPS\STAPPS\password.txt"
    DATA_STRUCTURE = r"\\storage.erasmusmc.nl\v\vcl02\ONCO\Data\01 Onderzoeksbestanden\STAPPS\STAPPS_Main\STAPPS\structure.json"
    FIGURE_FOLDER = r"\\storage.erasmusmc.nl\v\vcl02\ONCO\Data\01 Onderzoeksbestanden\STAPPS\STAPPS_Main\STAPPS\plots\figures"

elif LOC == "Work":
    DATA_DIR = r"\\storage.erasmusmc.nl\v\vcl02\ONCO\Data\01 Onderzoeksbestanden\STAPPS\STAPPS_Main\STAPPS\Data"
    PASSWORD = r"U:\password.txt"
    DATA_STRUCTURE = r"\\storage.erasmusmc.nl\v\vcl02\ONCO\Data\01 Onderzoeksbestanden\STAPPS\STAPPS_Main\STAPPS\structure.json"
    FIGURE_FOLDER = r"\\storage.erasmusmc.nl\v\vcl02\ONCO\Data\01 Onderzoeksbestanden\STAPPS\STAPPS_Main\STAPPS\plots\figures"


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
    )
)

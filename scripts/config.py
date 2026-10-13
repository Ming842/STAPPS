"""Application configuration and shared constants."""

DEBUG = False
LOC = "Work"

if LOC == "Home":
    DATA_DIR = "E:\\Data\\Zip-files pilot patienten"
    PASSWORD = "E:\OneDrive\School\Technical Medicine\TM Jaar 3\Medical Oncology - STAPPS\STAPPS\password.txt"
elif LOC == "Work":
    DATA_DIR = r"\\storage.erasmusmc.nl\v\vcl02\ONCO\Data\01 Onderzoeksbestanden\STAPPS\STAPPS_Main\STAPPS\Data"
    PASSWORD = r"U:\password.txt"
DATA_STRUCTURE = "structure.json"

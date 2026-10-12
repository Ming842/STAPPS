"""Application configuration and shared constants."""

DEBUG = False
LOC = "Home"

if LOC == "Home":
    DATA_DIR = "E:\\Data\\Zip-files pilot patienten"
    PASSWORD = "E:\OneDrive\School\Technical Medicine\TM Jaar 3\Medical Oncology - STAPPS\STAPPS\password.txt"
elif LOC == "Work":
    DATA_DIR = "C:\\Users\\116271\\Documents\\Data"
    PASSWORD = "C:\\Users\\116271\\Documents\\Data\\password.txt"
DATA_STRUCTURE = "structure.json"

"""Application configuration and shared constants."""

DEBUG = False
LOC = "Home"

if LOC == "Home":
    DATA_DIR = "E:\\Data\\Zip-files pilot patienten"
elif LOC == "Work":
    DATA_DIR = "C:\\Users\\116271\\Documents\\Data"
DATA_STRUCTURE = "structure.json"

"""Generate a small synthetic test-data table."""

import pandas as pd


test_data = pd.DataFrame(
	[
		{"patient_id": "P001", "age": 54, "sex": "F", "tumor_type": "Breast", "stage": "II", "treatment": "Chemotherapy", "response": "Partial"},
		{"patient_id": "P002", "age": 67, "sex": "M", "tumor_type": "Lung", "stage": "III", "treatment": "Immunotherapy", "response": "Stable"},
		{"patient_id": "P003", "age": 42, "sex": "F", "tumor_type": "Colon", "stage": "I", "treatment": "Surgery", "response": "Complete"},
		{"patient_id": "P004", "age": 73, "sex": "M", "tumor_type": "Prostate", "stage": "IV", "treatment": "Hormone therapy", "response": "Progressive"},
		{"patient_id": "P005", "age": 61, "sex": "F", "tumor_type": "Ovarian", "stage": "III", "treatment": "Chemotherapy", "response": "Partial"},
	]
)


if __name__ == "__main__":
	print(test_data.to_string(index=False))


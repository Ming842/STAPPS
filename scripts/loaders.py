"""Module for loading participant data from the Fitbit export and Cycle event Excel."""
import re
import os
import msoffcrypto

import pandas as pd
from typing import Any
from pathlib import Path

from config import DATA_DIR, PASSWORD
from parsers import parse_csv, parse_json
from structure_loader import StructureDataLoader

class ParticipantDataLoader():
    """Class to load and manage participant data from the Fitbit export and Cycle data."""
    def __init__(self, data_dir: str | Path = DATA_DIR, password: str = PASSWORD) -> None:
        self.data_dir = Path(data_dir)
        self.password = Path(password)
        self.pat_ids, self.pat_dirs = self._get_pt_ids_and_dirs(self.data_dir)

    def _get_pt_ids_and_dirs(self, path: Path) -> tuple[list[int], list[Path]]:
        """Return a list of files and directories in the given path."""
        if not path.exists():
            raise FileNotFoundError(f"The path {path} does not exist.")

        # Extract participant IDs from entries like "ST02" -> 2
        pt_ids: list[int] = []
        pt_dirs: list[Path] = []
        for item in path.iterdir():
            m = re.match(r"^ST(\d+)", item.name)
            if m:
                pt_ids.append(int(m.group(1)))
                pt_dirs.append(item)
        return pt_ids, pt_dirs

    def _format_cycles(self, cycles_data: pd.DataFrame, symptoms: bool) -> pd.DataFrame:
        """Format the cycles data."""
        # Placeholder for actual implementation
        count_cols = ["ED_Visits", "Hospitalizations", "Contact_HCP"]
        bool_cols = ["Dose_Modifications", "Chemo Postponed"]
        df = cycles_data.copy()
        if symptoms:
            # --- Count columns: "1 keer" -> 1, "2 keer" -> 2 ---
            for col in count_cols:
                df[col] = (
                    df[col]
                    .astype("string")
                    .str.extract(r"(\d+)\s*keer", flags=re.IGNORECASE)[0]
                    .pipe(pd.to_numeric, errors="coerce")
                    .astype(object)
                    .fillna(False)
                )

            # --- Boolean columns: any non-empty text -> True ---
            for col in bool_cols:
                values = df[col].astype("string").str.strip()
                df[col] = (
                    values.notna()
                    & values.ne("")
                    & values.ne("None.")
                )
        else:
            # If symptoms is False, set count columns to 0 and boolean columns to False
            for col in count_cols:
                df[col] = False
            for col in bool_cols:
                df[col] = False
        return df


    def make_file_paths(self, participant_id: int, dpath: Path, dtype: str) -> dict[str, Any]:
        """Make a file path for a given participant ID and data type."""
        if participant_id not in self.pat_ids:
            raise ValueError(f"Participant ID {participant_id} not found.")
        if len(dtype) == 0:
            raise ValueError(f"Data type must be specified.\n Choose from: {dtype}")

        index = self.pat_ids.index(participant_id)
        participant_dir = self.pat_dirs[index]

        path = self.data_dir.joinpath(participant_dir, dpath)
        if not path.exists():
            #change Fitbit in path to Google Health
            dpath = Path(*(
                "Google Health" if part == "Fitbit" else part
                for part in dpath.parts
            ))

            path = self.data_dir.joinpath(participant_dir, dpath)

        pattern = r"-\s*YYYY-MM-DD"
        if any(re.search(pattern, filename) for filename in dtype):
            base_names = [
                    re.sub(r"\s*-\s*YYYY-MM-DD(?:\.(?:csv|txt|xlsx|json))?$", "", filename)
                    for filename in dtype
                ]

            file_paths = [
                p
                for p in path.iterdir()
                if any(base_name in p.name for base_name in base_names)
            ]
        else:
            file_paths = [
                p
                for p in path.iterdir()
                if any(filename in p.name for filename in dtype)
            ]
        return file_paths

    def load_files(self, file_paths: list[Path]) -> list[dict[str, Any]]:
        """Load data from the given file paths and return a list of dictionaries."""
        loaded_data: list[dict[str, Any]] = []
        for file_path in file_paths:
            if not file_path.exists():
                raise FileNotFoundError(f"The file {file_path} does not exist.")
            if file_path.suffix == ".csv":
                loaded_data.extend(parse_csv(file_path))
            elif file_path.suffix == ".json":
                loaded_data.append(parse_json(file_path))
            elif file_path.suffix == ".txt":
                #only README files are txt, so they are ignored
                pass
            else:
                raise ValueError(f"Unsupported file type: {file_path.suffix}")
        return loaded_data

    def cycle_data(self) -> dict[str, Any]:
        """Load cycles data."""

        # Read password, strip newline/whitespace
        excel_path = Path(os.path.join(self.data_dir, "treatment_data.xlsx"))
        decrypted_path = Path(os.path.join(self.data_dir, "treatment_data_decrypted.xlsx"))
        password_path = Path(self.password)

        password = password_path.read_text(encoding="utf-8").strip()

        with excel_path.open("rb") as encrypted_file:
            office_file = msoffcrypto.OfficeFile(encrypted_file)
            office_file.load_key(password=password)

            with decrypted_path.open("wb") as decrypted_file:
                office_file.decrypt(decrypted_file)
        sheets = pd.read_excel(decrypted_path, sheet_name=None)

        #remove the decrypted file after reading
        decrypted_path.unlink()
        events = sheets["Cycle_Events"]

        #set admission time at 9:00 AM
        daily_mask = events["Date"].eq("Dagelijks")
        events.loc[~daily_mask, "Date"] = pd.to_datetime(
            events.loc[~daily_mask, "Date"], errors="coerce"
        )
        chemo_admmission = events.loc[events["Date"].notna()].copy()
        datetime_mask = pd.to_datetime(
            chemo_admmission["Date"], errors="coerce"
        ).notna()
        chemo_admmission.loc[datetime_mask, "Date"] += pd.Timedelta(hours=9)

        # change all entries with date to datetime, and keep the rest as is
        events.loc[:, "Date"] = chemo_admmission["Date"]
        data = pd.DataFrame()
        for _, group in events.groupby("PatientID"):
            baseline_rows = group[group["Period"] == "Baseline"]
            if baseline_rows["Bijwerking"].eq("wel").any():
                cycles = self._format_cycles(group, True)
                cycles["Bijwerking"] = True
            else:
                cycles = self._format_cycles(group, False)
                cycles["Bijwerking"] = False
            data = pd.concat([data, cycles], ignore_index=True)

        # set PatientID to integer
        data["participant_id"] = data["PatientID"].str.extract(r"(\d+)").astype(int)

        # move participant_id to the first column
        cols = data.columns.tolist()
        cols.insert(0, cols.pop(cols.index("participant_id")))
        data = data[cols]
        
        data.drop(columns=["PatientID"], inplace=True)
        return data

if __name__ == "__main__":
    loader = StructureDataLoader()
    participant_loader = ParticipantDataLoader()

    results_sleep_score = loader.find_paths("sleep_score")
    data1 = participant_loader.cycle_data()

    data = {}
    for parent_path, files in results_sleep_score:
        for part_id in participant_loader.pat_ids:
            print(f"Participant ID: {part_id}, Parent Path: {parent_path}, Files: {files}")
            paths = participant_loader.make_file_paths(part_id, parent_path, files)
            data[part_id] = participant_loader.load_files(paths)

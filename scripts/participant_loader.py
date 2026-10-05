"""Module for loading participant data from the Fitbit export."""

# import Any

import re

from typing import Any
from pathlib import Path

from config import DATA_DIR
from parsers import parse_csv, parse_json, parse_txt
from structure_loader import StructureDataLoader


class ParticipantDataLoader():
    """Class to load and manage participant data from the Fitbit export."""
    def __init__(self, data_dir: str | Path = DATA_DIR) -> None:
        self.data_dir = Path(data_dir)
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

if __name__ == "__main__":
    loader = StructureDataLoader()
    participant_loader = ParticipantDataLoader()

    results_sleep_score = loader.find_paths("sleep_score")
    data = {}
    for parent_path, files in results_sleep_score:
        for part_id in participant_loader.pat_ids:
            print(f"Participant ID: {part_id}, Parent Path: {parent_path}, Files: {files}")
            paths = participant_loader.make_file_paths(part_id, parent_path, files)
            data[part_id] = participant_loader.load_files(paths)

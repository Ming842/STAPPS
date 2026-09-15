"""Utilities for discovering and loading the Fitbit export data tree."""

from __future__ import annotations

import csv
import json
import re
import os

from matplotlib import pyplot as plt
from pathlib import Path
from typing import Any

from config import DATA_DIR, DATA_STRUCTURE

class StructureDataLoader:
    """Class to load and manage the structure.json file for Fitbit data."""

    def __init__(self, data_dir: str | Path = DATA_DIR,
                 structure_path: str | Path = DATA_STRUCTURE) -> None:
        self.data_dir = Path(data_dir)
        self.fitbit_dir = Path()
        self.structure_path = Path(structure_path)
        self.structure = self._read_structure()

    def _read_structure(self) -> dict[str, Any]:
        """Read the structure.json file and return its contents as a dictionary."""

        with self.structure_path.open("r", encoding="utf-8-sig") as data:
            return json.load(data)

    def find_paths(self, file_name: str) -> list[tuple[Path, list[str]]]:
        """Find all paths of names to a target node name in structure.json."""
        def _search_structure(
                structure: dict[str, Any],
                target_name: str,
                ancestors: list[str] | None = None,
            ) -> list[tuple[list[str], list[str]]]:
                """Recursively search and return all matching full paths for target_name."""

                if ancestors is None:
                    ancestors = []

                matches: list[tuple[list[str], list[str]]] = []
            
                current_name = structure.get("name")
                current_path = ancestors + ([current_name] if current_name else [])

                if current_name == target_name:
                    matches.append((current_path, structure.get("files", [])))
                else:
                    for f in structure.get("files", []):
                        if target_name in f:
                            matches.append((current_path, [f]))
                
                for child in structure.get("children", []):
                    child_matches = _search_structure(child, target_name, current_path)
                    if child_matches:
                        matches.extend(child_matches)

                return matches

        matches = _search_structure(self.structure, file_name)
        results = [(self.fitbit_dir.joinpath(*parents), files) for parents, files in matches]
        return results

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
                    re.sub(r"\s*-\s*YYYY-MM-DD(?:\.(?:csv|txt|xlsx))?$", "", filename)
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
                with file_path.open("r", encoding="utf-8-sig") as f:
                    reader = csv.DictReader(f)
                    loaded_data.extend(list(reader))
            elif file_path.suffix == ".json":
                with file_path.open("r", encoding="utf-8-sig") as f:
                    loaded_data.append(json.load(f))
            else:
                raise ValueError(f"Unsupported file type: {file_path.suffix}")
        return loaded_data


        
if __name__ == "__main__":
    loader = StructureDataLoader()
    participant_loader = ParticipantDataLoader()

    results = loader.find_paths("sleep_score")
    data = {}
    for parent_path, files in results:
        for participant_id in participant_loader.pat_ids:
            print(f"Participant ID: {participant_id}, Parent Path: {parent_path}, Files: {files}")
            paths = participant_loader.make_file_paths(participant_id, parent_path, files)
            data[participant_id] = participant_loader.load_files(paths)

    


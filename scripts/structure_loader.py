"""Utilities for discovering and loading the Fitbit export data tree."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from config import DATA_DIR, DATA_STRUCTURE

class StructureDataLoader:
    """Class to load and manage the structure.json file for Fitbit data."""

    def __init__(self, data_dir: str | Path = DATA_DIR,
                 structure_path: str | Path = DATA_STRUCTURE) -> None:
        """Initialize the structure loader.

        Args:
            data_dir (str | Path): Root folder containing participant data.
            structure_path (str | Path): Path to the JSON structure definition file.

        Returns:
            None: Initializes the loader and reads the structure tree.
        """
        self.data_dir = Path(data_dir)
        self.fitbit_dir = Path()
        self.structure_path = Path(structure_path)
        self.structure = self._read_structure()

    def _read_structure(self) -> dict[str, Any]:
        """Read the structure.json file and return its contents as a dictionary.

        Returns:
            dict[str, Any]: Parsed structure definition from the JSON file.
        """
        with self.structure_path.open("r", encoding="utf-8-sig") as structure_file:
            return json.load(structure_file)

    def find_paths(self, file_name: str) -> list[tuple[Path, list[str]]]:
        """Find all paths of names to a target node name in structure.json.

        Args:
            file_name (str): Name or regex-like pattern to match within the structure.

        Returns:
            list[tuple[Path, list[str]]]: Matching parent paths and file names.
        """
        pattern = re.compile(file_name)

        def _search_structure(
                structure: dict[str, Any],
                target_pattern: re.Pattern[str],
                ancestors: list[str] | None = None,
        ) -> list[tuple[list[str], list[str]]]:
            """Recursively search and return all matching full paths for target_pattern.

            Args:
                structure (dict[str, Any]): Current structure node being inspected.
                target_pattern (re.Pattern[str]): Regex used to match the current node or files.
                ancestors (list[str] | None): Parent path names already traversed.

            Returns:
                list[tuple[list[str], list[str]]]: Matching names and file lists beneath the current node.
            """
            if ancestors is None:
                ancestors = []

            matches: list[tuple[list[str], list[str]]] = []

            current_name = structure.get("name")
            current_path = ancestors + ([current_name] if current_name else [])

            if current_name and target_pattern.search(current_name):
                matches.append((current_path, structure.get("files", [])))
            else:
                for f in structure.get("files", []):
                    if target_pattern.search(f):
                        matches.append((current_path, [f]))

            for child in structure.get("children", []):
                child_matches = _search_structure(child, target_pattern, current_path)
                if child_matches:
                    matches.extend(child_matches)

            return matches

        matches = _search_structure(self.structure, pattern)
        results = [(self.fitbit_dir.joinpath(*parents), files) for parents, files in matches]
        return results

if __name__ == "__main__":
    loader = StructureDataLoader()
    paths = loader.find_paths("heart_rate")
    for path, files in paths:
        print(f"Path: {path}, Files: {files}")

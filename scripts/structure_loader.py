"""Utilities for discovering and loading the Fitbit export data tree."""

from __future__ import annotations

import json

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

        with self.structure_path.open("r", encoding="utf-8-sig") as structure_file:
            return json.load(structure_file)

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



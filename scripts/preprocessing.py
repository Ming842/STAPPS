"""Preprocessing module for data files in STAPPS."""
import pandas as pd

from config import DATA_DIR
from participant_loader import ParticipantDataLoader
from structure_loader import StructureDataLoader


class Preprocessor:
    """Class to handle preprocessing of data files."""

    def __init__(self):
        self.data_dir = DATA_DIR
        self.loader = StructureDataLoader()
        self.participant_loader = ParticipantDataLoader()

    def _process_data(self, data_type: str) -> pd.DataFrame:
        """Load data for all participants and return it in long format."""
        frames = []

        for parent_path, files in self.loader.find_paths(data_type):
            for part_id in self.participant_loader.pat_ids:
                paths = self.participant_loader.make_file_paths(
                    part_id,
                    parent_path,
                    files,
                )
                loaded_data = self.participant_loader.load_files(paths)
                if data_type == "steps":
                    for records in loaded_data:
                        frame = pd.DataFrame(records, columns=["dateTime", "value"])
                        frame["dateTime"] = pd.to_datetime(
                            frame["dateTime"],
                            format="%m/%d/%y %H:%M:%S"
                        )
                        # Convert value from text to numbers
                        frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
                        frame.insert(0, "participant_id", part_id)
                        frames.append(frame)
                else:
                    participant_data = pd.DataFrame(loaded_data)
                    participant_data.insert(0, "participant_id", part_id)
                    frames.append(participant_data)

        if not frames:
            return pd.DataFrame()

        return pd.concat(frames, ignore_index=True)

    def process_heart_rate(self) -> pd.DataFrame: 
        """Process heart rate data as long as it is available."""
        heart_rate_data = self._process_data("heart_rate")
        heart_rate_data.sort_values(by=["participant_id", "dateTime"], inplace=True)

        return heart_rate_data
    
    def process_heart_rate_variability(self) -> pd.DataFrame:
        """Process HRV data as long as it is available."""
        hrv_data = self._process_data("Heart Rate Variability Summary")
        hrv_data.insert(1, "dateTime", pd.to_datetime(
                            hrv_data["timestamp"],
                            format="ISO8601"
                        ))
        hrv_data.pop("timestamp")

        #sort data by participant_id and then by dateTime
        hrv_data.sort_values(by=["participant_id", "dateTime"], inplace=True)
        hrv_data.reset_index(drop=True, inplace=True)
        
        return hrv_data

    def process_sleep_score(self) -> pd.DataFrame:
        """Process Sleep Score data as long as it is available."""
        return self._process_data("sleep_score")

    def process_step_count(self) -> pd.DataFrame:
        """Process Step Count data as long as it is available."""
        return self._process_data("steps")

if __name__ == "__main__":
    preprocessor = Preprocessor()
    hrv_data = preprocessor.process_heart_rate_variability()
    sleep_data = preprocessor.process_sleep_score()
    steps_data = preprocessor.process_step_count()
    print(hrv_data.head())
    print(sleep_data.head())
    print(steps_data.head())

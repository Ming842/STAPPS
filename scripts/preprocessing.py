"""Preprocessing module for data files in STAPPS."""
import pandas as pd

from config import DATA_DIR
from loaders import ParticipantDataLoader
from structure_loader import StructureDataLoader

class Preprocessor:
    """Class to handle preprocessing of data files."""

    def __init__(self):
        self.data_dir = DATA_DIR
        self.loader = StructureDataLoader()
        self.participant_loader = ParticipantDataLoader()

    def _process_data(self, data_type: str, part_id: int = None) -> pd.DataFrame:
        """Load data for all participants and return it in long format.
        If part_id is provided, only load data for that participant.
        Args:
            data_type (str): The type of data to process (e.g., "steps", "heart_rate").
            part_id (int, optional): The participant ID to filter the data. Defaults to None.
        Returns:
            pd.DataFrame: A DataFrame containing the processed data.
        """
        frames = []

        for parent_path, files in self.loader.find_paths(data_type):
            for id in (part_id if part_id is not None else self.participant_loader.pat_ids):
                paths = self.participant_loader.make_file_paths(
                    id,
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
                        frame.insert(0, "participant_id", id)
                        frames.append(frame)

                elif data_type == "Heart Rate Variability Summary":
                    for records in loaded_data:
                        frame = pd.DataFrame(records, columns=["timestamp", "rmssd"])
                        frame["dateTime"] = pd.to_datetime(
                            frame["dateTime"],
                            format="ISO8601"
                        )
                        # Convert rmssd from text to numbers
                        frame["rmssd"] = pd.to_numeric(frame["rmssd"], errors="coerce")
                        frame.insert(0, "participant_id", id)
                        frames.append(frame)
                else:
                    participant_data = pd.DataFrame(loaded_data)
                    participant_data.insert(0, "participant_id", id)
                    frames.append(participant_data)

        if not frames:
            return pd.DataFrame()

        return pd.concat(frames, ignore_index=True)

    def process_heart_rate(self, part_id: int = None) -> pd.DataFrame: 
        """Process heart rate data as long as it is available."""
        heart_rate_data = self._process_data("heart_rate", part_id)
        heart_rate_data.sort_values(by=["participant_id", "dateTime"], inplace=True)

        return heart_rate_data

    def process_heart_rate_variability(self, part_id: int = None) -> pd.DataFrame:
        """Process HRV data as long as it is available."""
        hrv_data = self._process_data("Heart Rate Variability Summary", part_id)
        hrv_data.insert(1, "dateTime", pd.to_datetime(
                            hrv_data["timestamp"],
                            format="ISO8601"
                        ))
        hrv_data.pop("timestamp")

        #sort data by participant_id and then by dateTime
        hrv_data.sort_values(by=["participant_id", "dateTime"], inplace=True)
        hrv_data.reset_index(drop=True, inplace=True)

        return hrv_data

    def process_sleep_score(self, part_id: int = None) -> pd.DataFrame:
        """Process Sleep Score data as long as it is available."""
        return self._process_data("sleep_score", part_id)

    def process_step_count(self, part_id: int = None) -> pd.DataFrame:
        """Process Step Count data as long as it is available."""
        step_data = self._process_data("steps", part_id)

        # day column for grouping
        step_data['date'] = pd.to_datetime(step_data['dateTime']).dt.date
        
        for idx, stepdata in step_data.groupby("participant_id"):
            stepdata = stepdata.sort_values("dateTime")

            # Daily totals
            daily_totals = (
                stepdata.groupby("date")["value"]
                .sum()
                .sort_index()
            )

            # 1- to 7-day moving averages of daily totals
            daily_ma = {
                window: daily_totals.rolling(window=window, min_periods=1).mean()
                for window in range(2, 8)
            }

            for date, day_data in stepdata.groupby("date"):
                mask = (
                    (step_data["participant_id"] == idx) &
                    (step_data["date"] == date)
                )

                # Total steps of the day (repeated for each row of that day)
                step_data.loc[mask, "daily_steps"] = daily_totals.loc[date]
                # 1- to 7-day moving averages (repeated for each row of that day)
                for window, ma in daily_ma.items():
                    step_data.loc[mask, f"daily_steps_ma{window}"] = ma.loc[date]

                # Cumulative steps within the day
                step_data.loc[mask, "cumsum"] = day_data["value"].cumsum().values
        return step_data

if __name__ == "__main__":
    preprocessor = Preprocessor()
    hrv_data = preprocessor.process_heart_rate_variability()
    sleep_data = preprocessor.process_sleep_score()
    steps_data = preprocessor.process_step_count()
    print(hrv_data.head())
    print(sleep_data.head())
    print(steps_data.head())

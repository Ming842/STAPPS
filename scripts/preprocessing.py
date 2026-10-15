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
                        records["dateTime"] = pd.to_datetime(
                            records["timestamp"],
                            format="ISO8601"
                        )
                        records.pop("timestamp")
                        frame = pd.DataFrame({
                            "dateTime": [records["dateTime"]],
                            "rmssd": [records["rmssd"]],
                        })

                        # Convert rmssd from text to numbers
                        frame["rmssd"] = pd.to_numeric(frame["rmssd"], errors="coerce")
                        frame[frame["rmssd"] == 0] = pd.NA

                        frame.insert(0, "participant_id", id)
                        frames.append(frame)
                elif data_type == "resting_heart_rate":
                    for records in loaded_data:
                        records = pd.DataFrame(records)
                        records["dateTime"] = pd.to_datetime(
                            records["value"].str["date"],
                            format="%m/%d/%y",
                            errors="coerce"
                        )
                        frame = pd.DataFrame({
                            "dateTime": records["dateTime"],
                            "value": records["value"].str["value"],
                            "error": records["value"].str["error"],
                        })

                        for col in ["value", "error"]:
                            frame[col] = pd.to_numeric(frame[col], errors="coerce")

                        frame = frame[frame["value"] != 0]                     
                        
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

        #sort data by participant_id and then by dateTime
        hrv_data.sort_values(by=["participant_id", "dateTime"], inplace=True)
        hrv_data.reset_index(drop=True, inplace=True)

        return hrv_data

    def process_resting_heart_rate(self, part_id: int = None) -> pd.DataFrame:
        """Process resting heart rate data as long as it is available."""
        return self._process_data("resting_heart_rate", part_id)
    
    def process_sleep_score(self, part_id: int = None) -> pd.DataFrame:
        """Process Sleep Score data as long as it is available."""
        return self._process_data("sleep_score", part_id)

    def process_step_count(self, part_id: int = None) -> pd.DataFrame:
        """Process Step Count data as long as it is available."""

        step_data = self._process_data("steps", part_id)

        step_data["dateTime"] = pd.to_datetime(step_data["dateTime"])
        step_data["date"] = pd.to_datetime(step_data["dateTime"]).dt.normalize()

        for participant_id, participant_data in step_data.groupby("participant_id"):

            participant_data = participant_data.sort_values("dateTime")

            # Daily totals
            daily_totals = (
                participant_data.groupby("date")["value"]
                .sum()
                .sort_index()
            )

            # Determine validity of each day
            valid_series = participant_data.groupby("date").apply(
                lambda x: (
                    len(x[x["dateTime"].dt.hour >= 7]) / len(x)
                ) >= 0.8
            )

            # Daily moving averages
            daily_ma = {
                window: daily_totals.rolling(
                    window=window,
                    min_periods=1
                ).mean()
                for window in range(2, 8)
            }

            # Filtered moving averages
            daily_ma_filtered = {}

            for window, ma in daily_ma.items():

                invalid_days = (
                    (~valid_series)
                    .rolling(window=window, min_periods=1)
                    .sum()
                )

                daily_ma_filtered[window] = ma.where(invalid_days <= 1)

            # Write daily values back to step_data
            for date, day_data in participant_data.groupby("date"):

                mask = (
                    (step_data["participant_id"] == participant_id)
                    & (step_data["date"] == date)
                )

                # Daily total
                step_data.loc[mask, "daily_steps"] = daily_totals.loc[date]

                # Moving averages
                for window in range(2, 8):

                    step_data.loc[
                        mask,
                        f"daily_steps_ma{window}_raw"
                    ] = daily_ma[window].loc[date]

                    step_data.loc[
                        mask,
                        f"daily_steps_ma{window}"
                    ] = daily_ma_filtered[window].loc[date]

                # Daily cumulative sum
                step_data.loc[
                    mask,
                    "cumsum"
                ] = day_data["value"].cumsum().values

            # Keep original date
            participant_mask = (
                step_data["participant_id"] == participant_id
            )

            step_data.loc[participant_mask, "date_raw"] = (
                step_data.loc[participant_mask, "date"]
            )

            # Remove invalid dates
            valid_dates = valid_series[valid_series].index

            step_data.loc[
                participant_mask & ~step_data["date"].isin(valid_dates),
                "date"
            ] = pd.NaT

        return step_data

if __name__ == "__main__":
    preprocessor = Preprocessor()
    hrv_data = preprocessor.process_heart_rate_variability()
    sleep_data = preprocessor.process_sleep_score()
    steps_data = preprocessor.process_step_count()
    print(hrv_data.head())
    print(sleep_data.head())
    print(steps_data.head())

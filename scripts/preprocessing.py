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

    def _normalize_part_ids(self, part_id: int | list[int] | None) -> list[int]:
        """Return a list of participant IDs to process."""

        if part_id is None:
            return self.participant_loader.pat_ids
        if isinstance(part_id, int):
            return [part_id]
        return list(part_id)

    def _attach_participant_id(
            self,
            frame: pd.DataFrame,
            participant_id: int,
    ) -> pd.DataFrame:
        """Insert participant_id as the first column."""

        frame = frame.copy()
        frame.insert(0, "participant_id", participant_id)
        return frame

    def _parse_steps(
            self,
            loaded_data: list,
            participant_id: int,
    ) -> list[pd.DataFrame]:
        """Parse and aggregate step records into the final step DataFrame."""

        frames: list[pd.DataFrame] = []
        for records in loaded_data:
            frame = pd.DataFrame(records, columns=["dateTime", "value"])
            frame["dateTime"] = pd.to_datetime(
                frame["dateTime"],
                format="%m/%d/%y %H:%M:%S",
            )
            frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
            frames.append(frame)

        if not frames:
            return []

        step_data = pd.concat(frames, ignore_index=True)
        step_data = self._attach_participant_id(step_data, participant_id)
        step_data["date"] = step_data["dateTime"].dt.normalize()

        participant_data = step_data.sort_values("dateTime")

        daily_totals = (
            participant_data.groupby("date")["value"]
            .sum()
            .sort_index()
        )
        valid_series = participant_data.groupby("date").apply(
            lambda x: (
                len(x[x["dateTime"].dt.hour >= 7]) / len(x)
            ) >= 0.6
        )

        daily_ma = {
            window: daily_totals.rolling(
                window=window,
                min_periods=1,
            ).mean()
            for window in range(2, 8)
        }

        daily_ma_filtered = {}
        for window, ma in daily_ma.items():
            invalid_days = (
                (~valid_series)
                .rolling(window=window, min_periods=1)
                .sum()
            )
            daily_ma_filtered[window] = ma.where(invalid_days <= 1)

        for date, day_data in participant_data.groupby("date"):
            mask = step_data["date"] == date

            step_data.loc[mask, "daily_steps"] = daily_totals.loc[date]

            for window in range(2, 8):
                step_data.loc[mask, f"daily_steps_ma{window}_raw"] = daily_ma[window].loc[date]
                step_data.loc[mask, f"daily_steps_ma{window}"] = daily_ma_filtered[window].loc[date]

            step_data.loc[mask, "cumsum"] = day_data["value"].cumsum().values

        step_data["date_raw"] = step_data["date"]

        valid_dates = valid_series[valid_series].index
        step_data.loc[~step_data["date"].isin(valid_dates), "date"] = pd.NaT


        # compute cadence as steps per minute based on consecutive timestamps within each day
        cadence = participant_data.groupby("date", group_keys=False).apply(
            lambda group: group["value"].div(
                group["dateTime"].diff().dt.total_seconds().div(60).replace(0, pd.NA)
            ).fillna(0)
        )
        step_data["cadence"] = cadence.reindex(step_data.index).fillna(0)
        return [step_data]

    def _parse_hrv_summary(
            self,
            loaded_data: list,
            participant_id: int,
    ) -> list[pd.DataFrame]:
        """Parse heart rate variability summary records."""
        frames: list[pd.DataFrame] = []
        for records in loaded_data:
            date_time = pd.to_datetime(records["timestamp"], format="ISO8601")
            frame = pd.DataFrame({
                "dateTime": [date_time],
                "rmssd": [records["rmssd"]],
            })
            frame["rmssd"] = pd.to_numeric(frame["rmssd"], errors="coerce")
            frame.loc[frame["rmssd"] == 0, "rmssd"] = pd.NA
            frames.append(self._attach_participant_id(frame, participant_id))

        if not frames:
            return []
        hrv_frames = pd.concat(frames, ignore_index=True)
        hrv_frames.sort_values(by=["participant_id", "dateTime"], inplace=True)
        hrv_frames.reset_index(drop=True, inplace=True)
        return [hrv_frames]

    def _parse_resting_heart_rate(
            self,
            loaded_data: list,
            participant_id: int,
    ) -> list[pd.DataFrame]:
        """Parse resting heart rate records."""

        frames: list[pd.DataFrame] = []
        for records in loaded_data:
            records_df = pd.DataFrame(records)
            records_df["dateTime"] = pd.to_datetime(
                records_df["value"].str["date"],
                format="%m/%d/%y",
                errors="coerce",
            )
            frame = pd.DataFrame({
                "dateTime": records_df["dateTime"],
                "value": records_df["value"].str["value"],
                "error": records_df["value"].str["error"],
            })

            for col in ["value", "error"]:
                frame[col] = pd.to_numeric(frame[col], errors="coerce")

            frame = frame[frame["value"] != 0]
            frames.append(self._attach_participant_id(frame, participant_id))

        if not frames:
            return []

        resting_heart_rate = pd.concat(frames, ignore_index=True)
        resting_heart_rate.sort_values(by=["participant_id", "dateTime"], inplace=True)
        resting_heart_rate.reset_index(drop=True, inplace=True)
        return [resting_heart_rate]

    def _parse_heart_rate(
            self,
            loaded_data: list,
            participant_id: int,
    ) -> list[pd.DataFrame]:
        """Parse heart rate records."""

        frames: list[pd.DataFrame] = []
        for records in loaded_data:
            records_df = pd.DataFrame(records)
            records_df["dateTime"] = pd.to_datetime(
                records_df["dateTime"],
                format="%m/%d/%y %H:%M:%S",
                errors="coerce",
            )
            frame = pd.DataFrame({
                "dateTime": records_df["dateTime"],
                "bpm": records_df["value"].str["bpm"],
                "confidence": records_df["value"].str["confidence"],
            })

            for col in ["bpm", "confidence"]:
                frame[col] = pd.to_numeric(frame[col], errors="coerce")

            frame = frame[frame["bpm"] != 0]
            frames.append(self._attach_participant_id(frame, participant_id))
        return frames

    def _process_data(
            self,
            data_type: str,
            part_id: int | list[int] | None = None,
    ) -> pd.DataFrame:
        """Load data for all participants and return it in long format.
        If part_id is provided, only load data for that participant.
        Args:
            data_type (str): The type of data to process (e.g., "steps", "heart_rate").
            part_id (int | list[int], optional): The participant ID(s) to filter.
        Returns:
            pd.DataFrame: A DataFrame containing the processed data.
        """
        parser_map = {
            "steps": self._parse_steps,
            "Heart Rate Variability Summary": self._parse_hrv_summary,
            "resting_heart_rate": self._parse_resting_heart_rate,
            "^heart_rate": self._parse_heart_rate,
        }

        frames = []
        participant_ids = self._normalize_part_ids(part_id)
        parser = parser_map.get(data_type)

        for parent_path, files in self.loader.find_paths(data_type):
            for participant_id in participant_ids:
                paths = self.participant_loader.make_file_paths(
                    participant_id,
                    parent_path,
                    files,
                )
                loaded_data = self.participant_loader.load_files(paths)
                if parser is None:
                    participant_data = pd.DataFrame(loaded_data)
                    frames.append(self._attach_participant_id(participant_data, participant_id))
                    continue

                frames.extend(parser(loaded_data, participant_id))

        if not frames:
            return pd.DataFrame()

        return pd.concat(frames, ignore_index=True)

    def process_heart_rate(self, part_id: int = None) -> pd.DataFrame: 
        """Process heart rate data as long as it is available."""
        return self._process_data("^heart_rate", part_id)

    def process_heart_rate_variability(self, part_id: int = None) -> pd.DataFrame:
        """Process HRV data as long as it is available."""
        return self._process_data("Heart Rate Variability Summary", part_id)

    def process_resting_heart_rate(self, part_id: int = None) -> pd.DataFrame:
        """Process resting heart rate data as long as it is available."""
        return self._process_data("resting_heart_rate", part_id)

    def process_sleep_score(self, part_id: int = None) -> pd.DataFrame:
        """Process Sleep Score data as long as it is available."""
        return self._process_data("sleep_score", part_id)

    def process_step_count(self, part_id: int = None) -> pd.DataFrame:
        """Process Step Count data as long as it is available."""
        return self._process_data("steps", part_id)

    def get_daily_hr_stats(self, part_ids, data) -> pd.DataFrame:
        """Get daily statistics for all participants."""
        frames = []
        for part_id in self._normalize_part_ids(part_ids):
            data_part = data[data["participant_id"] == part_id]
            if data_part.empty:
                continue

            frame = data_part.groupby(data_part["dateTime"].dt.date).agg(
                daily_mean=("bpm", "mean"),
                daily_median=("bpm", "median"),
                daily_min=("bpm", "min"),
                daily_max=("bpm", "max"),
            ).reset_index()
            frame["participant_id"] = part_id
            frames.append(frame)

        if not frames:
            return pd.DataFrame()

        return pd.concat(frames, ignore_index=True)

    def merge_cadence_with_hr(self,cadence_data: pd.DataFrame, hr_data: pd.DataFrame) -> pd.DataFrame:
        """Merge cadence data with heart rate data based on the closest timestamp."""
        if cadence_data.empty or hr_data.empty:
            raise ValueError("Both cadence_data and hr_data must be non-empty DataFrames.")

        if "dateTime" not in cadence_data.columns or "dateTime" not in hr_data.columns:
            raise ValueError("Both DataFrames must contain a 'dateTime' column.")

        # Ensure dateTime columns are in datetime format
        cadence_data["dateTime"] = pd.to_datetime(cadence_data["dateTime"], errors="coerce")
        hr_data["dateTime"] = pd.to_datetime(hr_data["dateTime"], errors="coerce")

        # Drop rows with NaT in dateTime
        cadence_data = cadence_data.dropna(subset=["dateTime"])
        hr_data = hr_data.dropna(subset=["dateTime"])

        # Sort both DataFrames by dateTime
        cadence_data = cadence_data.sort_values("dateTime").reset_index(drop=True)
        hr_data = hr_data.sort_values("dateTime").reset_index(drop=True)
        hr_data["zone_num"] = pd.to_numeric(
            hr_data["zone"].str.replace("zone", "", regex=False),
            errors="coerce"
        )
        hr_minute = (
            hr_data
            .set_index("dateTime")
            .resample("1min")
            .agg({
                "bpm": "max",
                "zone_num": "max"
            })
            .reset_index()
        )

        hr_minute["zone"] = "zone" + hr_minute["zone_num"].astype(str)
        hr_minute.drop(columns="zone_num", inplace=True)
        # Merge using merge_asof to find the closest previous timestamp
        merged_df = pd.merge_asof(
            cadence_data,
            hr_minute,
            on="dateTime",
            direction="backward",
            suffixes=("_cadence", "_hr")
        )

        return merged_df

if __name__ == "__main__":
    preprocessor = Preprocessor()
    # heart_rate_data = preprocessor.process_heart_rate(part_id=[11])
    # print(heart_rate_data.head())
    hrv_data = preprocessor.process_heart_rate(part_id=[11])
    print(hrv_data.head())

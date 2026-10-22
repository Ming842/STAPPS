"""Preprocessing module for data files in STAPPS."""
import logging
from typing import Any

import pandas as pd
import numpy as np

from config import DATA_DIR
from loaders import ParticipantDataLoader
from parsers import normalize_part_ids, get_timestamp_column
from structure_loader import StructureDataLoader


class Preprocessor:
    """Class to handle preprocessing of data files."""

    def __init__(self) -> None:
        """Initialize the preprocessor with data loading dependencies.

        Returns:
            None: Creates the default data directory and loader instances.
        """
        self.data_dir = DATA_DIR
        self.loader = StructureDataLoader()
        self.participant_loader = ParticipantDataLoader()

    def _normalize_part_ids(self, part_id: int | list[int] | None) -> list[int]:
        """Return a list of participant IDs to process.

        Args:
            part_id (int | list[int] | None): Single participant ID, a list of IDs, or None for all participants.

        Returns:
            list[int]: Normalized participant IDs to process.
        """
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
        """Insert participant_id as the first column.

        Args:
            frame (pd.DataFrame): DataFrame to add a participant ID column to.
            participant_id (int): Participant identifier for the records.

        Returns:
            pd.DataFrame: Copy of the original DataFrame with a participant_id column prepended.
        """
        frame = frame.copy()
        frame.insert(0, "participant_id", participant_id)
        return frame

    def _parse_steps(
            self,
            loaded_data: list,
            participant_id: int,
    ) -> list[pd.DataFrame]:
        """Parse and aggregate step records into the final step DataFrame.

        Args:
            loaded_data (list): List of step-record dictionaries from the Fitbit export.
            participant_id (int): Participant identifier associated with the records.

        Returns:
            list[pd.DataFrame]: A list containing the processed step DataFrame.
        """
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
        """Parse heart rate variability summary records.

        Args:
            loaded_data (list): List of raw HRV summary records.
            participant_id (int): Participant identifier associated with the data.

        Returns:
            list[pd.DataFrame]: One processed HRV DataFrame wrapped in a list.
        """
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
        """Parse resting heart rate records.

        Args:
            loaded_data (list): List of raw resting heart rate records.
            participant_id (int): Participant identifier associated with the data.

        Returns:
            list[pd.DataFrame]: One processed resting-heart-rate DataFrame wrapped in a list.
        """
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
        """Parse heart rate records.

        Args:
            loaded_data (list): List of raw heart-rate records.
            participant_id (int): Participant identifier associated with the data.

        Returns:
            list[pd.DataFrame]: Processed heart-rate DataFrames related to the participant.
        """
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

        Args:
            data_type (str): The type of data to process (for example, "steps" or "heart_rate").
            part_id (int | list[int] | None): Participant ID or IDs to filter.

        Returns:
            pd.DataFrame: DataFrame containing the processed data for the requested participant(s).
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

    def process_heart_rate(self, part_id: int | list[int] | None = None) -> pd.DataFrame:
        """Process heart rate data as long as it is available.

        Args:
            part_id (int | list[int] | None): Participant ID or IDs to load.

        Returns:
            pd.DataFrame: Processed heart-rate data.
        """
        return self._process_data("^heart_rate", part_id)

    def process_heart_rate_variability(self, part_id: int | list[int] | None = None) -> pd.DataFrame:
        """Process HRV data as long as it is available.

        Args:
            part_id (int | list[int] | None): Participant ID or IDs to load.

        Returns:
            pd.DataFrame: Processed HRV data.
        """
        return self._process_data("Heart Rate Variability Summary", part_id)

    def process_resting_heart_rate(self, part_id: int | list[int] | None = None) -> pd.DataFrame:
        """Process resting heart rate data as long as it is available.

        Args:
            part_id (int | list[int] | None): Participant ID or IDs to load.

        Returns:
            pd.DataFrame: Processed resting-heart-rate data.
        """
        return self._process_data("resting_heart_rate", part_id)

    def process_sleep_score(self, part_id: int | list[int] | None = None) -> pd.DataFrame:
        """Process Sleep Score data as long as it is available.

        Args:
            part_id (int | list[int] | None): Participant ID or IDs to load.

        Returns:
            pd.DataFrame: Processed sleep-score data.
        """
        return self._process_data("sleep_score", part_id)

    def process_step_count(self, part_id: int | list[int] | None = None) -> pd.DataFrame:
        """Process Step Count data as long as it is available.

        Args:
            part_id (int | list[int] | None): Participant ID or IDs to load.

        Returns:
            pd.DataFrame: Processed step-count data.
        """
        return self._process_data("steps", part_id)

    def get_daily_hr_stats(self, part_ids: int | list[int], data: pd.DataFrame) -> pd.DataFrame:
        """Get daily heart-rate statistics for all participants.

        Args:
            part_ids (int | list[int]): Participant ID or IDs to summarize.
            data (pd.DataFrame): Heart-rate data including a participant_id and dateTime column.

        Returns:
            pd.DataFrame: Daily summary statistics per participant.
        """
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

def load_plot_data(part_ids: list, data_types: list[str] | tuple[str, ...] | str | None = None):
    """Load and organize plot-ready data for one or more participants.

    Args:
        part_ids (list): Participant IDs to include in the output payload.
        data_types (list[str] | tuple[str, ...] | str | None): Data types to request. Defaults to all supported types.

    Returns:
        tuple | tuple[list, dict]: Participant IDs together with the requested payload dictionary, or legacy tuple format.
    """
    participant_loader = ParticipantDataLoader()
    preprocessor = Preprocessor()

    part_ids = normalize_part_ids(part_ids)

    valid_types = {"rhr", "hrv", "hr", "steps", "cycles"}
    if data_types is None:
        logging.info("No data types specified. Defaulting to all available types.")
        data_types = ("rhr", "hrv", "hr", "steps", "cycles")
    elif isinstance(data_types, str):
        data_types = (data_types,)

    invalid = set(data_types) - valid_types
    if invalid:
        raise ValueError(f"Unsupported data types: {sorted(invalid)}\n Supported types: {sorted(valid_types)}")

    payload = {}

    if "rhr" in data_types:
        payload["rhr"] = preprocessor.process_resting_heart_rate(part_id=part_ids)
    if "hrv" in data_types:
        payload["hrv"] = preprocessor.process_heart_rate_variability(part_id=part_ids)
    if "hr" in data_types:
        payload["hr"] = preprocessor.process_heart_rate(part_id=part_ids)
    if "steps" in data_types:
        payload["steps"] = preprocessor.process_step_count(part_id=part_ids)
    if "cycles" in data_types:
        payload["cycles"] = participant_loader.cycle_data(part_id=part_ids)

    if data_types == ("rhr", "hrv", "steps", "cycles"):
        return (
            part_ids,
            payload.get("rhr"),
            payload.get("hrv"),
            payload.get("hr"),
            payload.get("steps"),
            payload.get("cycles"),
        )

    return part_ids, payload

def merge_cadence_with_hr(cadence_data: pd.DataFrame, hr_data: pd.DataFrame) -> pd.DataFrame:
    """Merge cadence data with heart rate data based on the closest timestamp.

    Args:
        cadence_data (pd.DataFrame): DataFrame containing cadence values and a dateTime column.
        hr_data (pd.DataFrame): DataFrame containing heart-rate values and a dateTime column.

    Returns:
        pd.DataFrame: Merged DataFrame with cadence and hour-zone information aligned by timestamp.
    """
    if cadence_data.empty or hr_data.empty:
        raise ValueError("Both cadence_data and hr_data must be non-empty DataFrames.")

    if "dateTime" not in cadence_data.columns or "dateTime" not in hr_data.columns:
        raise ValueError("Both DataFrames must contain a 'dateTime' column.")

    cadence_data["dateTime"] = pd.to_datetime(cadence_data["dateTime"], errors="coerce")
    hr_data["dateTime"] = pd.to_datetime(hr_data["dateTime"], errors="coerce")

    cadence_data = cadence_data.dropna(subset=["dateTime"])
    hr_data = hr_data.dropna(subset=["dateTime"])

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
            "bpm": "median",
            "zone_num": "median"
        })
        .reset_index()
    )

    hr_minute["zone_num"] = hr_minute["zone_num"].round().astype("Int64")
    hr_minute["zone"] = hr_minute["zone_num"].map(
        lambda value: f"zone{int(value)}" if pd.notna(value) else pd.NA
    )
    hr_minute.drop(columns="zone_num", inplace=True)
    merged_df = pd.merge_asof(
        cadence_data,
        hr_minute,
        on="dateTime",
        direction="backward",
        suffixes=("_cadence", "_hr")
    )

    return merged_df

def part_isin_data(part_id: int | list[int], data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Filter each dataset to rows belonging to the provided participant IDs.

    Args:
        part_id (int | list[int]): Participant ID or IDs to keep.
        data (dict[str, pd.DataFrame]): Mapping of dataset names to DataFrames.

    Returns:
        dict[str, pd.DataFrame]: Filtered data dictionary with only matching participants retained.
    """
    for key in data.keys():
        data[key] = data[key][data[key]['participant_id'].isin(part_id)]
    return data

def _flatten_hospitalization_dates(value: list) -> list[Any]:
    """Return the first valid date from each nested list while keeping direct scalar dates intact.

    Examples:
    - [None, [[date1, date2]], None] -> [date1]
    - [[[date1, date2]], [[date3, date4]]] -> [date1, date3]
    - [None, None, [[date1, date2]], [[date3, date4]]] -> [date1, date3]
    - [date1, date2] -> [date1, date2]
    """
    if value is None or value is pd.NaT:
        return None

    if isinstance(value, (list, tuple)):
        flattened: list[Any] = []
        for item in value:
            if item is None or item is pd.NaT:
                continue
            if isinstance(item, (list, tuple)):
                valids = []
                if len(item) == 1 and isinstance(item[0], (list, tuple)):
                    valids.append(item[0][0])
                else: 
                    for nested in item:
                        if isinstance(nested, (list, tuple)):
                                valids.append(nested[0])
                for valid in valids:
                    if pd.notna(valid):
                        flattened.append(valid)
                continue

            try:
                if pd.isna(item):
                    continue
            except (TypeError, ValueError):
                continue

            flattened.append(item)

        return flattened if flattened != [] else None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    return None

def _flatten_date_lists(value: list) -> list[Any]:
    """Return the first valid date from each nested list while keeping direct scalar dates intact.

    Examples:
    - [None, [[date1, date2]], None] -> [date1, date2]
    - [date1, None, None] -> [date1]
    - [None, None, [[date1, date2]], [[date3, date4]]] -> [date1, date2, date3, date4]
    - [date1, date2] -> [date1, date2]

    Args:
        value (list): A list potentially containing nested lists of dates.
    Returns:
        list: A flattened list of valid dates, or None if no valid dates are found.
    """
    if value is None or value is pd.NaT:
        return []

    if isinstance(value, (list, tuple)):
        flattened: list[Any] = []
        for item in value:
            if item is None or item is pd.NaT:
                continue
            if isinstance(item, (list, tuple)):
                for nested in item:
                    if isinstance(nested, (list, tuple)):
                        flattened.extend(nested)
                    else:
                        flattened.append(nested)
                continue

            try:
                if pd.isna(item):
                    continue
            except (TypeError, ValueError):
                continue

            flattened.append(item)

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    return flattened if flattened != [] else None
def _get_chemo_cycle(value: list) -> list[Any]:
    """Return the first valid chemo cycle from each nested list and discard empty values.

    Examples:
    - [False, False, True, False] -> [C3]
    - [False, True, True, False] -> [C2, C3]
    - [False, False, False, False] -> []
    - [True, True, True, True] -> [C1, C2, C3, C4]

    Args:
        value (list): A list of boolean values representing the chemo cycles.

    Returns:
        list: A list of strings representing the active chemo cycles.
    """
    if value is None or value is pd.NaT:
        return None
    if isinstance(value, list):
        cycles = []
        for index, item in enumerate(value):
            if item is np.True_ or item is True:
                cycles.append(f"C{index + 1}")
        return cycles

    return None

def _cycle_to_date(value: list, cycle_data: pd.DataFrame, participant_id: int | None = None, ) -> list[Any]:
    """Return the dates corresponding to the active chemo cycles for a participant.

    Args:
        value (list): A list of cycle labels, for example ["C1", "C2"].
        cycle_data (pd.DataFrame): Participant cycle data.
        participant_id (int | None): Participant ID to filter on.

    Returns:
        list: Dates matching the requested cycles for that participant. If no matching cycles are found, returns an empty list. If cycle is outside of the participant's data, returns last available date.
    """
    if value is None or value is pd.NaT:
        return None
    if not isinstance(value, list):
        return None

    if participant_id is None:
        participant_id = cycle_data["participant_id"].iloc[0] if not cycle_data.empty else None

    dates: list[Any] = []
    for cycle in value:
        if not isinstance(cycle, str) or not cycle.startswith("C"):
            continue

        mask = cycle_data["Period"].eq(cycle)
        if participant_id is not None:
            mask &= cycle_data["participant_id"].eq(participant_id)
        cycle_dates = cycle_data.loc[mask, "Date"]
        if cycle_dates.empty:
            dates.extend([cycle])
        else:
            dates.extend(cycle_dates.tolist())

    return dates if dates != [] else None



def add_side_effect_column(part_id: int | list[int], data: dict[pd.DataFrame]) -> dict[pd.DataFrame]:
    """Add a 'side_effect' column to each dataset based on the cycles DataFrame.

    Args:
        part_id (int | list[int]): Participant ID or IDs to filter.
        data (dict[str, pd.DataFrame]): Dictionary containing participant step, RHR, and cycle data.

    Returns:
        dict[str, pd.DataFrame]: Updated data dictionary with 'side_effect' column added to each dataset.
    """
    cycle_data = data.get("cycles")
    if cycle_data is None:
        raise ValueError("The data dictionary must include a 'cycles' DataFrame.")
    # last_time_stamp = None
    # for key in data.keys():
    #     df = data[key]
    #     time_col = get_timestamp_column(df)

    #     # Ensure datetime dtype; errors='coerce' will expose bad values as NaT
    #     df[time_col] = pd.to_datetime(df[time_col], errors="coerce")

    #     max_time = df[time_col].max()

    #     if pd.isna(max_time):
    #         raise ValueError(
    #             f"No valid timestamps in data[{key!r}] column {time_col!r}"
    #         )

    #     if last_time_stamp is None or last_time_stamp < max_time:
    #         last_time_stamp = max_time

    for key in data.keys():
        if key == "cycles":
            continue
        
        side_effect_mask = cycle_data[cycle_data['participant_id'].isin(part_id) & cycle_data['Bijwerking']]
        side_effect_ids = side_effect_mask['participant_id'].unique()
        side_effect_data = pd.DataFrame({
            "participant_id": side_effect_ids,
            "side_effect": True,
            "hospitalization_dates": (side_effect_mask
                                    .groupby("participant_id")["hospitalizations_date"]
                                    .apply(list)
                                    .apply(_flatten_hospitalization_dates)),
            "ed_visits_dates": (side_effect_mask
                                .groupby("participant_id")["ed_visits_date"]
                                .apply(list)
                                .apply(_flatten_date_lists)),
            "dose_changed_for_cycle": (side_effect_mask
                                       .groupby("participant_id")["Dose_Modifications"]
                                       .apply(list)
                                       .apply(_get_chemo_cycle))
        })
        side_effect_data["cycle_dates"] = side_effect_data.apply(
            lambda row: _cycle_to_date(row["dose_changed_for_cycle"], cycle_data, row["participant_id"]),
            axis=1
        )
        side_effect = data[key]["participant_id"].isin(side_effect_ids)
        data[key]["side_effect"] = side_effect
        cols = [
            "participant_id",
            "hospitalization_dates",
            "ed_visits_dates",
            "cycle_dates",
        ]
        data[key] = pd.merge(
            data[key].reset_index(drop=True),
            side_effect_data[cols].reset_index(drop=True),
            on="participant_id",
            how="left",
            validate="many_to_one",
        )
    return data

def get_moving_mean_valid(
    data: pd.DataFrame | pd.Series,
    windows: int | list[int],
) -> dict[int, pd.Series]:
    """Return rolling means for each requested window while allowing at most one invalid day.

    Args:
        data (pd.DataFrame | pd.Series): Data containing a datetime index or date/time column and a numeric value.
        windows (int | list[int]): One or more window sizes to compute.

    Returns:
        dict[int, pd.Series]: Mapping from window size to the computed rolling mean series.
    """
    if isinstance(windows, int):
        windows = [windows]
    else:
        windows = list(windows)

    if isinstance(data, pd.Series):
        series = data.copy()
        value_name = series.name if series.name not in {None, "date", "dateTime"} else "value"
        if isinstance(series.index, pd.DatetimeIndex):
            data = pd.DataFrame({"date": series.index, value_name: series.to_numpy()})
        else:
            data = pd.DataFrame({"date": pd.to_datetime(series.index, errors="coerce"), value_name: series.to_numpy()})
    else:
        data = data.copy()

    if data.empty:
        return {window: pd.Series(dtype=float) for window in windows}

    if "date" not in data.columns:
        if "dateTime" in data.columns:
            data["date"] = pd.to_datetime(data["dateTime"], errors="coerce")
        elif isinstance(data.index, pd.DatetimeIndex):
            data["date"] = data.index
        else:
            data["date"] = pd.to_datetime(data.index, errors="coerce")

    if "cadence" in data.columns:
        value_col = "cadence"
    elif "rmssd" in data.columns:
        value_col = "rmssd"
    elif "value" in data.columns:
        value_col = "value"
    else:
        value_cols = [c for c in data.columns if c not in {"date", "dateTime", "zone"}]
        if not value_cols:
            raise ValueError("No numeric value column found for moving mean computation")
        value_col = value_cols[0]

    data = data[["date", value_col, *( ["zone"] if "zone" in data.columns else [] )]].copy()
    data = data.sort_values("date").reset_index(drop=True)

    if "zone" in data.columns:
        grouped = []
        for _, zone_df in data.groupby("zone", sort=False):
            zone_df = zone_df.sort_values("date").reset_index(drop=True)
            valid_series = zone_df[value_col].notna()
            zone_output = []
            for window in windows:
                rolling = zone_df[value_col].rolling(window=window, min_periods=1).mean()
                invalid_days = (~valid_series).rolling(window=window, min_periods=1).sum()
                zone_output.append(rolling.where(invalid_days <= 1))
            grouped.append(pd.concat(zone_output, axis=1, keys=windows))
        result = {}
        for window in windows:
            result[window] = pd.concat([g[window] for g in grouped], axis=0)
        return result

    valid_series = data[value_col].notna()
    result = {}
    for window in windows:
        rolling = data[value_col].rolling(window=window, min_periods=1).mean()
        invalid_days = (~valid_series).rolling(window=window, min_periods=1).sum()
        result[window] = rolling.where(invalid_days <= 1)
    return result

def get_baseline_data(
    data: pd.DataFrame,
    date_col: str,
    cycle_data: pd.DataFrame,
) -> pd.DataFrame:
    """Return the baseline subset of a dataset based on cycle dates.

    Args:
        data (pd.DataFrame): DataFrame containing the timestamp column used for the baseline cut-off.
        date_col (str): Name of the datetime column in ``data``.
        cycle_data (pd.DataFrame): Cycle data containing ``Period`` and ``Date`` fields.

    Returns:
        pd.DataFrame: Rows in ``data`` that fall before the computed baseline end date.
    """
    if data.empty:
        return data.copy()
    if cycle_data.empty:
        raise ValueError("Cycle data is empty. Cannot determine baseline without cycle information.")
    if date_col not in data.columns:
        raise ValueError(f"data must contain the column '{date_col}'.")
    if not {"Period", "Date"}.issubset(cycle_data.columns):
        raise ValueError("cycle_data must contain the columns 'Period' and 'Date'.")

    c1_rows = cycle_data.loc[cycle_data["Period"] == "C1", "Date"]
    if c1_rows.empty:
        c1_rows = cycle_data.loc[cycle_data["Period"].astype(str).str.startswith("Week"), "Date"]
    if c1_rows.empty:
        raise ValueError("cycle_data does not contain a C1 date.")

    c1_date = c1_rows.iloc[0]
    if c1_date == "Dagelijks":
        end_baseline = pd.to_datetime(data[date_col].min()) + pd.Timedelta(days=3)
        logging.warning("C1 date is 'Dagelijks'. Using the first three days as the baseline.")
    else:
        end_baseline = pd.to_datetime(c1_date, errors="coerce")
        if pd.isna(end_baseline):
            raise ValueError("C1 date could not be parsed as a valid datetime.")

    start_baseline = pd.to_datetime(data[date_col].min())
    baseline_days = end_baseline - start_baseline
    if baseline_days <= pd.Timedelta(days=3):
        if "C2" in cycle_data["Period"].values:
            end_baseline = pd.to_datetime(cycle_data.loc[cycle_data["Period"] == "C2", "Date"].iloc[0], errors="coerce")
            if pd.isna(end_baseline):
                raise ValueError("C2 date could not be parsed as a valid datetime.")
            logging.warning("Baseline period is less than 3 days; using 3 days before C2 as baseline period.") 
        else:
            end_baseline = pd.to_datetime(data[date_col].min()) + pd.Timedelta(days=3)
            if end_baseline > pd.to_datetime(data[date_col].max()):
                raise ValueError("Not enough data to establish a baseline period. Ensure there are at least 3 days of data before C1 or C2.")
            logging.warning("Baseline period before chemo is less than 3 days; C2 non existent. Using the first three days as baseline period.")

    baseline = data[data[date_col] < end_baseline].copy()
    if baseline.empty:
        raise ValueError("No baseline data available before the calculated end date.")
    return baseline

def get_before_and_after_chemo(data): 
    """Add numeric chemo timing columns per record.

    Adds:
    - before_after_chemo: "before", "during", or "after"
    - days_after_chemo: integer days after the last chemo date (0 otherwise)

    Args:
        data (dict[str, pd.DataFrame]): Dictionary containing participant datatypes and cycle data.
    
    Returns:
        dict[str, pd.DataFrame]: Updated data dictionary with chemo timing columns added to each dataset

    """
    cycle_data = data.get("cycles")
    if cycle_data is None:
        raise ValueError("The data dictionary must include a 'cycles' DataFrame.")
    dict_no_ae = {}
    for key in data.keys():
        if key == "cycles":
            continue

        df = data[key].copy()
        df_no_ae = df[df["side_effect"] != True]

        time_col = get_timestamp_column(df_no_ae) 
        df_no_ae[time_col] = pd.to_datetime(df_no_ae[time_col], errors="coerce")

        df_no_ae["before_after_chemo"] = pd.NA
        df_no_ae["days_before_chemo"] = 0
        df_no_ae["days_after_chemo"] = 0
        part_ids = list(df_no_ae["participant_id"].dropna().unique())
        for part_id in part_ids:
            part_cycle_data = cycle_data[cycle_data["participant_id"] == part_id].copy()
            part_cycle_data["Date"] = pd.to_datetime(part_cycle_data["Date"], errors="coerce")
            
            c1_date = part_cycle_data["Date"][part_cycle_data["Period"] == "C1"]
            if c1_date.empty:
                logging.warning(f"No C1 date found for participant {part_id}. Skipping chemo timing calculations.")
                part_ids.remove(part_id)
                continue
            else:
                c1_date = c1_date.item().normalize()  # Normalize to remove time component

            c2_date = part_cycle_data["Date"][part_cycle_data["Period"] == "C2"]
            if c2_date.empty:
                logging.warning(f"No C2 date found for participant {part_id}. Using last datapoint")
                c2_date = df_no_ae[time_col][df_no_ae["participant_id"] == part_id].max().normalize()  # Normalize to remove time component
                continue
            else:
                c2_date = c2_date.item().normalize() # Normalize to remove time component       
            mask = df_no_ae["participant_id"] == part_id

            for idx in df_no_ae.index[mask]:
                timestamp = df_no_ae.at[idx, time_col]
                if pd.isna(timestamp):
                    df_no_ae.at[idx, "days_after_chemo"] = pd.NA
                    continue


                if timestamp == c1_date:
                    df_no_ae.at[idx, "days_after_chemo"] = 0
                elif timestamp > c2_date:
                    df_no_ae.at[idx, "days_after_chemo"] = 1000
                else:
                    df_no_ae.at[idx, "days_after_chemo"] = (timestamp - c1_date).days




        df_no_ae = df_no_ae[df_no_ae["days_after_chemo"] != 1000]
        df_no_ae = df_no_ae[df_no_ae["participant_id"].isin(part_ids)]
        dict_no_ae[key] = df_no_ae
    return dict_no_ae


if __name__ == "__main__":
    pat_ids, data = load_plot_data([1,2,4,5,6,7,8,10,11,12,14], data_types=("steps", "hrv", "cycles"))
    data = add_side_effect_column([1,2,4,5,6,7,8,10,11,12,14], data)
    
    print(data["steps"].head())
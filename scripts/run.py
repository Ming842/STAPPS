import matplotlib.pyplot as plt
import pandas as pd

from statistics_def import get_hr_zones
from loaders import write_to_pickle
from preprocessing import (load_plot_data,
                           add_side_effect_column,
                           get_smoothed_cadence_by_zone, merge_cadence_with_hr)

pat_ids = list(range(1, 31))
pat_ids, data = load_plot_data(pat_ids, data_types=("rhr", "hr", "steps", "hrv", "sleep","cycles"))
data = add_side_effect_column(pat_ids, data)
steps_data = data["steps"]
hr_data = data["hr"]
cycles = data["cycles"]
sleep_data = data["sleep"]

hr_data_zoned, zone_thresholds = get_hr_zones(hr_data, cycles)
smoothed_cadence_by_zone = []

for pat_id in pat_ids:
    patient_steps = steps_data.loc[
        steps_data["participant_id"].eq(pat_id)
    ]
    patient_hr = hr_data_zoned.loc[
        hr_data_zoned["participant_id"].eq(pat_id)
    ]

    data_merge = merge_cadence_with_hr(
        patient_steps[["dateTime", "cadence"]], patient_hr
    )
    patient_smoothed = get_smoothed_cadence_by_zone(
        data_merge, window_size=3
    )
    patient_smoothed["participant_id"] = pat_id
    smoothed_cadence_by_zone.append(patient_smoothed)

smoothed_cadence_by_zone = pd.concat(
    smoothed_cadence_by_zone, ignore_index=True
)
smoothed_cadence_by_zone = smoothed_cadence_by_zone[
    ["participant_id"]
    + [
        column
        for column in smoothed_cadence_by_zone.columns
        if column != "participant_id"
    ]
]

for zone in range(1, 6):
    zone_name = f"zone{zone}"
    column_name = f"smoothed_cadence_by_{zone_name}"
    cadence_data = smoothed_cadence_by_zone[
        smoothed_cadence_by_zone["zone"].eq(zone_name)
    ]
    cadence_data = cadence_data[["participant_id", "date", "cadence"]].copy()
    pat_ids = cadence_data["participant_id"].unique()
    data[column_name] = cadence_data

datatypes = ["smoothed_cadence_by_zone1",
                "smoothed_cadence_by_zone2",
                "smoothed_cadence_by_zone3",
                "smoothed_cadence_by_zone4",
                "smoothed_cadence_by_zone5",
                "cycles"]

data_filt = {dt: data[dt] for dt in datatypes if dt in data}
pat_ids =  data_filt["cycles"]["participant_id"].unique()

data_filt = add_side_effect_column(pat_ids, data_filt)

for datatype in datatypes:
    if datatypes != "cycles":
        data[datatype] = data_filt[datatype]

write_to_pickle(data)

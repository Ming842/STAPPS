    # pat_ids = list(range(1, 31))
    # pat_ids, data = load_plot_data(pat_ids, data_types=("hr", "steps", "cycles"))
    # data = add_side_effect_column(pat_ids, data)
    # steps_data = data["steps"]
    # hr_data = data["hr"]
    # cycles = data["cycles"]

    # hr_data_zoned, zone_thresholds = get_hr_zones(hr_data, cycles)
    # smoothed_cadence_by_zone = []
    # for pat_id in pat_ids:
    #     patient_steps = steps_data.loc[
    #         steps_data["participant_id"].eq(pat_id)
    #     ]
    #     patient_hr = hr_data_zoned.loc[
    #         hr_data_zoned["participant_id"].eq(pat_id)
    #     ]

    #     data_merge = merge_cadence_with_hr(
    #         patient_steps[["dateTime", "cadence"]], patient_hr
    #     )
    #     patient_smoothed = get_smoothed_cadence_by_zone(
    #         data_merge, window_size=3
    #     )
    #     patient_smoothed["participant_id"] = pat_id
    #     smoothed_cadence_by_zone.append(patient_smoothed)

    # smoothed_cadence_by_zone = pd.concat(
    #     smoothed_cadence_by_zone, ignore_index=True
    # )
    # smoothed_cadence_by_zone = smoothed_cadence_by_zone[
    #     ["participant_id"]
    #     + [
    #         column
    #         for column in smoothed_cadence_by_zone.columns
    #         if column != "participant_id"
    #     ]
    # ]

    # for zone in range(1, 6):
    #     zone_name = f"zone{zone}"
    #     column_name = f"smoothed_cadence_by_{zone_name}"
    #     cadence_data = smoothed_cadence_by_zone[
    #         smoothed_cadence_by_zone["zone"].eq(zone_name)
    #     ]
    #     cadence_data = cadence_data[["participant_id", "date", "cadence"]].copy()
    #     data[column_name] = cadence_data

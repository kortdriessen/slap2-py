import os as os

import numpy as np
import polars as pl


def build_atomic_syndf(trial_data, refdata, trace_group="dF", trace_type="matchFilt"):

    ntrials = len(trial_data[1])
    fs = int(refdata["params"]["analyzeHz"][0][0])

    data_frames = []
    for dmd in trial_data.keys():
        trial_arrays = [
            trial_data[dmd][trl][trace_group][trace_type] for trl in range(ntrials)
        ]
        full_array = np.concatenate(trial_arrays, axis=0)
        full_array = full_array.swapaxes(0, 1)
        n_sources = full_array.shape[0]
        time = np.arange(0, full_array.shape[1] / fs, step=1 / fs)
        data_flat = full_array.flatten()
        # Build the source column: 0,0,...0, 1,1,...1, etc.
        source_flat = np.repeat(np.arange(n_sources), len(time))

        # Build the time column: time[0], time[1], ... for each source
        time_flat = np.tile(time.reshape(-1), n_sources)

        single_trial_len = trial_arrays[0].shape[0]
        trials_flat_single_source = np.repeat(
            np.arange(1, ntrials + 1), single_trial_len
        )
        trials_flat = np.tile(trials_flat_single_source, n_sources)

        # Construct Polars DataFrame
        df = pl.DataFrame({
            "source-ID": source_flat,
            "time": time_flat,
            "data": data_flat,
            "dmd": dmd,
            "trace_group": trace_group,
            "trace_type": trace_type,
            "trial": trials_flat,
            "channel": 2,
        })
        data_frames.append(df)
    return pl.concat(data_frames)


def build_noise_est_df(trial_data, trace_group="dF", trace_type="matchFilt"):

    noise_est_dataframes = []
    ntrials = len(trial_data[1])
    for dmd in trial_data.keys():
        trial_arrays = [
            trial_data[dmd][trl]["noiseEst"][trace_group][trace_type].flatten()
            for trl in range(ntrials)
        ]
        n_sources = trial_arrays[0].shape[0]
        trial_label_arrays = [
            np.ones_like(trial_arrays[0], dtype=int) * (trl + 1)
            for trl in range(ntrials)
        ]
        source_arrays = np.tile(np.arange(n_sources), len(trial_arrays))
        trial_arrays = np.array(trial_arrays).flatten()
        trial_label_arrays = np.array(trial_label_arrays).flatten()
        source_arrays = np.array(source_arrays).flatten()
        df = pl.DataFrame({
            "source-ID": source_arrays,
            "trial": trial_label_arrays,
            "noise": trial_arrays,
            "dmd": dmd,
            "trace_group": trace_group,
            "trace_type": trace_type,
        })
        noise_est_dataframes.append(df)
    return pl.concat(noise_est_dataframes)


def create_full_syndf(trial_data, refdata, trace_group="dF"):
    dF_keys_to_construct = ["matchFilt", "denoised", "nonneg", "events"]
    syndfs = []
    for key in dF_keys_to_construct:
        syndfs.append(
            build_atomic_syndf(
                trial_data, refdata, trace_group=trace_group, trace_type=key
            )
        )
    syndf = pl.concat(syndfs)
    noise_dfs = []
    for key in dF_keys_to_construct:
        noise_dfs.append(
            build_noise_est_df(trial_data, trace_group=trace_group, trace_type=key)
        )
    noise_df = pl.concat(noise_dfs)
    syndf = syndf.join(
        noise_df,
        on=["source-ID", "trial", "dmd", "trace_group", "trace_type"],
        how="left",
    )
    return syndf


def create_roidf(soma_info, trial_data, refdata):
    fs = int(refdata["params"]["analyzeHz"][0][0])
    roidfs = []
    for trace_type in ["F", "Fsvd"]:
        for dmd_id in soma_info.keys():
            dmd = int(dmd_id.split("-")[-1])
            for soma_ix, soma in enumerate(soma_info[dmd_id]):
                ntrials = len(trial_data[dmd])
                alltrials = [
                    trial_data[dmd][trl]["ROIs"][trace_type][:, :, soma_ix]
                    for trl in range(ntrials)
                ]
                all_trials = np.concatenate(alltrials, axis=1)
                soma_flat = all_trials.flatten()
                channels = np.array([2, 1])
                channels_full = np.repeat(channels, all_trials.shape[1])
                time = np.arange(0, all_trials.shape[1] / fs, step=1 / fs)
                time_full = np.tile(time, all_trials.shape[0])
                df = pl.DataFrame({
                    "time": time_full,
                    "data": soma_flat,
                    "channel": channels_full,
                    "dmd": dmd,
                    "soma-ID": soma,
                    "trace_type": trace_type,
                })
                roidfs.append(df)
    return pl.concat(roidfs)


def create_lsdf(trial_data, refdata, trace_group="dF", trace_type="ls"):

    ntrials = len(trial_data[1])
    fs = int(refdata["params"]["analyzeHz"][0][0])

    data_frames = []
    for i, channel_label in enumerate([2, 1]):
        for dmd in trial_data.keys():
            trial_arrays = [
                trial_data[dmd][trl][trace_group][trace_type][i]
                for trl in range(ntrials)
            ]
            full_array = np.concatenate(trial_arrays, axis=0)
            full_array = full_array.swapaxes(0, 1)
            n_sources = full_array.shape[0]
            time = np.arange(0, full_array.shape[1] / fs, step=1 / fs)
            data_flat = full_array.flatten()
            # Build the source column: 0,0,...0, 1,1,...1, etc.
            source_flat = np.repeat(np.arange(n_sources), len(time))

            # Build the time column: time[0], time[1], ... for each source
            time_flat = np.tile(time.reshape(-1), n_sources)

            single_trial_len = trial_arrays[0].shape[0]
            trials_flat_single_source = np.repeat(
                np.arange(1, ntrials + 1), single_trial_len
            )
            trials_flat = np.tile(trials_flat_single_source, n_sources)

            # Construct Polars DataFrame
            df = pl.DataFrame({
                "source-ID": source_flat,
                "time": time_flat,
                "data": data_flat,
                "dmd": dmd,
                "trace_group": trace_group,
                "trace_type": trace_type,
                "trial": trials_flat,
                "channel": channel_label,
            })
            data_frames.append(df)
    return pl.concat(data_frames)


def create_fzerodf(trial_data, refdata):

    ntrials = len(trial_data[1])
    fs = int(refdata["params"]["analyzeHz"][0][0])

    data_frames = []
    for i, channel_label in enumerate([2, 1]):
        for dmd in trial_data.keys():
            trial_arrays = [trial_data[dmd][trl]["F0"][i] for trl in range(ntrials)]
            full_array = np.concatenate(trial_arrays, axis=0)
            full_array = full_array.swapaxes(0, 1)
            n_sources = full_array.shape[0]
            time = np.arange(0, full_array.shape[1] / fs, step=1 / fs)
            data_flat = full_array.flatten()
            # Build the source column: 0,0,...0, 1,1,...1, etc.
            source_flat = np.repeat(np.arange(n_sources), len(time))

            # Build the time column: time[0], time[1], ... for each source
            time_flat = np.tile(time.reshape(-1), n_sources)

            single_trial_len = trial_arrays[0].shape[0]
            trials_flat_single_source = np.repeat(
                np.arange(1, ntrials + 1), single_trial_len
            )
            trials_flat = np.tile(trials_flat_single_source, n_sources)

            # Construct Polars DataFrame
            df = pl.DataFrame({
                "source-ID": source_flat,
                "time": time_flat,
                "data": data_flat,
                "dmd": dmd,
                "trace_group": "baseline",
                "trace_type": "F0",
                "trial": trials_flat,
                "channel": channel_label,
            })
            data_frames.append(df)
    return pl.concat(data_frames)

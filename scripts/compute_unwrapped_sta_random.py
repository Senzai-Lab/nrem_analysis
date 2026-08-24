import sys
from pathlib import Path

import numpy as np
import pynapple as nap


MOUSE_IDS_DUAL = ["99b", "103c", "106b", "107b", "110b"]

N_RANDOM_UNITS = 30
DEFAULT_MIN_RATE_HZ = 1.0
DEFAULT_MAX_RATE_HZ = 20.0
RANDOM_SEED = 0
PERIEVENT_CHUNK_SIZE = 2_000

STA_KWARGS = dict(
    bin_size=2,
    window=(-501, 501),
    time_unit="ms",
)


def generate_poisson_spikes(
    t_start: float,
    t_end: float,
    n_units: int,
    *,
    min_rate_hz: float,
    max_rate_hz: float,
    seed: int | None = None,
    time_support: nap.IntervalSet | None = None,
) -> nap.TsGroup:
    """Generate independent homogeneous Poisson spike trains in seconds."""

    rng = np.random.default_rng(seed)
    rates = rng.uniform(min_rate_hz, max_rate_hz, size=n_units)
    full_support = nap.IntervalSet(
        start=t_start,
        end=t_end,
        time_units="s",
    )
    if time_support is None:
        time_support = full_support
    else:
        time_support = time_support.intersect(full_support)
    if len(time_support) == 0:
        raise ValueError("time_support does not overlap [t_start, t_end]")

    spikes = {}
    for unit_id, rate_hz in enumerate(rates):
        timestamps = []
        for start, end in zip(time_support.start, time_support.end):
            n_spikes = rng.poisson(rate_hz * (end - start))
            timestamps.append(
                np.sort(rng.uniform(start, end, size=n_spikes))
            )
        timestamps = np.concatenate(timestamps)
        spikes[unit_id] = nap.Ts(
            t=timestamps,
            time_units="s",
            time_support=time_support,
        )

    return nap.TsGroup(
        spikes,
        time_support=time_support,
        metadata={"simulated_rate_hz": rates},
    )


def unwrap_epochs(data):
    d = data.d.copy()
    t = data.t

    for start, end in zip(
        data.time_support.start,
        data.time_support.end,
    ):
        i = np.searchsorted(t, start, side="left")
        j = np.searchsorted(t, end, side="right")
        d[i:j] = np.unwrap(d[i:j])

    return nap.Tsd(t=t, d=d, time_support=data.time_support)


def compute_unwrapped_sta(
    data,
    units,
    bin_size,
    window,
    time_unit,
    fill_value=np.nan,
    event_chunk_size=PERIEVENT_CHUNK_SIZE,
):
    if event_chunk_size <= 0:
        raise ValueError("event_chunk_size must be positive")

    stas = []
    times = None

    for uid in units.index:
        events = units[uid].restrict(data.time_support)

        if len(events) == 0:
            print(f"Filling unit {uid} with {fill_value}")
            stas.append(None)
            continue

        event_times = events.t
        lag_times = None
        summed = None
        counts = None

        for start in range(0, len(event_times), event_chunk_size):
            chunk = nap.Ts(
                t=event_times[start:start + event_chunk_size],
                time_units="s",
                time_support=events.time_support,
            )
            perievent = nap.compute_perievent(
                data=data,
                events=chunk,
                window=window,
                time_unit=time_unit,
            )

            zero_idx = np.argmin(np.abs(perievent.index))
            centered = perievent.d - perievent.d[zero_idx]

            if summed is None:
                lag_times = perievent.t
                summed = np.zeros(centered.shape[0], dtype=float)
                counts = np.zeros(centered.shape[0], dtype=np.int64)

            summed += np.nansum(centered, axis=1)
            counts += np.count_nonzero(~np.isnan(centered), axis=1)

        mean_sta = np.full(summed.shape, np.nan)
        np.divide(summed, counts, out=mean_sta, where=counts > 0)
        sta = nap.Tsd(
            t=lag_times,
            d=mean_sta,
            time_support=perievent.time_support,
        )
        sta = sta.bin_average(bin_size=bin_size, time_units=time_unit)
        zero_idx = np.argmin(np.abs(sta.index))
        sta = sta - sta[zero_idx]

        times = sta.t
        stas.append(sta.d)

    if times is None:
        raise ValueError("No units have events within data.time_support.")

    stas = [
        np.full(times.shape, fill_value) if sta is None else sta
        for sta in stas
    ]

    return nap.TsdFrame(
        t=times,
        d=np.column_stack(stas),
        columns=units.index,
        metadata={
            "simulated_rate_hz": units["simulated_rate_hz"].to_numpy(),
        },
    )


def main(
    input_path: str,
    output_path: str,
    task_id: int,
    total_tasks: int,
):
    print(f"Input path: {input_path}")
    print(f"Output path: {output_path}")
    input_path = Path(input_path)
    output_path = Path(output_path)

    task_id = int(task_id)
    total_tasks = int(total_tasks)
    min_rate_hz = DEFAULT_MIN_RATE_HZ
    max_rate_hz = DEFAULT_MAX_RATE_HZ

    if total_tasks <= 0:
        raise ValueError("total_tasks must be positive")
    if not 0 <= task_id < total_tasks:
        raise ValueError("task_id must satisfy 0 <= task_id < total_tasks")

    print(f"Task ID: {task_id} ({task_id}/{total_tasks})")
    print(f"Random units per mouse: {N_RANDOM_UNITS}")
    print(f"Random rate range: [{min_rate_hz}, {max_rate_hz}] Hz")

    selected_mice = [
        mouse_id
        for mouse_idx, mouse_id in enumerate(MOUSE_IDS_DUAL)
        if mouse_idx % total_tasks == task_id
    ]

    print(f"Selected mice for this task: {len(selected_mice)}")
    for mouse_id in selected_mice:
        print(f"  {mouse_id}")

    for mouse_idx, mouse_id in enumerate(MOUSE_IDS_DUAL):
        if mouse_idx % total_tasks != task_id:
            continue

        data_dir = input_path / mouse_id
        save_dir = output_path / mouse_id
        save_dir.mkdir(parents=True, exist_ok=True)

        print(
            f"Processing mouse {mouse_id} "
            f"({mouse_idx + 1}/{len(MOUSE_IDS_DUAL)})"
        )
        print(f"Data directory: {data_dir}")
        print("-" * 100)

        virtual_hd = nap.load_file(data_dir / "virtual_hd.npz")
        random_units = generate_poisson_spikes(
            t_start=virtual_hd.start_time(),
            t_end=virtual_hd.end_time(),
            n_units=N_RANDOM_UNITS,
            min_rate_hz=min_rate_hz,
            max_rate_hz=max_rate_hz,
            seed=RANDOM_SEED + mouse_idx,
            time_support=virtual_hd.time_support,
        )

        print("Computing whole-session STA with random spikes")
        sta_random = compute_unwrapped_sta(
            data=unwrap_epochs(np.deg2rad(virtual_hd)),
            units=random_units,
            **STA_KWARGS,
        )

        output_file = save_dir / "sta_random_spikes.npz"
        print(f"Saving random-spike STA: {output_file}")
        sta_random.save(output_file)

        print(f"Finished mouse: {mouse_id}")
        print("-" * 100)


if __name__ == "__main__":
    print("Python version")
    print(sys.version)
    print("Version info")
    print(sys.version_info)

    if len(sys.argv) not in (5, 7):
        print(f"Incorrect number of arguments: {len(sys.argv) - 1}")
        print(
            "Usage: python compute_unwrapped_sta_random.py "
            "<input_path> <output_path> "
            "<task_id> <total_tasks> "
        )
        sys.exit(1)

    main(
        sys.argv[1],
        sys.argv[2],
        sys.argv[3],
        sys.argv[4],
    )

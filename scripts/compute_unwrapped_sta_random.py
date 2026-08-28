import sys
from pathlib import Path

import numpy as np
import pynapple as nap
from scipy import stats


MOUSE_IDS_DUAL = ["99b", "103c", "106b", "107b", "110b"]

N_RANDOM_UNITS = 30
DEFAULT_MIN_RATE_HZ = 1.0
DEFAULT_MAX_RATE_HZ = 20.0
RANDOM_SEED = 0

NREM_STA_KWARGS = dict(
    window=(-51, 51),
    time_unit="ms",
)


def detect_upstates(
    spikes: nap.TsGroup,
    bin_size=0.01,
    percentile=80,
    merge_thr=0.1,
    short_thr=0.04,
    long_thr=2,
    time_units="s",
):
    population_rate = spikes.count(
        bin_size=bin_size,
        time_units=time_units,
    ).sum(axis=1)
    population_rate = population_rate / bin_size
    population_rate = population_rate.smooth(
        std=2 * bin_size,
        time_units=time_units,
    )
    threshold = np.percentile(population_rate, q=percentile)

    up_states = population_rate.threshold(
        threshold,
        method="above",
    ).time_support
    up_states = up_states.merge_close_intervals(
        threshold=merge_thr,
        time_units=time_units,
    )
    up_states = up_states.drop_short_intervals(
        threshold=short_thr,
        time_units=time_units,
    )
    up_states = up_states.drop_long_intervals(
        threshold=long_thr,
        time_units=time_units,
    )
    return up_states, threshold


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


def unwrap_epochs(data: nap.Tsd) -> nap.Tsd:
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
    data: nap.Tsd,
    units: nap.TsGroup,
    *,
    window,
    time_unit,
):
    """Compute complete-event STAs and SEMs after zero-lag centering."""

    stas = []
    errors = []
    times = None

    for uid in units.index:
        events = units[uid].restrict(data.time_support)

        if len(events) == 0:
            print(f"No events in {uid}. Filling with NaN")
            stas.append(None)
            errors.append(None)
            continue

        perievent = nap.compute_perievent(
            data=data,
            events=events,
            window=window,
            time_unit=time_unit,
        )

        complete_events = ~np.any(np.isnan(perievent.d), axis=0)
        perievent = perievent[:, complete_events]
        if perievent.shape[1] == 0:
            print(f"No complete events in {uid}. Filling with NaN")
            stas.append(None)
            errors.append(None)
            continue

        zero_idx = np.argmin(np.abs(perievent.index))
        perievent = perievent - perievent[zero_idx]

        sta = np.mean(perievent, axis=1)
        stas.append(sta.d)
        errors.append(stats.sem(perievent.d, axis=1))
        times = sta.t

    if times is None:
        raise ValueError(
            "No units have complete events within data.time_support."
        )

    stas = [
        np.full(times.shape, np.nan) if sta is None else sta
        for sta in stas
    ]
    errors = [
        np.full(times.shape, np.nan) if error is None else error
        for error in errors
    ]

    sta = nap.TsdFrame(
        t=times,
        d=np.column_stack(stas),
        columns=units.index,
    )
    error = nap.TsdFrame(
        t=times,
        d=np.column_stack(errors),
        columns=units.index,
    )
    return sta, error


def process_mouse(
    data_dir: Path,
    save_dir: Path,
    mouse_idx: int,
):
    turn_units = nap.load_file(data_dir / "turn_units.npz")
    hd_units = nap.load_file(data_dir / "hd_units.npz")
    virtual_hd = nap.load_file(data_dir / "virtual_hd.npz")
    sleep = nap.load_file(data_dir / "sleep.npz")
    session = nap.load_file(data_dir / "session.npz")

    nrem = sleep[sleep["state"] == "nrem"].intersect(
        session[session["state"] == "homecage"]
    )
    print("Detecting NREM upstates")
    up_states, _ = detect_upstates(hd_units.restrict(nrem))
    up_hd = virtual_hd.restrict(up_states)
    up_hd = unwrap_epochs(np.deg2rad(up_hd))

    print("Computing NREM upstate STA with real spikes")
    sta_real, _ = compute_unwrapped_sta(
        data=up_hd,
        units=turn_units,
        **NREM_STA_KWARGS,
    )
    real_output = save_dir / "sta_nrem_upstate.npz"
    print(f"Saving real-spike STA: {real_output}")
    sta_real.save(real_output)

    random_units = generate_poisson_spikes(
        t_start=virtual_hd.start_time(),
        t_end=virtual_hd.end_time(),
        n_units=N_RANDOM_UNITS,
        min_rate_hz=DEFAULT_MIN_RATE_HZ,
        max_rate_hz=DEFAULT_MAX_RATE_HZ,
        seed=RANDOM_SEED + mouse_idx,
        time_support=up_hd.time_support,
    )

    print("Computing NREM upstate STA with random spikes")
    sta_random, _ = compute_unwrapped_sta(
        data=up_hd,
        units=random_units,
        **NREM_STA_KWARGS,
    )
    random_output = save_dir / "sta_random_spikes.npz"
    print(f"Saving random-spike STA: {random_output}")
    sta_random.save(random_output)


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
    if total_tasks <= 0:
        raise ValueError("total_tasks must be positive")
    if not 0 <= task_id < total_tasks:
        raise ValueError("task_id must satisfy 0 <= task_id < total_tasks")

    selected_mice = [
        (mouse_idx, mouse_id)
        for mouse_idx, mouse_id in enumerate(MOUSE_IDS_DUAL)
        if mouse_idx % total_tasks == task_id
    ]

    print(f"Task ID: {task_id} ({task_id}/{total_tasks})")
    print(f"Random units per mouse: {N_RANDOM_UNITS}")
    print(
        "Random rate range: "
        f"[{DEFAULT_MIN_RATE_HZ}, {DEFAULT_MAX_RATE_HZ}] Hz"
    )
    print(f"Selected mice for this task: {len(selected_mice)}")
    for _, mouse_id in selected_mice:
        print(f"  {mouse_id}")

    for mouse_idx, mouse_id in selected_mice:
        data_dir = input_path / mouse_id
        save_dir = output_path / mouse_id
        save_dir.mkdir(parents=True, exist_ok=True)

        print(
            f"Processing mouse {mouse_id} "
            f"({mouse_idx + 1}/{len(MOUSE_IDS_DUAL)})"
        )
        print(f"Data directory: {data_dir}")
        print("-" * 100)

        process_mouse(data_dir, save_dir, mouse_idx)

        print(f"Finished mouse: {mouse_id}")
        print("-" * 100)


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print(f"Incorrect number of arguments: {len(sys.argv) - 1}")
        print(
            "Usage: python compute_unwrapped_sta_random.py "
            "<input_path> <output_path> "
            "<task_id> <total_tasks>"
        )
        sys.exit(1)

    main(
        sys.argv[1],
        sys.argv[2],
        sys.argv[3],
        sys.argv[4],
    )

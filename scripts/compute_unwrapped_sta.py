import sys
from pathlib import Path

import numpy as np
import pynapple as nap


MOUSE_IDS_DUAL = ["99b", "103c", "106b", "107b", "110b"]

NREM_UPSTATE = dict(
    bin_size=2,
    window=(-501, 501),
    time_unit="ms",
)

def detect_upstates(
    spikes: nap.TsGroup,
    bin_size = 0.01,
    percentile = 80,
    merge_thr = 0.1,
    short_thr = 0.04,
    long_thr = 2,
    time_units='s'
    ) -> nap.IntervalSet:
    x = spikes.count(bin_size=bin_size, time_units=time_units).sum(axis=1)
    x = x / bin_size
    x = x.smooth(std=2 * bin_size, time_units=time_units)
    thr = np.percentile(x, q=percentile)

    up_states = x.threshold(thr, method='above').time_support
    up_states = up_states.merge_close_intervals(threshold=merge_thr, time_units=time_units)
    up_states = up_states.drop_short_intervals(threshold=short_thr, time_units=time_units)
    up_states = up_states.drop_long_intervals(threshold=long_thr, time_units=time_units)
    return up_states, thr

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

    return nap.Tsd(t=t, d=d, time_support=data.time_support,)

def compute_unwrapped_sta(
    data,
    units,
    bin_size,
    window,
    time_unit,
    fill_value=np.nan,
):
    stas = []
    times = None

    for uid in units.index:
        events = units[uid].restrict(data.time_support)

        if len(events) == 0:
            print(f"Filling unit {uid} with {fill_value}")
            stas.append(None)
            continue

        perievent = nap.compute_perievent(
            data=data,
            events=events,
            window=window,
            time_unit=time_unit,
        )

        zero_idx = np.argmin(np.abs(perievent.index))
        sta = np.nanmean(perievent - perievent[zero_idx], axis=1)
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
    print(f"Task ID: {task_id} ({task_id}/{total_tasks})")

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

        hd_units        = nap.load_file(data_dir / "hd_units.npz")
        turn_units      = nap.load_file(data_dir / "turn_units.npz")
        sleep           = nap.load_file(data_dir / "sleep.npz")
        session         = nap.load_file(data_dir / "session.npz")
        virtual_hd      = nap.load_file(data_dir / "virtual_hd.npz")

        nrem            = sleep[sleep['state'] == 'nrem'].intersect(session[session['state'] == 'homecage'])
        print("Detecting upstates")
        upstate, _ = detect_upstates(hd_units.restrict(nrem))
        upstate_decoded = virtual_hd.restrict(upstate)

        print("Computing NREM STA")
        sta_nrem = compute_unwrapped_sta(
            data=unwrap_epochs(np.deg2rad(upstate_decoded)),
            units=turn_units,
            **NREM_UPSTATE,
        )

        nrem_output = save_dir / 'sta_nrem_upstate.npz'
        print(f"Saving NREM STA: {nrem_output}")
        sta_nrem.save(nrem_output)

        print(f"Finished mouse: {mouse_id}")
        print("-" * 100)


if __name__ == "__main__":
    print("Python version")
    print(sys.version)
    print("Version info")
    print(sys.version_info)

    if len(sys.argv) != 5:
        print(f"Incorrect number of arguments: {len(sys.argv) - 1}")
        print(
            "Usage: python sta.py "
            "<input_path> <output_path> "
            "[task_id] [total_tasks]"
        )
        sys.exit(1)

    main(
        sys.argv[1],
        sys.argv[2],
        sys.argv[3],
        sys.argv[4],
        )
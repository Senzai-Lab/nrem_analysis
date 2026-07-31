import sys
from pathlib import Path

import numpy as np
import pynapple as nap


MOUSE_IDS_DUAL = ["99b", "103c", "106b", "107b", "110b"]

AWAKE_STA_KWARGS = dict(
    bin_size=10,
    window=(-500, 500),
    time_unit="ms",
)

NREM_STA_KWARGS = dict(
    bin_size=2,
    window=(-100, 100),
    time_unit="ms",
)

def unwrap_epochs(data):
    values = data.d.copy()
    times = data.t

    for start, end in zip(
        data.time_support.start,
        data.time_support.end,
    ):
        first = np.searchsorted(times, start, side="left")
        last = np.searchsorted(times, end, side="right")
        values[first:last] = np.unwrap(values[first:last])

    return nap.Tsd(
        t=times,
        d=values,
        time_support=data.time_support,
    )

def compute_unwrapped_sta(
    data,
    units,
    bin_size,
    window,
    time_unit,
    fill_value=np.nan,
):
    unit_stas = []
    for uid in units.index:
        events = units[uid].restrict(data.time_support)

        if len(events) == 0:
            print(f"Filling unit {uid} with {fill_value}")
            unit_stas.append(None)
            continue

        result = nap.compute_perievent(data=data, events=events, window=window, time_unit=time_unit,)
        # remove baseline
        zero_idx = np.argmin(np.abs(result.index))
        result = result - result[zero_idx]
        result = np.nanmean(result, axis=1)
        result = result.bin_average(bin_size=bin_size, time_units=time_unit)

        unit_stas.append(result.d)

    unit_stas = [
        np.full(result.t.shape, fill_value) if sta is None else sta
        for sta in unit_stas
    ]

    return nap.TsdFrame(
        t=result.t,
        d=np.column_stack(unit_stas),
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

        turn_units      = nap.load_file(data_dir / "turn_units.npz")
        head_direction  = nap.load_file(data_dir / "head_direction.npz")
        sweeps          = nap.load_file(data_dir / "sweeps.npz")
        
        print("Computing awake STA")
        sta_awake = compute_unwrapped_sta(
            data=unwrap_epochs(np.deg2rad(head_direction)),
            units=turn_units,
            **AWAKE_STA_KWARGS,
        )

        awake_output = save_dir / 'sta_awake.npz'
        print(f"Saving awake STA: {awake_output}")
        sta_awake.save(awake_output)

        print("Computing continuous-NREM STA")
        sta_nrem = compute_unwrapped_sta(
            data=unwrap_epochs(np.deg2rad(sweeps)),
            units=turn_units,
            **NREM_STA_KWARGS,
        )

        nrem_output = save_dir / 'sta_nrem.npz'
        print(f"Saving continuous-NREM STA: {nrem_output}")
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
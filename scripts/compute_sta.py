import sys
from pathlib import Path

import numpy as np
import pynapple as nap


MOUSE_IDS_DUAL = ["99b", "103c", "106b", "107b", "110b"]
STATE_NAMES = ["continuous", "fragmented", "stationary"]
STA_KINDS = ["awake", "nrem", "nrem_cont"]
AWAKE_STA_KWARGS = dict(binsize=10, window=(-2000, 2000), time_unit="ms")
NREM_STA_KWARGS = dict(binsize=1, window=(-200, 200), time_unit="ms")


def compute_circular_sta(angles, events, **kwargs):
    angles = np.deg2rad(angles)
    angles_cartesian = np.column_stack([np.sin(angles), np.cos(angles)])
    sta = nap.compute_event_triggered_average(
            angles_cartesian,
            events,
            **kwargs,
            )
    return nap.TsdFrame(
        t=sta.t,
        d=np.rad2deg(np.arctan2(sta[:, :, 0].d, sta[:, :, 1].d)),
    )


def load_virtual_hds(input_path, data_dir, mouse_id, with_states=False):
    print(f"Loading sleep/session epochs for {mouse_id}")
    sleep = nap.load_file(input_path / mouse_id / "sleep.npz")
    session = nap.load_file(input_path / mouse_id / "session.npz")
    nrem = sleep[sleep["state"] == "nrem"]
    nrem_homecage = nrem.intersect(session[session["state"] == "homecage"])
    print(f"Found {len(nrem_homecage)} NREM homecage epochs")

    virtual_hds = []
    states = []
    for i, _ in enumerate(nrem_homecage):
        position_path = data_dir / f"{mouse_id}_{i}_position.npz"
        print(
            f"Loading decoded position for epoch {i + 1}/{len(nrem_homecage)}: "
            f"{position_path}"
        )
        virtual_hds.append(nap.load_file(position_path))

        if with_states:
            states_path = data_dir / f"{mouse_id}_{i}_states.npz"
            print(
                f"Loading decoded states for epoch {i + 1}/{len(nrem_homecage)}: "
                f"{states_path}"
            )
            states.append(nap.load_file(states_path))

    virtual_hds = np.concatenate(virtual_hds)
    print(f"Loaded virtual HD samples: {virtual_hds.shape[0]}")
    if not with_states:
        return virtual_hds

    states = np.concatenate(states)
    print(f"Loaded state samples: {states.shape[0]}")
    return virtual_hds, states


def main(input_path: str, output_path: str, task_id: int = None, total_tasks: int = None):
    print(f"Input path: {input_path}")
    print(f"Output path: {output_path}")
    if task_id is not None and total_tasks is not None:
        task_id = int(task_id)
        total_tasks = int(total_tasks)
        print(f"Task ID: {task_id} ({task_id}/{total_tasks})")

    input_path = Path(input_path)
    output_path = Path(output_path)

    jobs = [
        (mouse_id, sta_kind)
        for mouse_id in MOUSE_IDS_DUAL
        for sta_kind in STA_KINDS
    ]
    print(
        f"Total STA jobs: {len(jobs)} "
        f"({len(MOUSE_IDS_DUAL)} mice x {len(STA_KINDS)} STA kinds)"
    )
    if task_id is not None and total_tasks is not None:
        selected_jobs = [
            (job_idx, mouse_id, sta_kind)
            for job_idx, (mouse_id, sta_kind) in enumerate(jobs)
            if job_idx % total_tasks == task_id
        ]
        print(f"Selected jobs for this task: {len(selected_jobs)}")
        for job_idx, mouse_id, sta_kind in selected_jobs:
            print(f"  Job {job_idx}: {mouse_id} {sta_kind}")
    else:
        print("No task split requested; running all jobs")

    for job_idx, (mouse_id, sta_kind) in enumerate(jobs):
        if task_id is not None and total_tasks is not None:
            if job_idx % total_tasks != task_id:
                continue

        data_dir = output_path / mouse_id
        print(f"Processing {mouse_id}: {sta_kind} ({job_idx}/{len(jobs)})")
        print(f"Data directory: {data_dir}")
        print(f"--" * 50)

        sc_units_path = input_path / mouse_id / "sc_units.npz"
        print(f"Loading SC units: {sc_units_path}")
        sc_units = nap.load_file(sc_units_path)
        print(f"Loaded SC units: {len(sc_units)}")

        if sta_kind == "awake":
            head_direction_path = input_path / mouse_id / "head_direction.npz"
            output_file = data_dir / "sta_awake_sc.npz"
            print(f"Loading head direction: {head_direction_path}")
            head_direction = nap.load_file(head_direction_path)
            print("Computing awake STA")
            sta = compute_circular_sta(head_direction, sc_units, **AWAKE_STA_KWARGS)
            print(f"Saving STA: {output_file}")
            sta.save(output_file)

        elif sta_kind == "nrem":
            output_file = data_dir / "sta_nrem_sc.npz"
            virtual_hds = load_virtual_hds(input_path, data_dir, mouse_id)
            print("Computing NREM STA")
            sta = compute_circular_sta(virtual_hds, sc_units, **NREM_STA_KWARGS)
            print(f"Saving STA: {output_file}")
            sta.save(output_file)

        elif sta_kind == "nrem_cont":
            output_file = data_dir / "sta_nrem_cont_sc.npz"
            virtual_hds, states = load_virtual_hds(
                input_path,
                data_dir,
                mouse_id,
                with_states=True,
            )
            print("Selecting continuous epochs with probability > 0.5")
            continuous_eps = states["continuous"].threshold(0.5).time_support
            print(f"Continuous epochs: {len(continuous_eps)}")
            print("Computing continuous NREM STA")
            sta = compute_circular_sta(
                virtual_hds,
                sc_units,
                **NREM_STA_KWARGS,
                epochs=continuous_eps,
            )
            print(f"Saving STA: {output_file}")
            sta.save(output_file)

        print(f"Finished: {mouse_id}: {sta_kind}")
        print(f"--" * 50)


if __name__ == "__main__":
    print("Python version")
    print(sys.version)
    print("Version info.")
    print(sys.version_info)

    if len(sys.argv) != 5 and len(sys.argv) != 3:
        print(f"Incorrect number of arguments: {len(sys.argv)-1}.")
        print("Usage: python compute_sta_refactored.py <input_path> <output_path> [task_id] [total_tasks]")
        sys.exit(1)

    if len(sys.argv) == 5:
        main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        main(sys.argv[1], sys.argv[2])

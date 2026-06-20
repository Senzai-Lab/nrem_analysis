import sys
from pathlib import Path

import numpy as np
import pynapple as nap


MOUSE_IDS_DUAL = ["99b", "103c", "106b", "107b", "110b"]
STATE_NAMES = ["continuous", "fragmented", "stationary"]
AWAKE_STA_KWARGS = dict(binsize=10, window=(-2000, 2000), time_unit="ms")
NREM_STA_KWARGS = dict(binsize=1, window=(-200, 200), time_unit="ms")


def compute_circular_sta(angles, events, **kwargs):
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


def main(input_path: str, output_path: str, task_id: int = None, total_tasks: int = None):
    print(f"Input path: {input_path}")
    print(f"Output path: {output_path}")
    if task_id is not None and total_tasks is not None:
        task_id = int(task_id)
        total_tasks = int(total_tasks)
        print(f"Task ID: {task_id} ({task_id}/{total_tasks})")

    input_path = Path(input_path)
    output_path = Path(output_path)

    for job_idx, mouse_id in enumerate(MOUSE_IDS_DUAL):
        if task_id is not None and total_tasks is not None:
            if job_idx % total_tasks != task_id:
                continue

        data_dir = output_path / mouse_id
        print(f"Processing {mouse_id}")
        print(f"--" * 50)

        sleep           = nap.load_file(input_path / mouse_id / "sleep.npz")
        turn_units      = nap.load_file(input_path / mouse_id / "turn_units.npz")
        head_direction  = nap.load_file(input_path / mouse_id / "head_direction.npz")
        session         = nap.load_file(input_path / mouse_id / "session.npz")
        nrem            = sleep[sleep["state"] == "nrem"]
        nrem_homecage = nrem.intersect(session[session["state"] == "homecage"])

        virtual_hds = []
        states = []
        for i, _ in enumerate(nrem_homecage):
            decoded = nap.load_file(data_dir / f"{mouse_id}_{i}.npz")
            virtual_hds.append(nap.Tsd(t=decoded.t, d=decoded.values[:, 3]))
            states.append(
                nap.TsdFrame(
                    t=decoded.t,
                    d=decoded.values[:, :3],
                    columns=STATE_NAMES,
                )
            )

        virtual_hds = np.concatenate(virtual_hds)
        states = np.concatenate(states)

        sta_awake = compute_circular_sta(head_direction, turn_units, **AWAKE_STA_KWARGS)
        sta_awake.save(data_dir / f"sta_awake.npz")

        sta_nrem = compute_circular_sta(virtual_hds, turn_units, **NREM_STA_KWARGS)
        sta_nrem.save(data_dir / f"sta_nrem.npz")

        continuous_eps = states["continuous"].threshold(0.5).time_support
        sta = compute_circular_sta(virtual_hds, turn_units, **NREM_STA_KWARGS, epochs=continuous_eps)
        sta.save(data_dir / "sta_nrem_cont.npz")

        print(f"Finished: {mouse_id}")
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
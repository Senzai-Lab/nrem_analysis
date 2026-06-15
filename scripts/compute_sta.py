import sys
from pathlib import Path
from itertools import product

import numpy as np
import pynapple as nap

MOUSE_IDS_DUAL = ["99b", "103c", "106b", "107b", "110b"]
MOVEMENT_VARS = ["movement_var2", "movement_var20"]


def main(input_path: str, output_path: str, task_id: int = None, total_tasks: int = None):
    print(f"Input path: {input_path}")
    print(f"Output path: {output_path}")
    if task_id is not None and total_tasks is not None:
        task_id = int(task_id)
        total_tasks = int(total_tasks)
        print(f"Task ID: {task_id} ({task_id}/{total_tasks})")

    input_path = Path(input_path)
    output_path = Path(output_path)

    jobs = list(product(MOVEMENT_VARS, MOUSE_IDS_DUAL))

    for job_idx, (mv, mouse_id) in enumerate(jobs):
        if task_id is not None and total_tasks is not None:
            if job_idx % total_tasks != task_id:
                continue

        data_dir = output_path / mv / mouse_id
        print(f"Processing {mv} / {mouse_id}")
        print(f"--" * 50)

        if not data_dir.exists():
            print(f"Data dir {data_dir} does not exist, skipping...")
            continue

        sleep = nap.load_file(PROCESSED_DIR / "dual" / mouse_id / "sleep.npz")
        turn_units = nap.load_file(PROCESSED_DIR / "dual" / mouse_id / "turn_units.npz")
        head_direction = nap.load_file(PROCESSED_DIR / "dual" / mouse_id / "head_direction.npz")
        session = nap.load_file(PROCESSED_DIR / "dual" / mouse_id / "session.npz")
        nrem = sleep[sleep['state'] == 'nrem']

        virtual_hds = []
        states = []
        for i, ep in enumerate(nrem.intersect(session[session['state'] == 'homecage'])):
            decoded = nap.load_file(data_dir / f"{mouse_id}_{i}.npz")
            vh = nap.Tsd(t=decoded.t, d=decoded.values[:, 3])
            virtual_hds.append(vh)
            states_i = nap.TsdFrame(t=decoded.t, d=decoded.values[:, :3], columns=STATE_NAMES)
            states.append(states_i)

        virtual_hds = np.concatenate(virtual_hds)
        states = np.concatenate(states)
        for angles, state in zip([head_direction, np.concatenate(virtual_hds)], ["awake", "nrem"]):
            print(f"Computing STA for {state}...")
            angles = np.deg2rad(angles)
            angles_cart = np.column_stack([np.sin(angles), np.cos(angles)])
            sta = nap.compute_event_triggered_average(angles_cart, turn_units, binsize=1, window=(-500, 500), time_unit='ms')
            sta = nap.TsdFrame(t=sta.t, d=np.rad2deg(np.arctan2(sta[:, :, 0].d, sta[:, :, 1].d)))
            sta.save(data_dir / f"sta_{state}.npz")
        
        # Compute STA only during continuous epochs of NREM
        continuous_eps = states['continuous'].threshold(0.5).time_support
        angles = np.deg2rad(np.concatenate(virtual_hds))
        angles_cart = np.column_stack([np.sin(angles), np.cos(angles)])
        sta = nap.compute_event_triggered_average(angles_cart, turn_units, binsize=1, window=(-500, 500), time_unit='ms', epochs=continuous_eps)
        sta.save(data_dir / f"sta_nrem_cont.npz")

        print(f"Finished: {mv} / {mouse_id}")
        print(f"--" * 50)



if __name__ == "__main__":
    print("Python version")
    print(sys.version)
    print("Version info.")
    print(sys.version_info)

    if len(sys.argv) != 5 and len(sys.argv) != 3:
        print(f"Incorrect number of arguments: {len(sys.argv)-1}.")
        print("Usage: python compute_sta.py <input_path> <output_path> [task_id] [total_tasks]")
        sys.exit(1)

    if len(sys.argv) == 5:
        main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        main(sys.argv[1], sys.argv[2])
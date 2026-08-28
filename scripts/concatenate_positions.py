import sys
from pathlib import Path
from nrem_analysis.constant import MOUSE_IDS_TTX

import numpy as np
import pynapple as nap


def save_atomic(data, path: Path):
    temporary_path = path.with_suffix(".tmp.npz")
    data.save(temporary_path)
    temporary_path.replace(path)


def concatenate_positions(input_path: Path, output_path: Path, mouse_id: str):
    sleep = nap.load_file(input_path / mouse_id / "sleep.npz")
    sessions = nap.load_file(input_path / mouse_id / "sessions.npz")
    predict_ep = sleep[sleep["state"] == "nrem"].intersect(
        sessions[sessions["label"] == "homecage"]
    )

    data_dir = output_path / mouse_id
    save_dir = input_path / mouse_id
    position_paths = [
        data_dir / f"{mouse_id}_{i}_position.npz"
        for i in range(len(predict_ep))
    ]
    state_paths = [
        data_dir / f"{mouse_id}_{i}_states.npz"
        for i in range(len(predict_ep))
    ]
    if not position_paths:
        raise ValueError(f"No NREM homecage epochs found for {mouse_id}.")
    missing_paths = [
        path for path in position_paths + state_paths if not path.is_file()
    ]
    if missing_paths:
        raise FileNotFoundError(
            f"Missing decoded files for {mouse_id}: "
            f"{[path.name for path in missing_paths]}"
        )

    print(
        f"Concatenating {len(position_paths)} position and state files "
        f"for {mouse_id}"
    )
    virtual_hd = np.concatenate([nap.load_file(path) for path in position_paths])
    virtual_states = np.concatenate([nap.load_file(path) for path in state_paths])

    virtual_hd_path = save_dir / "virtual_hd.npz"
    print(f"Saving concatenated position: {virtual_hd_path}")
    save_atomic(virtual_hd, virtual_hd_path)

    virtual_states_path = save_dir / "virtual_states.npz"
    print(f"Saving concatenated states: {virtual_states_path}")
    save_atomic(virtual_states, virtual_states_path)


def main(input_path: str, output_path: str):
    print(f"Input path: {input_path}")
    print(f"Output path: {output_path}")
    input_path = Path(input_path)
    output_path = Path(output_path)

    for mouse_id in MOUSE_IDS_TTX:
        concatenate_positions(input_path, output_path, mouse_id)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Incorrect number of arguments: {len(sys.argv) - 1}.")
        print(
            "Usage: python concatenate_positions.py "
            "<input_path> <output_path>"
        )
        sys.exit(1)

    main(sys.argv[1], sys.argv[2])

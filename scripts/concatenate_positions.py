import sys
from pathlib import Path

import numpy as np
import pynapple as nap


MOUSE_IDS_DUAL = ["99b", "103c", "106b", "107b", "110b"]


def concatenate_positions(input_path: Path, output_path: Path, mouse_id: str):
    sleep = nap.load_file(input_path / mouse_id / "sleep.npz")
    session = nap.load_file(input_path / mouse_id / "session.npz")
    predict_ep = sleep[sleep["state"] == "nrem"].intersect(
        session[session["state"] == "homecage"]
    )

    data_dir = output_path / mouse_id
    position_paths = [
        data_dir / f"{mouse_id}_{i}_position.npz"
        for i in range(len(predict_ep))
    ]
    missing_indices = [
        i for i, path in enumerate(position_paths) if not path.is_file()
    ]
    if missing_indices:
        raise FileNotFoundError(
            f"Missing decoded position files for {mouse_id}, "
            f"epoch indices: {missing_indices}"
        )
    if not position_paths:
        raise ValueError(f"No NREM homecage epochs found for {mouse_id}.")

    print(f"Concatenating {len(position_paths)} position files for {mouse_id}")
    position_epochs = [nap.load_file(path) for path in position_paths]
    virtual_hd = np.concatenate(position_epochs)

    virtual_hd_path = data_dir / "virtual_hd.npz"
    print(f"Saving concatenated position: {virtual_hd_path}")
    temporary_path = virtual_hd_path.with_suffix(".tmp.npz")
    virtual_hd.save(temporary_path)
    temporary_path.replace(virtual_hd_path)


def main(input_path: str, output_path: str):
    print(f"Input path: {input_path}")
    print(f"Output path: {output_path}")
    input_path = Path(input_path)
    output_path = Path(output_path)

    for mouse_id in MOUSE_IDS_DUAL:
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

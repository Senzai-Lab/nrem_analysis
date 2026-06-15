import sys
from pathlib import Path

import numpy as np
import pynapple as nap

from replay_trajectory_classification import (
    SortedSpikesClassifier,
    Environment,
    RandomWalk,
    Uniform,
    Identity,
    DiagonalDiscrete,
    make_track_graph,
)

STATE_NAMES = ["continuous", "fragmented", "stationary"]
MOUSE_IDS_DUAL = ["99b", "103c", "106b", "107b", "110b", "111b"]
BIN_SIZE_S = 0.001

def get_environment(num_nodes: int = 360, place_bin_size: float = 1.0):
    radius = 180 / np.pi
    angle = np.linspace(2 * np.pi, 0, num=num_nodes, endpoint=False)
    node_positions = np.stack((radius * np.cos(angle), radius * np.sin(angle)), axis=1)

    node_ids = np.arange(node_positions.shape[0])
    edges = np.stack((node_ids, np.roll(node_ids, shift=1)), axis=1)

    track_graph = make_track_graph(node_positions, edges)

    n_nodes = len(track_graph.nodes)
    edge_order = np.stack(
        (np.roll(np.arange(n_nodes - 1, -1, -1), 1),
         np.arange(n_nodes - 1, -1, -1)),
        axis=1,
    )

    return Environment(
        place_bin_size=place_bin_size,
        track_graph=track_graph,
        edge_order=edge_order,
        edge_spacing=0,
    )

def fit_classifier(head_direction, train_spikes, movement_var=2.0, state_prob=0.99):
    environment = get_environment()
    continuous_transition_types = [
        [RandomWalk(movement_var=movement_var), Uniform(), Identity()],
        [Uniform(),                             Uniform(), Uniform()],
        [RandomWalk(movement_var=movement_var), Uniform(), Identity()],
    ]
    classifier = SortedSpikesClassifier(
        environments=environment,
        continuous_transition_types = continuous_transition_types,
        discrete_transition_type=DiagonalDiscrete(state_prob),
        )
    classifier.fit(head_direction, train_spikes)
    return classifier

def main(input_path: str, output_path: str, task_id: int = None, total_tasks: int = None):
    print(f"Input path: {input_path}")
    print(f"Output path: {output_path}")
    if task_id is not None and total_tasks is not None:
        task_id = int(task_id)
        total_tasks = int(total_tasks)
        print(f"Task ID: {task_id} ({task_id}/{total_tasks})")


    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load data
    for subject_id in MOUSE_IDS_DUAL:
        print(f"Processing subject: {subject_id}")
        print(f"--" * 50)
        save_dir = output_path / subject_id
        save_dir.mkdir(parents=True, exist_ok=True)

        session         = nap.load_file(input_path / subject_id / "session.npz")
        sleep           = nap.load_file(input_path / subject_id / "sleep.npz")
        hd_units        = nap.load_file(input_path / subject_id / "hd_units.npz")
        head_direction  = nap.load_file(input_path / subject_id / "head_direction.npz")

        if len(hd_units) == 0:
            print(f"No HD units in {subject_id}, skipping...")
            continue

        dt = 1/head_direction.rate
        train_ep = head_direction.time_support
        train_ep = nap.IntervalSet(train_ep.start - dt / 2, train_ep.end + dt / 2) # binned spikes are centered on the middle, so we shift left to align
        train_data = hd_units.count(bin_size=BIN_SIZE_S, ep=train_ep, time_units="s").astype(np.bool_)
        classifier = fit_classifier(head_direction.values, train_data.values, movement_var=2.0, state_prob=0.99)
        
        predict_ep = sleep[sleep['state'] == 'nrem'].intersect(session[session['state'] == 'homecage']) 
        for i, epoch in enumerate(predict_ep):
            if task_id is not None and total_tasks is not None:
                if i % total_tasks != task_id:
                    continue
            fname = save_dir / f"{subject_id}_{i}.npz"
            if fname.is_file():
                continue
            
            start, end = epoch.start.item(), epoch.end.item()
            print(f"Decoding epoch {i+1}/{len(predict_ep)}: {start:.1f}:{end:.1f}s ({(end - start):.1f} s)")

            spikes = hd_units.count(bin_size=BIN_SIZE_S, ep=epoch, time_units="s").astype(np.bool_)
            result = classifier.predict(spikes.values, spikes.t, state_names=STATE_NAMES)
            
            t = result['time'].to_numpy()
            d = result['acausal_posterior']
            states = d.sum(dim='position').to_numpy()
            position = d.sum(dim='state').idxmax(dim='position').to_numpy()
            combined = nap.TsdFrame(t=t, d=np.hstack((states, position[:, None])), columns=list(STATE_NAMES + ["position"]))
            combined.save(fname)
        
        print(f"Finished: {subject_id}")
        print(f"--" * 50)

if __name__ == "__main__":
    # Logging
    print("Python version")
    print(sys.version)
    print("Version info.")
    print(sys.version_info)

    if len(sys.argv) != 5 and len(sys.argv) != 3:
        print(f"Incorrect number of arguments: {len(sys.argv)-1}.")
        sys.exit(1)

    if len(sys.argv) == 5:
        main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        main(sys.argv[1], sys.argv[2])
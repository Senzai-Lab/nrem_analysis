from nrem_analysis.constant import PROCESSED_DIR, INTERIM_DIR, MOUSE_IDS_DUAL
import pynapple as nap
import numpy as np

if __name__ == "__main__":
    for mv in ["movement_var2", "movement_var20"]:
        for mouse_id in MOUSE_IDS_DUAL:
            data_dir = INTERIM_DIR  / mv / mouse_id
            print(data_dir)

            if not data_dir.exists():
                continue

            sleep = nap.load_file(PROCESSED_DIR / "dual" / mouse_id / "sleep.npz")
            turn_units = nap.load_file(PROCESSED_DIR / "dual" / mouse_id / "turn_units.npz")
            head_direction = nap.load_file(PROCESSED_DIR / "dual" / mouse_id / "head_direction.npz")
            session = nap.load_file(PROCESSED_DIR / "dual" / mouse_id / "session.npz")
            nrem = sleep[sleep['state'] == 'nrem']
            order = np.argsort(turn_units['turn_index'].values)

            virtual_hds = []
            # states = []
            for i, ep in enumerate(nrem.intersect(session[session['state'] == 'homecage'])):
                decoded = nap.load_file(data_dir / f"{mouse_id}_{i}.npz")
                # states_i = nap.TsdFrame(t=decoded.t, d=decoded.values[:, :3], columns=STATE_NAMES)
                vh = nap.Tsd(t=decoded.t, d=decoded.values[:, 3])
                # states.append(states_i)
                virtual_hds.append(vh)

            # states = np.concatenate(states)
            for angles, state in zip([np.concatenate(virtual_hds), head_direction], ["virtual_hd", "head_direction"]):
                angles = np.rad2deg(angles)
                angles_cart = np.column_stack([np.sin(angles), np.cos(angles)])
                print(f"Computing STA for {state}...")   
                print(f"nrem epohs: {len(nrem)}")
                sta = nap.compute_event_triggered_average(angles_cart, turn_units, binsize=1, window=(-500, 500), time_unit='ms')
                sta = nap.TsdFrame(t=sta.t, d=np.rad2deg(np.arctan2(sta[:, :, 0].d, sta[:, :, 1].d)))
                sta.save(data_dir / f"sta_{state}.npz")
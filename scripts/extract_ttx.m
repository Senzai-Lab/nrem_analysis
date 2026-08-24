% constants
mouse_ids = ["83b", "85b", "116b", "119b"];

data_dir = "D:\common_datasets\ucsf\";
raw_data_dir = data_dir + "raw\";
processed_data_dir = data_dir + "processed\";
input_path = raw_data_dir + "ttx\";
output_path = raw_data_dir + "ttx\";

metadata_fn = "MetaAnalysis_20260608-metaUnitFeature";
meta = load(raw_data_dir + metadata_fn).meta;

%% Sleep states
for i = 1:numel(mouse_ids)
    filename = "YutaTest" + mouse_ids(i) + ".SleepState.states.mat";
    states = load(input_path + mouse_ids(i) + "\"+ filename).SleepState.ints;
    
    fprintf("Loading: %s\n", filename);
    
    wake = states.WAKEstate;
    nrem = states.NREMstate;
    rem = states.REMstate;
    
    fprintf("Wake: %d, NREM: %d, REM: %d epochs\n", ...
        size(wake, 1), size(nrem, 1), size(rem, 1));
    
    save(output_path + mouse_ids(i) + "\" + "wake.mat", "wake");
    save(output_path + mouse_ids(i) + "\" + "nrem.mat", 'nrem');
    save(output_path + mouse_ids(i) + "\" + "rem.mat", 'rem');
end

%% head direction cells from ADn shanks
for i = 1:numel(mouse_ids)
    mouse_dir = input_path + mouse_ids(i) + "\";
    load(mouse_dir + "YutaTest" + mouse_ids(i) + "_BayesianDecoding_training.mat");
    fprintf("Mouse ID: %s (%d), HD Cells: %d\\%d\n", mouse_ids(i), i, sum(unit_idx), numel(unit_idx));
    hd_uid = unit_idx;
    save(output_path + mouse_ids(i) + "\" + "hd_uid.mat", "hd_uid");
end
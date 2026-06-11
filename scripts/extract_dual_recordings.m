% 1 : 99b
% 2 : 100b
% 3 : 102b
% 4 : 103c
% 5 : 106b
% 6 : 107b 
% 7 : 110b
% 8 : 111b

% constants
filenames = ["99b", "100b", "102b", "103c", "106b", "107b", "110b", "111b"];
shank_sc = [3, 3, 3, 3, 3, 3, 5, 3]; % shank number for SC
shanks_hd = [2, 2, 2, 2, 2, 2, 4, 2]; % number of shanks for ADn

data_dir = "D:\common_datasets\ucsf\";
raw_data_dir = data_dir + "raw\";
processed_data_dir = data_dir + "processed\";
input_path = raw_data_dir + "dual\";
output_path = raw_data_dir + "dual\";

metadata_fn = "MetaAnalysis_20260608-metaUnitFeature";
meta = load(raw_data_dir + metadata_fn).meta;

% Sleep states
for i = 1:numel(filenames)
    filename = "YutaTest" + filenames(i) + ".SleepState.states.mat";
    states = load(input_path + filenames(i) + "\"+ filename).SleepState.ints;
    
    fprintf("Loading: %s\n", filename);
    
    wake = states.WAKEstate;
    nrem = states.NREMstate;
    rem = states.REMstate;
    
    fprintf("Wake: %d, NREM: %d, REM: %d epochs\n", ...
        size(wake, 1), size(nrem, 1), size(rem, 1));
    
    save(output_path + filenames(i) + "\" + "wake.mat", "wake");
    save(output_path + filenames(i) + "\" + "nrem.mat", 'nrem');
    save(output_path + filenames(i) + "\" + "rem.mat", 'rem');
end

% Turn modulated SC cell ids from SC shank
for i = 1:numel(shank_sc)
    mouse_id = meta.mouseID == i; % 99b to 111b
    shank_id = meta.shankID == shank_sc(i);
    mask = meta.SC_TurnModCell & mouse_id & shank_id;
    
    turn_uid = meta.unitIDshank(mask); % Cluster IDs for Turn modulated cell
    turn_index = meta.TurnIdx(mask); % Turn modulation strength, >0 is CW, <0 is CCW
    
    fprintf("Mouse ID: %s (%d), Turn Modulated Cell Count: %d ", filenames(i), i, numel(turn_uid));
    fprintf("CW Turn cells: %d, CCW Turn cells: %d\n", sum(turn_index>0), sum(turn_index<0));
    
    save(output_path + filenames(i) + "\" + "turn_uid.mat", "turn_uid");
    save(output_path + filenames(i) + "\" + "turn_index.mat", "turn_index");
end

% update metaUnitFeature with HD cell ids for 106b
% load("D:\common_datasets\ucsf\raw\dual\106b\YutaTest106b_BayesianDecoding_training_OpenField2.mat");
% mouse_id = meta.mouseID == 5;
% shank_id = meta.shankID == 1 | meta.shankID == 2;
% meta.HDCidx(mouse_id & shank_id) = unit_idx;

% head direction cells from ADn shanks
for i = 1:numel(shank_sc)
    mouse_id = meta.mouseID == i;
    if i == 7
        shank_id = (meta.shankID == 1) | (meta.shankID == 2) | (meta.shankID == 3) | (meta.shankID == 4);
    else
        shank_id = (meta.shankID == 1) | (meta.shankID == 2);
    end
    mask = meta.HDCidx & mouse_id & shank_id;
    hd_uid = meta.unitIDshank(mask);
    fprintf("Mouse ID: %s (%d), HD Cell Count: %d\n", filenames(i), i, numel(hd_uid));

    save(output_path + filenames(i) + "\" + "hd_uid.mat", "hd_uid");
end

% load openfield and homecage periods
load(input_path + "MetaAnalysis_20240530-HomeCage-OpenField-Periods.mat");
save(output_path + "openfield_periods", "OpenFieldPeriods");
save(output_path + "homecage_periods", "HomeCagePeriods");
%% Fold 1: Data Preparation, Network Creation, and Training

%-------------------------------
% (1) Data Preparation
%-------------------------------

% Set data folder path and list all CDLChannelEst files
dataFolder = fullfile(pwd, 'Data', 'CDLData');
allFiles   = dir(fullfile(dataFolder, 'CDLChannelEst_*.mat'));
fileNames  = {allFiles.name};

% Helper to check if a string contains any of a list of substrings
isInList = @(str, list) any(contains(str, list));

% Define Fold 1 train/validation channel splits
trainCDL     = ["CDL-A", "CDL-B"];
trainDS      = ["DS28ns", "DS35ns", "DS42ns", "DS100ns"];
trainDoppler = ["D10Hz", "D97Hz"];

valCDL     = ["CDL-C"];
valDS      = ["DS125ns"];
valDoppler = ["D389Hz"];

% Filter filenames for training
trainFiles = fileNames( ...
    cellfun(@(f) isInList(f, trainCDL) & isInList(f, trainDS) & isInList(f, trainDoppler), ...
            fileNames) ...
);

% Filter filenames for validation
valFiles = fileNames( ...
    cellfun(@(f) isInList(f, valCDL) & isInList(f, valDS) & isInList(f, valDoppler), ...
            fileNames) ...
);

% Build full file paths
trainPaths = fullfile(dataFolder, trainFiles);
valPaths   = fullfile(dataFolder, valFiles);

% Create signalDatastores (ReadSize=1 so each read returns one file’s data)
trainSds = signalDatastore(trainPaths, 'ReadSize', 1);
valSds   = signalDatastore(valPaths,   'ReadSize', 1);

% Read and concatenate training data into [58×32×2×Ntrain]
HTrainRealCell = readall(trainSds);
HTrainReal     = cat(1, HTrainRealCell{:});          % stacks along dim1
HTrainReal     = permute(HTrainReal, [2, 3, 4, 1]);   % → [58×32×2×Ntrain]

% Read and concatenate validation data into [58×32×2×Nval]
HValRealCell = readall(valSds);
HValReal     = cat(1, HValRealCell{:});
HValReal     = permute(HValReal, [2, 3, 4, 1]);       % → [58×32×2×Nval]

numTrainSamples = size(HTrainReal, 4);
numValSamples   = size(HValReal,   4);

fprintf("numTrainSamples = %d\n", numTrainSamples);
fprintf("numValSamples   = %d\n", numValSamples);

% Save file lists for auditing
writelines(trainFiles', 'train_files_fold1.txt');
writelines(valFiles',   'val_files_fold1.txt');

% Display a few sample filenames
disp("Sample Training Files:");
disp(trainFiles(1:min(5, numel(trainFiles)))');

disp("Sample Validation Files:");
disp(valFiles(1:min(5, numel(valFiles)))');

% Crop CSI data from [58×32×2] to [32×32×2] so it matches the network’s hard-coded reshape
HTrainReal = HTrainReal(1:32, 1:32, :, :);   % → [32×32×2×numTrainSamples]
HValReal   = HValReal(  1:32, 1:32, :, :);   % → [32×32×2×numValSamples]

% Verify sizes after cropping
szAfterTrain = size(HTrainReal);
szAfterVal   = size(HValReal);
fprintf("After cropping (train): %s\n", mat2str(szAfterTrain));
fprintf("After cropping (val):   %s\n", mat2str(szAfterVal));

%-------------------------------
% (2) Create and Initialize Network
%-------------------------------

% Define network parameters
inputSize     = szAfterTrain(1:3);    % [32 32 2]
encodedSize   = 64;
embeddingSize = 16;
windowSize    = 8;
numPatches    = inputSize(1)*inputSize(2)/(2*2);  % = 256

% Build the CSIFormer network (returns a dlnetwork or LayerGraph)
CSIFormerNet = helperCSIFormerCreateNetwork( ...
    inputSize, encodedSize, embeddingSize, numPatches, windowSize );

% Convert to dlnetwork if needed, then initialize parameters
if isa(CSIFormerNet, 'dlnetwork')
    dlnet = CSIFormerNet;
else
    dlnet = dlnetwork(CSIFormerNet);
end
dlnet = initialize(dlnet);

% Quick forward-pass sanity check
dummyIn = rand(inputSize, 'single');    % random [32×32×2]
dlX     = dlarray(dummyIn, "SSCB");     % Spatial-Spatial-Channel-Batch

try
    dlY = predict(dlnet, dlX);
    disp("→ Forward pass OK: No reshape errors");
catch ME
    error("! Forward-pass error: %s", ME.message);
end

%-------------------------------
% (3) Training Options
%-------------------------------

trainNow    = true;
saveNetwork = true;

% Custom NMSE(dB) metric for display (not used as loss)
trainingMetric = @(YPred, YTrue) helperNMSELossdB(YPred, YTrue);

epochs        = 10;
batchSize     = 500;
initLearnRate = 8e-4;

options = trainingOptions("adam", ...
    InitialLearnRate    = initLearnRate, ...
    LearnRateSchedule   = "piecewise", ...
    LearnRateDropPeriod = 30, ...    % Warm-up for first 30 epochs
    LearnRateDropFactor = 0.1, ...
    MaxEpochs           = epochs, ...
    MiniBatchSize       = batchSize, ...
    Shuffle             = "every-epoch", ...
    ValidationData      = {HValReal, HValReal}, ...
    ValidationFrequency = 200, ...
    OutputNetwork       = "best-validation-loss", ...
    Metrics             = trainingMetric, ...
    Plots               = "training-progress", ...
    Verbose             = false, ...
    VerboseFrequency    = 100);

%-------------------------------
% (4) Train or Load Network
%-------------------------------

if trainNow
    % Train the autoencoder: input = HTrainReal, target = HTrainReal
    [trainedNet, trainInfo] = trainnet( ...
        HTrainReal, HTrainReal, dlnet, @(X, T) mse(X, T), options );

    if saveNetwork
        timestamp = string(datetime("now", "Format", "dd_MM_HH_mm"));
        save("CSIFormerTrainedNetwork_" + timestamp, 'trainedNet');
    end
else
    % If not training now, load a previously saved network
    trainedNetName = "trainedNetEnc" + num2str(compressionFactor);
    load("CSIFormerTrainedNets.mat", trainedNetName);
    trainedNet = eval(trainedNetName);
end

% At this point, 'trainedNet' is your trained dlnetwork for Fold 1.
% You can proceed to evaluate NMSE on validation or a held-out test set.

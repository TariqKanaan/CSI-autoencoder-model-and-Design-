%% trainFold1_fullCSI_padded.m
% Fold 1: Load, pad to unify to [72×32×2], build network, and train.

clear; clc;

%% (1) Data Preparation (Fold 1)

% 1.1 Set data folder path and gather all “CDLChannelEst_*.mat” files
dataFolder = fullfile(pwd, 'Data', 'CDLData');
allFiles   = dir(fullfile(dataFolder, 'CDLChannelEst_*.mat'));
fileNames  = {allFiles.name};

% 1.2 Helper to check if a filename contains any substring from a list
isInList = @(str, list) any(contains(str, list));

% 1.3 Fold 1 channel splits
trainCDL     = ["CDL-A", "CDL-B"];
trainDS      = ["DS28ns", "DS35ns", "DS42ns", "DS100ns"];
trainDoppler = ["D10Hz", "D97Hz"];

valCDL       = ["CDL-C"];
valDS        = ["DS125ns"];
valDoppler   = ["D389Hz"];

% 1.4 Filter filenames for training
trainFiles = fileNames( ...
    cellfun(@(f) isInList(f, trainCDL)    & ...
                isInList(f, trainDS)     & ...
                isInList(f, trainDoppler), fileNames) ...
);

% 1.5 Filter filenames for validation
valFiles = fileNames( ...
    cellfun(@(f) isInList(f, valCDL)     & ...
                isInList(f, valDS)       & ...
                isInList(f, valDoppler), fileNames) ...
);

% 1.6 Build full file paths and create signalDatastores (ReadSize=1)
trainPaths = fullfile(dataFolder, trainFiles);
valPaths   = fullfile(dataFolder, valFiles);

trainSds = signalDatastore(trainPaths, 'ReadSize', 1);
valSds   = signalDatastore(valPaths,   'ReadSize', 1);

% 1.7 Read & concatenate training data → [58×32×2×Ntrain]
HTrainRealCell = readall(trainSds);
HTrainReal     = cat(1, HTrainRealCell{:});           % stacks along dim1
HTrainReal     = permute(HTrainReal, [2, 3, 4, 1]);    % → [58×32×2×Ntrain]

% 1.8 Read & concatenate validation data → [72×32×2×Nval] (for CDL-C)
HValRealCell = readall(valSds);
HValReal     = cat(1, HValRealCell{:});
HValReal     = permute(HValReal, [2, 3, 4, 1]);        % likely [72×32×2×Nval]

numTrainSamples = size(HTrainReal, 4);
numValSamples   = size(HValReal,   4);

fprintf("numTrainSamples = %d\n", numTrainSamples);
fprintf("numValSamples   = %d\n", numValSamples);

% 1.9 Display a few sample filenames for sanity check
disp("Sample Training Files:");
disp(trainFiles(1:min(5, numel(trainFiles)))');

disp("Sample Validation Files:");
disp(valFiles(1:min(5, numel(valFiles)))');

% 1.10 Save file lists for audit
writelines(trainFiles', 'train_files_fold1.txt');
writelines(valFiles',   'val_files_fold1.txt');

%-------------------------------
% (2) Zero-Pad to unify heights
%-------------------------------

% 2.1 Determine heights of each dataset
hTrain = size(HTrainReal, 1);  % e.g., 58
hVal   = size(HValReal,   1);   % e.g., 72 (for CDL-C)

% 2.2 The new “max height” to pad to
Hmax = max(hTrain, hVal);      % = 72

% 2.3 If training samples are shorter, pad zeros downwards
if hTrain < Hmax
    padSize = Hmax - hTrain;   % e.g., 72 - 58 = 14
    padTensor = zeros(padSize, size(HTrainReal,2), size(HTrainReal,3), size(HTrainReal,4), 'like', HTrainReal);
    HTrainReal = cat(1, HTrainReal, padTensor);  % → [72×32×2×Ntrain]
end

% 2.4 If validation samples are shorter, pad zeros downward
if hVal < Hmax
    padSize = Hmax - hVal;
    padTensor = zeros(padSize, size(HValReal,2), size(HValReal,3), size(HValReal,4), 'like', HValReal);
    HValReal = cat(1, HValReal, padTensor);      % → [72×32×2×Nval]
end

% 2.5 Verify both are now [Hmax×32×2×N]
szTrainPadded = size(HTrainReal);  % should be [72 32 2 20032]
szValPadded   = size(HValReal);    % should be [72 32 2 10016]
fprintf("After padding (train): %s\n", mat2str(szTrainPadded));
fprintf("After padding (val):   %s\n", mat2str(szValPadded));

%-------------------------------
% (3) Build and initialize CSIFormer network
%-------------------------------

inputSize     = [Hmax, size(HTrainReal,2), size(HTrainReal,3)];  % [72 32 2]
encodedSize   = 64;
embeddingSize = 16;
windowSize    = 8;

% 3.1 Compute Hpatch, Wpatch, numPatches for [72×32] → patch splitting
H = inputSize(1);  % 72
W = inputSize(2);  % 32
Hpatch = floor((H - 2)/2) + 1;   % floor((72-2)/2)+1 = 36
Wpatch = floor((W - 2)/2) + 1;   % floor((32-2)/2)+1 = 16
numPatches = Hpatch * Wpatch;    % = 36 * 16 = 576

% 3.2 Create the network using the custom helper that expects dynamic Hpatch/Wpatch
CSIFormerNet = helperCSIFormerCreateNetwork_Custom( ...
    inputSize, encodedSize, embeddingSize, numPatches, windowSize );

% 3.3 Convert to dlnetwork if needed, then initialize
if isa(CSIFormerNet, 'dlnetwork')
    dlnet = CSIFormerNet;
else
    dlnet = dlnetwork(CSIFormerNet);
end
dlnet = initialize(dlnet);

% 3.4 Quick forward-pass check on [72×32×2]
dummyIn = rand(inputSize, 'single');   % random [72×32×2]
dlX     = dlarray(dummyIn, "SSCB");    % “SSCB”: Spatial-Spatial-Channel-Batch
try
    dlY = predict(dlnet, dlX);
    disp("→ Forward pass OK on [72×32×2] sample.");
catch ME
    error("! Forward-pass error: %s", ME.message);
end

%-------------------------------
% (4) Training Options
%-------------------------------

trainNow    = true;
saveNetwork = true;

trainingMetric = @(YPred, YTrue) helperNMSELossdB(YPred, YTrue);

epochs        = 10;
batchSize     = 500;
initLearnRate = 8e-4;

options = trainingOptions("adam", ...
    InitialLearnRate    = initLearnRate, ...
    LearnRateSchedule   = "piecewise", ...
    LearnRateDropPeriod = 30, ...
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
% (5) Train (or load) the network
%-------------------------------

if trainNow
    [trainedNet, trainInfo] = trainnet( ...
        HTrainReal, HTrainReal, dlnet, @(X, T) mse(X, T), options );

    if saveNetwork
        timestamp = string(datetime("now", "Format", "dd_MM_HH_mm"));
        save("CSIFormerTrainedNetwork_Fold1_padded_" + timestamp, 'trainedNet');
        disp("→ Network saved as: CSIFormerTrainedNetwork_Fold1_padded_" + timestamp + ".mat");
    end
else
    % To load a previously saved network, comment out trainNow=true and do:
    trainedNetName = "trainedNetEnc" + num2str(encodedSize);
    load("CSIFormerTrainedNetworks.mat", trainedNetName);
    trainedNet = eval(trainedNetName);
end

%-------------------------------
% (6) Evaluate on Validation (NMSE)
%-------------------------------

% Forward-pass on all HValReal
YValPred = predict(trainedNet, dlarray(HValReal, "SSCB"));
YValPred = stripdims(extractdata(YValPred));  % → [72×32×2×10016]

orig    = HValReal;
nmseVec = squeeze( sum(sum(sum((orig - YValPred).^2,1),2),3) ...
                 ./ sum(sum(sum(orig.^2,1),2),3) );
nmse_dB = 10 * log10(nmseVec);

fprintf("Validation NMSE (Fold 1, padded), mean (dB): %.2f dB\n", mean(nmse_dB));
fprintf("Validation NMSE (Fold 1, padded), std  (dB): %.2f dB\n", std(nmse_dB));

% (Optional) You can load a separate “Test” set (e.g. CDL-D/E) and compute its NMSE similarly.


%% trainFold1_fullCSI_fixed.m
% Fold 1: Full‐CSI Autoencoder Training with Dynamic Cropping to Common Size

%% (1) Data‐Preparation: Reload raw cells from file lists

% (1.1) Read the saved file‐lists (from prepare1_fold1.m)
trainFiles = readlines("train_files_fold1.txt");
valFiles   = readlines("val_files_fold1.txt");

dataFolder = fullfile(pwd, 'Data', 'CDLData');

% (1.2) Build full paths again
trainPaths = fullfile(dataFolder, cellstr(trainFiles));
valPaths   = fullfile(dataFolder, cellstr(valFiles));

% (1.3) Create datastores that return one .mat per Read
trainSds = signalDatastore(trainPaths, 'ReadSize', 1);
valSds   = signalDatastore(valPaths,   'ReadSize', 1);

% (1.4) Read all data into cell arrays:
% Each element is a [rows×32×2] matrix (rows may vary: e.g. 58 for train, 72 for some val)
HTrainRawCell = readall(trainSds);
HValRawCell   = readall(valSds);

% (1.5) For sanity, inspect sizes of a few samples
% disp(size(HTrainRawCell{1}));  % e.g. [58 32 2]
% disp(size(HValRawCell{1}));    % maybe [72 32 2], etc.

numTrain = numel(HTrainRawCell);
numVal   = numel(HValRawCell);

fprintf("Reloaded %d training samples, %d validation samples.\n", numTrain, numVal);

%% (2) Determine common row‐count across all samples

% (2.1) Extract row‐counts for train and val
trainRows = cellfun(@(x) size(x,1), HTrainRawCell);  % e.g. all 58
valRows   = cellfun(@(x) size(x,1), HValRawCell);    % some may be 72

commonRows = min([trainRows; valRows]);  
fprintf("Cropping all samples to %d rows (commonRows).\n", commonRows);

%% (3) Crop each cell and build 4‐D arrays [commonRows×32×2×N]

% (3.1) Build HTrainReal
HTrainRealCropped = cell(numTrain, 1);
for i = 1:numTrain
    sample = HTrainRawCell{i};      % e.g. [58×32×2]
    % Crop the first dimension to commonRows:
    HTrainRealCropped{i} = sample(1:commonRows, :, :);  % → [commonRows×32×2]
end
% Concatenate into [commonRows×32×2×numTrain]:
HTrainReal = cat(4, HTrainRealCropped{:});
clear HTrainRealCropped;

% (3.2) Build HValReal
HValRealCropped = cell(numVal, 1);
for i = 1:numVal
    sample = HValRawCell{i};      % e.g. [72×32×2]
    HValRealCropped{i} = sample(1:commonRows, :, :);  % → [commonRows×32×2]
end
HValReal = cat(4, HValRealCropped{:});
clear HValRealCropped;

% (3.3) Verify final sizes
szTrain = size(HTrainReal);  % → [commonRows, 32, 2, numTrain]
szVal   = size(HValReal);    % → [commonRows, 32, 2, numVal]
fprintf("After cropping:\n");
fprintf("  HTrainReal size = [%d %d %d %d]\n", szTrain);
fprintf("  HValReal   size = [%d %d %d %d]\n", szVal);

%% (4) Build and initialize the CSIFormer network on [commonRows×32×2]

inputSize     = szTrain(1:3);   % [commonRows, 32, 2]
encodedSize   = 64;
embeddingSize = 16;
windowSize    = 8;

% Compute Hpatch and Wpatch:
H = inputSize(1);  % commonRows (should be 58)
W = inputSize(2);  % 32
Hpatch = floor((H - 2)/2) + 1;  % = 29 when H=58
Wpatch = floor((W - 2)/2) + 1;  % = 16 when W=32

numPatches = Hpatch * Wpatch;   % e.g. 29×16 = 464

% Create the network using the custom helper
CSIFormerNet = helperCSIFormerCreateNetwork_Custom( ...
    inputSize, encodedSize, embeddingSize, numPatches, windowSize );

% Wrap into dlnetwork if needed, then initialize all parameters
if isa(CSIFormerNet, 'dlnetwork')
    dlnet = CSIFormerNet;
else
    dlnet = dlnetwork(CSIFormerNet);
end
dlnet = initialize(dlnet);

% (4.1) Sanity‐check forward pass on a random [commonRows×32×2] sample
dummyIn = rand(inputSize, 'single');
dlX     = dlarray(dummyIn, "SSCB");  % Spatial-Spatial-Channel-Batch
try
    dlY = predict(dlnet, dlX);
    disp("→ Forward pass OK on [commonRows×32×2] sample.");
catch ME
    error("! Forward‐pass error: %s", ME.message);
end

%% (5) Training Options

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

%% (6) Train (or Load) the Network

if trainNow
    [trainedNet, trainInfo] = trainnet( ...
        HTrainReal, HTrainReal, dlnet, @(X, T) mse(X, T), options );

    if saveNetwork
        timestamp = string(datetime("now", "Format", "dd_MM_HH_mm"));
        save("CSIFormerTrainedNetwork_Fold1_" + timestamp, 'trainedNet');
        disp("→ Network saved: CSIFormerTrainedNetwork_Fold1_" + timestamp + ".mat");
    end
else
    trainedNetName = "trainedNetEnc" + num2str(encodedSize);
    load("CSIFormerTrainedNetworks.mat", trainedNetName);
    trainedNet = eval(trainedNetName);
end

%% (7) Evaluate on Validation (Compute NMSE)

YValPred = predict(trainedNet, dlarray(HValReal, "SSCB"));
YValPred = stripdims(extractdata(YValPred));  % [commonRows×32×2×numVal]

orig    = HValReal;
nmseVec = squeeze( sum(sum(sum((orig - YValPred).^2, 1), 2), 3) ...
                 ./ sum(sum(sum(orig.^2, 1), 2), 3) );
nmse_dB = 10 * log10(nmseVec);

fprintf("Validation NMSE (Fold 1), mean (dB): %.2f dB\n", mean(nmse_dB));
fprintf("Validation NMSE (Fold 1), std  (dB): %.2f dB\n", std(nmse_dB));

% (If you have a separate test‐set for Fold 1—e.g., CDL-D/E files—load and evaluate similarly.)

%% Configuration
clear; clc;
%addpath 'C:\Users\TARIQ\Documents\MATLAB\Examples\R2024a\deeplearning_shared\CSICompressionAutoencoderExample'
% Flags
doVisualizeCSI = true;
doGenerateData = false ;
doTrainNet = false;

%% Data and Channel Parameters
params.nSizeGrid         = 48;
params.subcarrierSpacing = 30; % kHz
params.NumSubcarriers    = params.nSizeGrid * 12;
params.rxAntennaSize     = [2 1 1 1 1];
params.txAntennaSize     = [4 4 2 1 1]; % larger array for training
params.maxDoppler        = 2;          % Hz
params.rmsDelaySpread    = 100e-9;     % seconds
params.delayProfile      = 'CDL-B';
params.truncationFactor  = 19;

% Dataset sizes
params.numTrainSamples = 10000;
params.numValSamples   = 3000;
params.numTestSamples  = 2000;

Tdelay = 1 / (params.NumSubcarriers * params.subcarrierSpacing * 1e3);
rmsTauSamples = params.rmsDelaySpread / Tdelay;
params.maxDelay = round((params.rmsDelaySpread / Tdelay) * params.truncationFactor / 2) * 2;

carrier = nrCarrierConfig;
carrier.NSizeGrid = params.nSizeGrid;
carrier.SubcarrierSpacing = params.subcarrierSpacing;
waveInfo = nrOFDMInfo(carrier);
samplesPerSlot = sum(waveInfo.SymbolLengths(1:waveInfo.SymbolsPerSlot));

channel = nrCDLChannel;
channel.DelayProfile = params.delayProfile;
channel.DelaySpread = params.rmsDelaySpread;
channel.MaximumDopplerShift = params.maxDoppler;
channel.RandomStream = "Global stream";
channel.TransmitAntennaArray.Size = params.txAntennaSize;
channel.ReceiveAntennaArray.Size = params.rxAntennaSize;
channel.ChannelFiltering = false;
channel.NumTimeSamples = samplesPerSlot;
channel.SampleRate = waveInfo.SampleRate;

channelInfo = info(channel);

opt.NumSubcarriers  = params.NumSubcarriers;
opt.NumSymbols      = carrier.SymbolsPerSlot;
opt.NumTxAntennas   = channelInfo.NumInputSignals;
opt.NumRxAntennas   = channelInfo.NumOutputSignals;
opt.maxDelay        = params.maxDelay;
opt.targetStd       = 0.0212;
opt.targetMean      = 0.5;

%% Generate Data if needed
if doGenerateData
    dataSet.Name     = ["CDLChannelEst_train", "CDLChannelEst_val", "CDLChannelEst_test"];
    dataSet.Size     = [params.numTrainSamples, params.numValSamples, params.numTestSamples];
    dataSet.location = fullfile(pwd, "Data", "processedData");

    for i = 1:numel(dataSet.Name)
        dataSetName = dataSet.Name(i);
        dataSetSize = dataSet.Size(i);
        dataDir     = dataSet.location;

        dataExists = dataSetExists(dataDir, dataSetName, channel, carrier);
        if ~dataExists
            disp("Starting " + dataSetName + " data generation");
            tic;
            generateData(dataDir, dataSetName, dataSetSize, carrier, channel, opt);
            disp(string(seconds(toc)) + " - Finished " + dataSetName);
        else
            disp(dataSetName + " already exists. Skipping.");
        end
    end
end

%% Load Dataset for Training
if doTrainNet
    % Read training data
    trainSds = signalDatastore(fullfile(pwd, "Data", "processedData", "CDLChannelEst_train*"));
    HTrainRealCell = readall(trainSds);
    HTrainReal = cat(1, HTrainRealCell{:});
    HTrainReal = permute(HTrainReal, [2 3 4 1]);  % [H, W, C, N]

    % Read validation data
    valSds = signalDatastore(fullfile(pwd, "Data", "processedData", "CDLChannelEst_val*"));
    HValRealCell = readall(valSds);
    HValReal = cat(1, HValRealCell{:});
    HValReal = permute(HValReal, [2 3 4 1]);

    % Input shape and Transformer config
    inputSize = size(HTrainReal(:,:,:,1)); % [H W C]
    encodedSize = 64;
    embeddingSize = 16;
    windowSize = 8;
    numPatches = inputSize(1)*inputSize(2)/(2*2);

    CSIFormerNet = helperCSIFormerCreateNetwork(inputSize, encodedSize, ...
                                                embeddingSize, numPatches, windowSize);

    % Training settings
    epochs = 10;
    batchSize = 500;
    initLearnRate = 8e-4;
    trainingMetric = @(x,t) helperNMSELossdB(x,t);

    options = trainingOptions("adam", ...
        InitialLearnRate=initLearnRate, ...
        LearnRateSchedule="piecewise", ...
        LearnRateDropPeriod=30, ...
        LearnRateDropFactor=0.1, ...
        MaxEpochs=epochs, ...
        MiniBatchSize=batchSize, ...
        Shuffle="every-epoch", ...
        ValidationData={HValReal, HValReal}, ...
        ValidationFrequency=200, ...
        OutputNetwork="best-validation-loss", ...
        Metrics=trainingMetric, ...
        Plots="training-progress", ...
        Verbose=false, ...
        VerboseFrequency=100);

    % Train the network
    [trainedNet, trainInfo] = trainnet(HTrainReal, HTrainReal, CSIFormerNet, @(X,T) mse(X,T), options);

    % Save the trained model
    save("CSIFormerTrainedNetwork_" + string(datetime("now","Format","dd_MM_HH_mm")), "trainedNet")
end

%% Configuration
clear; clc;

% Flags
doGenerateData = true;

%% Data and Channel Parameters
params.nSizeGrid         = 48;      % PRBs
params.subcarrierSpacing = 30;      % kHz
params.NumSubcarriers    = params.nSizeGrid * 12;
params.rxAntennaSize     = [2 1 1 1 1];     % ULA with 2 elements
params.txAntennaSize     = [4 4 2 1 1];     % UPA with 32 elements
params.maxDoppler        = 5;              % Hz, higher for mobility
params.rmsDelaySpread    = 100e-9;         % 100 ns
params.truncationFactor  = 19;

% Dataset sizes
params.numTrainSamples = 10000;
params.numValSamples   = 3000;
params.numTestSamples  = 2000;

% Time delay setup
Tdelay = 1 / (params.NumSubcarriers * params.subcarrierSpacing * 1e3);
params.maxDelay = round((params.rmsDelaySpread / Tdelay) * params.truncationFactor / 2) * 2;

%% Carrier Configuration
carrier = nrCarrierConfig;
carrier.NSizeGrid = params.nSizeGrid;
carrier.SubcarrierSpacing = params.subcarrierSpacing;

waveInfo = nrOFDMInfo(carrier);
samplesPerSlot = sum(waveInfo.SymbolLengths(1:waveInfo.SymbolsPerSlot));

%% Generate CDL-A to CDL-E Data
if doGenerateData
    dataSet.Name     = ["CDL-A", "CDL-B", "CDL-C", "CDL-D", "CDL-E"];
    dataSet.Size     = [params.numTrainSamples, params.numTrainSamples, ...
                        params.numTrainSamples, params.numTrainSamples, ...
                        params.numTrainSamples];
    dataSet.location = fullfile(pwd, "Data", "CDLData");

    for i = 1:numel(dataSet.Name)
        delayProfile = dataSet.Name(i);

        % Configure CDL Channel
        channel = nrCDLChannel;
        channel.DelayProfile = delayProfile;
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

        % Check if data exists
        dataDir = dataSet.location;
        dataSetName = "CDLChannelEst_" + delayProfile;
        dataSetSize = dataSet.Size(i);

        dataExists = dataSetExists(dataDir, dataSetName, channel, carrier);
        if ~dataExists
            disp("Generating data for " + delayProfile);
            tic;
            generateData(dataDir, dataSetName, dataSetSize, carrier, channel, opt);
            disp("Finished " + delayProfile + " in " + string(seconds(toc)));
        else
            disp(dataSetName + " already exists. Skipping.");
        end
    end
end

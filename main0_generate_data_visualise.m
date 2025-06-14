%% Main Configuration
clear; clc;

% Flags
doVisualizeCSI = true;   % Set true to visualize CSI
doGenerateData = false;  % Set true to generate dataset

%% Common Parameters
params.nSizeGrid         = 52;
params.subcarrierSpacing = 15; % kHz
params.NumSubcarriers    = params.nSizeGrid * 12;
params.rxAntennaSize     = [2 1 1 1 1];
params.txAntennaSize     = [2 2 2 1 1];
params.maxDoppler        = 5;          % Hz
params.rmsDelaySpread    = 300e-9;     % seconds
params.delayProfile      = 'CDL-C';
params.truncationFactor  = 10;         % Truncation factor in delay domain

% Dataset sizes
params.numTrainSamples = 10000;
params.numValSamples   = 3000;
params.numTestSamples  = 2000;

%% Derived Parameters
Tdelay = 1/(params.NumSubcarriers * params.subcarrierSpacing * 1e3);
rmsTauSamples = params.rmsDelaySpread / Tdelay;
maxTruncationFactor = floor(params.NumSubcarriers / rmsTauSamples);
params.maxDelay = round((params.rmsDelaySpread / Tdelay) * params.truncationFactor / 2) * 2;

%% Carrier and OFDM Info
carrier = nrCarrierConfig;
carrier.NSizeGrid = params.nSizeGrid;
carrier.SubcarrierSpacing = params.subcarrierSpacing;
waveInfo = nrOFDMInfo(carrier);
samplesPerSlot = sum(waveInfo.SymbolLengths(1:waveInfo.SymbolsPerSlot));

%% Channel Configuration
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

%% Visualization
if doVisualizeCSI
    [pathGains, sampleTimes] = channel();
    pathFilters = getPathFilters(channel);
    offset = 0;

    Hest = nrPerfectChannelEstimate(carrier, pathGains, pathFilters, offset, sampleTimes);
    reset(channel);

    [nSub, nSym, nRx, nTx] = size(Hest);
    helperPlotChannelResponse(Hest);

    % Mean across symbols
    Hmean = squeeze(mean(Hest, 2));
    Hmean = permute(Hmean, [1 3 2]); % [Subcarriers, Tx, Rx]
    Hdft2 = fft2(Hmean);

    midPoint = floor(nSub / 2);
    lowerEdge = midPoint - (nSub - params.maxDelay) / 2 + 1;
    upperEdge = midPoint + (nSub - params.maxDelay) / 2;
    Htemp = Hdft2([1:lowerEdge-1 upperEdge+1:end], :, :);

    Htrunc = ifft2(Htemp);
    HtruncReal = cat(3, real(Htrunc), imag(Htrunc));

    figure;
    subplot(1,2,1);
    imagesc(HtruncReal(:,:,1,1));
    xlabel("Tx Antennas"); ylabel("Compressed Subcarriers"); title("In-phase");
    subplot(1,2,2);
    imagesc(HtruncReal(:,:,2,1));
    xlabel("Tx Antennas"); ylabel("Compressed Subcarriers"); title("Quadrature");

    helperPlotCSIFeedbackPreprocessingSteps(...
        Hmean(:,:,1), Hdft2(:,:,1), Htemp(:,:,1), Htrunc(:,:,1), ...
        nSub, nTx, params.maxDelay, "Frequency-Spatial");
end

%% Dataset Generation
if doGenerateData
    opt.NumSubcarriers  = params.NumSubcarriers;
    opt.NumSymbols      = carrier.SymbolsPerSlot;
    opt.NumTxAntennas   = channelInfo.NumInputSignals;
    opt.NumRxAntennas   = channelInfo.NumOutputSignals;
    opt.maxDelay        = params.maxDelay;
    opt.targetStd       = 0.0212;
    opt.targetMean      = 0.5;

    dataSet.Name     = ["CDLChannelEst_train", "CDLChannelEst_val", "CDLChannelEst_test"];
    dataSet.Size     = [params.numTrainSamples, params.numValSamples, params.numTestSamples];
    dataSet.location = fullfile(pwd, "processedData");

    for i = 1:numel(dataSet.Name)
        dataSetName = dataSet.Name(i);
        dataSetSize = dataSet.Size(i);
        dataDir     = dataSet.location;

        dataExists = dataSetExists(dataDir, dataSetName, channel, carrier);
        if ~dataExists
            disp("Starting "+ dataSetName +" data generation");
            tic;
            generateData(dataDir, dataSetName, dataSetSize, carrier, channel, opt);
            t = seconds(toc); t.Format = "hh:mm:ss";
            disp(string(t) + " - Finished "+ dataSetName +" data generation");
        else
            disp("Data set exists. Skipping data generation.");
        end
    end
end

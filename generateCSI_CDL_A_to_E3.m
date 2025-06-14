%% Configuration
clear; clc;

% Flags
doGenerateData = true;

%% Constants
c = 3e8;                  
fc = 3.5e9;               
velocities_kmh = [3, 30, 120];
dopplers = (velocities_kmh / 3.6) / c * fc;
variationFactors = [0.8, 1.0, 1.2];

cdlProfiles = ["CDL-A", "CDL-B", "CDL-C", "CDL-D", "CDL-E"];
cdlLOSMap = containers.Map( ...
    {'CDL-A','CDL-B','CDL-C','CDL-D','CDL-E'}, ...
    [false, false, false, true, true] ...
);

baseDelaySpreads = containers.Map( ...
    {'CDL-A','CDL-B','CDL-C','CDL-D','CDL-E'}, ...
    [200e-9, 125e-9, 125e-9, 35e-9, 35e-9] ...
);

%% Data Parameters (Fix maxDelay for consistent size)
params.NumSubcarriers    = 1024;
params.nSizeGrid         = params.NumSubcarriers / 12;
params.subcarrierSpacing = 30;
params.rxAntennaSize     = [4 1 1 1 1];
params.txAntennaSize     = [32 1 1 1 1];
params.maxDelay          = 32;  % Fixed for all configs

samplesPerProfile = 5000;
params.numTrainSamples = 4000;
params.numValSamples   = 500;
params.numTestSamples  = 500;

%% Carrier Config
carrier = nrCarrierConfig;
params.nSizeGrid = round(params.NumSubcarriers / 12); 
carrier.SubcarrierSpacing = params.subcarrierSpacing;

waveInfo = nrOFDMInfo(carrier);
samplesPerSlot = sum(waveInfo.SymbolLengths(1:waveInfo.SymbolsPerSlot));

%% Data Generation
if doGenerateData
    rootDir = fullfile(pwd, "Data", "CDLData");

    for profile = cdlProfiles
        baseDS = baseDelaySpreads(profile);
        isLOS = cdlLOSMap(profile);

        for vFactor = variationFactors
            delaySpread = baseDS * vFactor;

            for doppler = dopplers
                for boost = ["Baseline", "Boosted"]

                    % Debug print config
                    fprintf('\n--- Preparing config ---\n');
                    fprintf('Profile       : %s\n', profile);
                    fprintf('DelaySpread   : %.1f ns (vFactor=%.1f)\n', delaySpread * 1e9, vFactor);
                    fprintf('Doppler       : %.2f Hz (Velocity ≈ %.1f km/h)\n', doppler, doppler * 3.6 / fc * c);
                    fprintf('Boosted LOS   : %s\n', boost);
                    fprintf('Fixed maxDelay: %d\n', params.maxDelay);

                    % Channel setup
                    channel = nrCDLChannel;
                    channel.DelayProfile = profile;
                    channel.DelaySpread = delaySpread;
                    channel.MaximumDopplerShift = doppler;
                    channel.RandomStream = "Global stream";
                    channel.TransmitAntennaArray.Size = params.txAntennaSize;
                    channel.ReceiveAntennaArray.Size = params.rxAntennaSize;
                    channel.ChannelFiltering = false;
                    channel.NumTimeSamples = samplesPerSlot;
                    channel.SampleRate = waveInfo.SampleRate;

                    if strcmp(boost, "Boosted") && isLOS
                        channel.KFactor = 10;
                    else
                        channel.KFactor = 0;
                    end

                    channelInfo = info(channel);

                    % Generation options
                    opt.NumSubcarriers = params.NumSubcarriers;
                    opt.NumSymbols     = carrier.SymbolsPerSlot;
                    opt.NumTxAntennas  = channelInfo.NumInputSignals;
                    opt.NumRxAntennas  = channelInfo.NumOutputSignals;
                    opt.maxDelay       = params.maxDelay;
                    opt.targetStd      = 0.0212;
                    opt.targetMean     = 0.5;

                    tag = sprintf('%s_DS%.0fns_D%.0fHz_%s', ...
                        profile, delaySpread*1e9, doppler, boost);

                    dataDir = rootDir;
                    dataSetName = "CDLChannelEst_" + tag;

                    fprintf('Tag           : %s\n', tag);
                    fprintf('Output shape  : [1 32 32 2] expected\n');
                    fprintf('Generating    : %d samples\n', samplesPerProfile);

                    if ~dataSetExists(dataDir, dataSetName, channel, carrier)
                        tic;
                        fprintf('⏳ Generating %s ...\n', dataSetName);
                        generateData(dataDir, dataSetName, samplesPerProfile, carrier, channel, opt);
                        elapsed = toc;
                        fprintf('✅ Done: %s in %s\n', tag, string(seconds(elapsed)));
                    else
                        fprintf('⚠️  %s already exists. Skipping.\n', dataSetName);
                    end
                end
            end
        end
    end
end

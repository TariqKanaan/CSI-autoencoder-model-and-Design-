%% CDL-A to CDL-E CSI Visualization Script
clear; clc;

% Visualization toggle
doVisualizeCSI = true;

% CDL profiles to visualize
cdlProfiles = ["CDL-A", "CDL-B", "CDL-C", "CDL-D", "CDL-E"];

% Common Parameters
params.nSizeGrid         = 52;
params.subcarrierSpacing = 15; % kHz
params.NumSubcarriers    = params.nSizeGrid * 12;
params.rxAntennaSize     = [2 1 1 1 1];
params.txAntennaSize     = [2 2 2 1 1];
params.maxDoppler        = 5;              % Hz
params.rmsDelaySpread    = 300e-9;         % seconds
params.truncationFactor  = 10;             % Compression truncation

% Derived Constants
Tdelay = 1 / (params.NumSubcarriers * params.subcarrierSpacing * 1e3);
rmsTauSamples = params.rmsDelaySpread / Tdelay;
params.maxDelay = round((params.rmsDelaySpread / Tdelay) * params.truncationFactor / 2) * 2;

% Carrier and OFDM Info
carrier = nrCarrierConfig;
carrier.NSizeGrid = params.nSizeGrid;
carrier.SubcarrierSpacing = params.subcarrierSpacing;
waveInfo = nrOFDMInfo(carrier);
samplesPerSlot = sum(waveInfo.SymbolLengths(1:waveInfo.SymbolsPerSlot));

%% Loop through CDL profiles
for iProfile = 1:length(cdlProfiles)
    profile = cdlProfiles(iProfile);
    fprintf('--- Visualizing Profile: %s ---\n', profile);

    % Channel Configuration
    channel = nrCDLChannel;
    channel.DelayProfile = profile;
    channel.DelaySpread = params.rmsDelaySpread;
    channel.MaximumDopplerShift = params.maxDoppler;
    channel.RandomStream = "Global stream";
    channel.TransmitAntennaArray.Size = params.txAntennaSize;
    channel.ReceiveAntennaArray.Size = params.rxAntennaSize;
    channel.ChannelFiltering = false;
    channel.NumTimeSamples = samplesPerSlot;
    channel.SampleRate = waveInfo.SampleRate;

    if doVisualizeCSI
        % Generate one channel realization
        [pathGains, sampleTimes] = channel();
        pathFilters = getPathFilters(channel);
        offset = 0;

        % Perfect channel estimation
        Hest = nrPerfectChannelEstimate(carrier, pathGains, pathFilters, offset, sampleTimes);
        reset(channel);

        % Plot channel response
        helperPlotChannelResponse(Hest);
        sgtitle(sprintf('Raw Channel Response - %s', profile));

        % Mean across symbols
        Hmean = squeeze(mean(Hest, 2)); % [Subcarrier x Rx x Tx]
        Hmean = permute(Hmean, [1 3 2]); % [Subcarrier x Tx x Rx]

        % 2D FFT
        Hdft2 = fft2(Hmean);

        % Truncation in frequency domain
        nSub = size(Hmean,1);
        midPoint = floor(nSub / 2);
        lowerEdge = midPoint - (nSub - params.maxDelay) / 2 + 1;
        upperEdge = midPoint + (nSub - params.maxDelay) / 2;
        Htemp = Hdft2([1:lowerEdge-1 upperEdge+1:end], :, :);

        % Inverse FFT for compressed CSI
        Htrunc = ifft2(Htemp);
        HtruncReal = cat(3, real(Htrunc), imag(Htrunc));

        % Plot In-Phase and Quadrature
        figure('Name', sprintf('CSI Compression - %s', profile));
        subplot(1,2,1);
        imagesc(HtruncReal(:,:,1));
        title('In-Phase'); xlabel('Tx Antennas'); ylabel('Compressed Subcarriers'); colorbar;

        subplot(1,2,2);
        imagesc(HtruncReal(:,:,2));
        title('Quadrature'); xlabel('Tx Antennas'); ylabel('Compressed Subcarriers'); colorbar;

        % Step-by-step visualization
        helperPlotCSIFeedbackPreprocessingSteps( ...
            Hmean(:,:,1), Hdft2(:,:,1), Htemp(:,:,1), Htrunc(:,:,1), ...
            nSub, size(Hmean,2), params.maxDelay, sprintf("Freq-Spatial - %s", profile));
    end
end

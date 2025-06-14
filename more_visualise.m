%% CDL-C CSI Visualization Script with Extended Visualizations and Numerical Sanity Checks
clear; clc;

% --- Visualization Toggle ---
doVisualizeCSI = true;

% --- CDL Profile to Visualize ---
profile = "CDL-C";  % Change this to visualize a different profile
fprintf('--- Visualizing Profile: %s ---\n', profile);

% --- Common Parameters ---
params.nSizeGrid         = 52;
params.subcarrierSpacing = 15;       % kHz
params.NumSubcarriers    = params.nSizeGrid * 12;
params.rxAntennaSize     = [2 1 1 1 1];
params.txAntennaSize     = [2 2 2 1 1];
params.maxDoppler        = 5;        % Hz
params.rmsDelaySpread    = 300e-9;   % seconds
params.truncationFactor  = 10;       % Compression truncation

% --- Derived Constants ---
Tdelay = 1 / (params.NumSubcarriers * params.subcarrierSpacing * 1e3);
params.maxDelay = round((params.rmsDelaySpread / Tdelay) * ...
                        params.truncationFactor / 2) * 2;

% --- Carrier and OFDM Info ---
carrier = nrCarrierConfig;
carrier.NSizeGrid         = params.nSizeGrid;
carrier.SubcarrierSpacing = params.subcarrierSpacing;
waveInfo = nrOFDMInfo(carrier);
samplesPerSlot = sum(waveInfo.SymbolLengths(1:waveInfo.SymbolsPerSlot));

% --- Channel Configuration ---
channel = nrCDLChannel;
channel.DelayProfile          = profile;
channel.DelaySpread           = params.rmsDelaySpread;
channel.MaximumDopplerShift   = params.maxDoppler;
channel.RandomStream          = "Global stream";
channel.TransmitAntennaArray.Size  = params.txAntennaSize;
channel.ReceiveAntennaArray.Size   = params.rxAntennaSize;
channel.ChannelFiltering      = false;
channel.NumTimeSamples        = samplesPerSlot;
channel.SampleRate            = waveInfo.SampleRate;

% --- Generate One Channel Realization ---
[pathGains, sampleTimes] = channel();
pathFilters  = getPathFilters(channel);
offset = 0;

% --- Perfect Channel Estimation ---
Hest = nrPerfectChannelEstimate(carrier, pathGains, pathFilters, offset, sampleTimes);
reset(channel);

% --- Plot Raw Channel Response ---
helperPlotChannelResponse(Hest);
sgtitle(sprintf('Raw Channel Response - %s', profile));

% --- Time-Averaged Channel Estimation ---
Hmean = squeeze(mean(Hest, 2));    % [Subcarrier x Rx x Tx]
Hmean = permute(Hmean, [1 3 2]);   % [Subcarrier x Tx x Rx]

% --- 2D FFT (Frequency–Spatial) ---
Hdft2 = fft2(Hmean);               % [Subcarrier x Tx x Rx]

% --- Frequency Domain Truncation (keeping 'params.maxDelay' rows) ---
nSub     = size(Hmean, 1);
midPoint = floor(nSub / 2);
lowerEdge = midPoint - (nSub - params.maxDelay)/2 + 1;
upperEdge = midPoint + (nSub - params.maxDelay)/2;

% Extract only the rows we keep
keptIdx = [1:(lowerEdge-1), (upperEdge+1):nSub];
Htemp_small = Hdft2(keptIdx, :, :);  % [maxDelay x Tx x Rx]

% To reconstruct full-sized frequency matrix with zeros in truncated part:
Htemp_full = zeros(size(Hdft2));     % [nSub x Tx x Rx]
Htemp_full(keptIdx, :, :) = Htemp_small;

% --- Compressed CSI via 2D IFFT on Full-Sized Matrix ---
Htrunc_full = ifft2(Htemp_full);     % [nSub x Tx x Rx]
HtruncReal_full = cat(3, real(Htrunc_full), imag(Htrunc_full));

% --- Plot In-Phase and Quadrature Components (Full-Sized Compressed CSI) ---
figure('Name', sprintf('CSI Compression - %s', profile));
subplot(1,2,1);
imagesc(HtruncReal_full(:,:,1));  % For Rx1 (Real part)
title('In-Phase (Real Part)');
xlabel('Tx Antennas');
ylabel('Subcarriers');
colorbar;

subplot(1,2,2);
imagesc(HtruncReal_full(:,:,2));  % For Rx1 (Imag part)
title('Quadrature (Imag Part)');
xlabel('Tx Antennas');
ylabel('Subcarriers');
colorbar;

% --- Step-by-Step CSI Feedback Preprocessing Visualization ---
helperPlotCSIFeedbackPreprocessingSteps( ...
    Hmean(:,:,1), ...           % [Subcarrier x Tx] for Rx1
    Hdft2(:,:,1), ...           % Full 2D FFT for Rx1
    Htemp_full(:,:,1), ...      % Full-sized truncated FFT for Rx1
    Htrunc_full(:,:,1), ...     % Compressed CSI for Rx1
    nSub, size(Hmean,2), params.maxDelay, ...
    sprintf("Freq-Spatial Steps - %s", profile));

%%% -------------------- ADDITIONAL VISUALIZATIONS -------------------- %%%

%% 1. Doppler Spectrum (Time Variance)
figure('Name', 'Doppler Spectrum');
HtimeVar = squeeze(Hest(:, :, 1));  % [Subcarrier x Time] for Tx1→Rx1
dopplerSpec = abs(fftshift(fft(HtimeVar, [], 2), 2));
imagesc(dopplerSpec);
title('Doppler Spectrum (Tx1 → Rx1)');
xlabel('OFDM Symbols (Time)');
ylabel('Subcarriers');
colorbar;

%% 2. Channel Impulse Response (Average over Rx)
% IFFT across subcarriers to get delay-domain taps:
Himpulse = ifft(Hmean, [], 1);      % [nSub x Tx x Rx]
Havg = mean(abs(Himpulse), 3);      % [nSub x Tx] averaged over Rx
figure('Name', 'Channel Impulse Response');
imagesc(Havg);
title('Average Channel Impulse Response (over Rx)');
xlabel('Tx Antennas');
ylabel('Delay Taps');
colorbar;

%% 3. Singular Values of Channel Matrix
figure('Name', 'Singular Values of Channel');
[~, S, ~] = svd(Hmean(:,:,1));      % Rx1
singularValues = diag(S);
stem(singularValues, 'filled');
title('Singular Values of Channel Matrix (Tx1 → Rx1)');
xlabel('Index');
ylabel('Singular Value');

%% 4. Channel Capacity Estimation (MIMO)
snr_dB = 20;                        % Example SNR
H1 = Hmean(:,:,1);                  % [nSub x Tx] for Rx1
% Estimate capacity using single-subcarrier snapshot (for illustration)
capacity = log2(det(eye(size(H1,1)) + ...
            10^(snr_dB/10)/size(H1,2) * (H1 * H1')));
fprintf('Estimated Channel Capacity (CDL-C, SNR=%d dB): %.2f bps/Hz\n', snr_dB, real(capacity));

%% 5. Phase of Channel Coefficients
figure('Name', 'Phase of Channel Coefficients');
imagesc(angle(Hmean(:,:,1)));      % Rx1
title('Phase of Channel Coefficients (Tx1 → Rx1)');
xlabel('Tx Antennas');
ylabel('Subcarriers');
colorbar;

%% 6. CSI Compression Error Visualization
figure('Name', 'CSI Compression Error');
compressionError = abs(Hmean(:,:,1) - Htrunc_full(:,:,1));  % Rx1
imagesc(compressionError);
title('CSI Compression Error (|Hmean - Htrunc|) (Tx1 → Rx1)');
xlabel('Tx Antennas');
ylabel('Subcarriers');
colorbar;

%% 7. Antenna Correlation Matrix (Magnitude)
% Reshape [Subcarrier x Tx x Rx] → [ (nSub*Rx) x Tx ] for correlation
Habs = reshape(abs(Hmean), [], size(Hmean,2));  % [ (nSub*Rx) x Tx ]
corrMatrix = corrcoef(Habs);
figure('Name', 'Antenna Correlation Matrix');
imagesc(corrMatrix);
title('Antenna Correlation Matrix (Magnitude)');
xlabel('Tx Antenna Index');
ylabel('Tx Antenna Index');
colorbar;

%% -------------------- Numerical Sanity Checks --------------------
% 1) CIR Total Power Should ≈ 1 (unit power assumption)
%    Compute per-Tx average over delays and Rx antennas:
cirPowerMatrix = sum(abs(Himpulse).^2, 1);   % [1 x Tx x Rx]: sum over delay taps
cirPowerPerTx = squeeze(mean(cirPowerMatrix, 3)); % [Tx x 1]: average over Rx
fprintf('Average CIR Total Power per Tx: ');
fprintf('[%.3f ', cirPowerPerTx(1));
fprintf('%.3f]\n', cirPowerPerTx(2));

% 2) Compression Error (Relative MSE in dB)
mse      = mean2(abs(Hmean(:,:,1) - Htrunc_full(:,:,1)).^2);
normPow  = mean2(abs(Hmean(:,:,1)).^2);
relErr_dB = 10*log10(mse / normPow);
fprintf('Compression Relative MSE: %.2f dB\n', relErr_dB);

% 3) Max Off-Diagonal Antenna Correlation
Habs_big = reshape(abs(Hmean), [], size(Hmean,2));  % [ (nSub*Rx) x Tx ]
corrMatrix_full = corrcoef(Habs_big);
offDiagIdx = tril(true(size(corrMatrix_full)),-1);
maxOffDiag = max(abs(corrMatrix_full(offDiagIdx)));
fprintf('Max Off-Diagonal Correlation: %.2f\n', maxOffDiag);
mse = mean2(abs(Hmean(:,:,1) - Htrunc_full(:,:,1)).^2);
normPower = mean2(abs(Hmean(:,:,1)).^2);
relError_dB = 10*log10(mse / normPower);
fprintf('Relative MSE of compression: %.2f dB\n', relError_dB);
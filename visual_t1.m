% ==============================
% visualize_CSI_processed.m
% ==============================

% 1) Set the folder where your processed CSI files live
dataFolder = 'C:\Users\TARIQ\Documents\MATLAB\Examples\R2024a\deeplearning_shared\CSICompressionAutoencoderExample _Copy\Data\CDLData';

% 2) Find all files with “_processed_” in their name
searchPattern = fullfile(dataFolder, '*_processed_*.mat');
fileList = dir(searchPattern);

% If no files found, stop
if isempty(fileList)
    error('No *_processed_*.mat files found in %s', dataFolder);
end

% 3) Display the list of available processed files
fprintf('Available “processed” CSI files:\n\n');
for idx = 1:numel(fileList)
    fprintf('  %2d: %s\n', idx, fileList(idx).name);
end
fprintf('\n');

% 4) Ask user to pick one (or just take the first if you prefer)
prompt = sprintf('Select a file to load (1–%d): ', numel(fileList));
selectedIdx = input(prompt);
if isempty(selectedIdx) || selectedIdx < 1 || selectedIdx > numel(fileList)
    error('Invalid selection.');
end

selectedFile = fullfile(dataFolder, fileList(selectedIdx).name);
fprintf('\nLoading: %s\n\n', fileList(selectedIdx).name);

% 5) Load the selected .mat file
loadedData = load(selectedFile);

% 6) Show what variables are inside
varNames = fieldnames(loadedData);
fprintf('Variables contained in this file:\n');
for k = 1:numel(varNames)
    v = loadedData.(varNames{k});
    sz = size(v);
    fprintf('   • %-20s   (size: [%s])\n', varNames{k}, num2str(sz));
end
fprintf('\n');

% 7) Attempt to identify a CSI-like array (3D or 4D)
%    We look for the first variable with at least 3 dimensions.
csiVarName = '';
for k = 1:numel(varNames)
    v = loadedData.(varNames{k});
    if ndims(v) >= 3 && isnumeric(v)
        csiVarName = varNames{k};
        break;
    end
end

if isempty(csiVarName)
    error('No 3D-or-4D numeric array found to visualize as CSI.');
end

fprintf('→ Identified CSI variable: ''%s''  (size: [%s])\n\n', ...
    csiVarName, num2str(size(loadedData.(csiVarName))));

% 8) Extract that variable
CSI = loadedData.(csiVarName);

% 9) Decide which “slice” to plot
%    If CSI is 4D (e.g. antennas × subcarriers × OFDM symbols × samples),
%    you can pick a particular OFDM symbol (dim=3) and/or a particular sample (dim=4).
dims = ndims(CSI);
sz   = size(CSI);

switch dims
    case 3
        % Common layout: [antennas × subcarriers × snapshots]
        % We’ll pick snapshot #1
        sliceIdx = 1;
        data2D = abs(CSI(:, :, sliceIdx));
        titleStr = sprintf('%s (abs, slice %d of %d)', csiVarName, sliceIdx, sz(3));
    case 4
        % Common layout: [antennas × subcarriers × OFDMsymbol × sampleIndex]
        % Let’s pick OFDMsymbol = 1, sampleIndex = 1
        symIdx     = 1;
        sampleIdx  = 1;
        data2D = abs(squeeze(CSI(:, :, symIdx, sampleIdx)));
        titleStr = sprintf('%s (abs, OFDMsym %d, sample %d)', ...
                           csiVarName, symIdx, sampleIdx);
    otherwise
        error('Unsupported number of dimensions: %d', dims);
end

% 10) Plot the chosen 2D “slice”
figure('Name','CSI Visualization','NumberTitle','off');
imagesc(data2D);
xlabel('Subcarriers');
ylabel('Antenna elements');
title(titleStr);
colorbar;

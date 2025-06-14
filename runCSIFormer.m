%% runCSIFormer.m
% CSI Feedback with Transformer Autoencoder

% 1) Determine projectRoot (handles script vs. function case)
thisFile = mfilename("fullpath");
if ~isempty(thisFile)
    [projectRoot, ~, ~] = fileparts(thisFile);
else
    projectRoot = pwd;
end

% 2) Add required subfolders to MATLAB path
%    – CSIFormer helpers live in models/CSIFormer
%    – CSINet helpers (if still needed) live in models/CSINet
%    – Training scripts live in train/
%    – Utilities may be under projectRoot/utils or projectRoot/experiments/utils
pathsToAdd = {
    fullfile(projectRoot, "models", "CSIFormer"), ...
    fullfile(projectRoot, "models", "CSINet"), ...
    fullfile(projectRoot, "train")
};

% Look for utils in either location
utilsPath1 = fullfile(projectRoot, "utils");
utilsPath2 = fullfile(projectRoot, "experiments", "utils");
if isfolder(utilsPath1)
    pathsToAdd{end+1} = utilsPath1;
elseif isfolder(utilsPath2)
    pathsToAdd{end+1} = utilsPath2;
else
    warning("No utils folder found under projectRoot:\n  %s\n  %s\nProceeding without utilities.", ...
            utilsPath1, utilsPath2);
end

% Add all existing folders on the path
for p = pathsToAdd
    if isfolder(p{1})
        addpath(p{1});
        fprintf("Added to path: %s\n", p{1});
    end
end

% 3) Locate processedData under Data/
dataFolder = fullfile(projectRoot, "Data", "processedData");
if ~isfolder(dataFolder)
    error("Cannot find folder:\n    %s\nEnsure your Data/processedData path is correct.", dataFolder);
end

% 4) Load training data via signalDatastore
trainPattern = fullfile(dataFolder, "CDLChannelEst_train*.mat");
trainSds = signalDatastore(trainPattern);
if isempty(trainSds.Files)
    error("No training files found matching '%s'.", trainPattern);
end

HTrainCell = readall(trainSds);
HTrain = cat(1, HTrainCell{:});
HTrain = permute(HTrain, [2,3,4,1]);
fprintf("Loaded %d training samples.\n", size(HTrain,4));

valPattern = fullfile(dataFolder, "CDLChannelEst_val*.mat");
valSds = signalDatastore(valPattern);
if isempty(valSds.Files)
    error("No validation files found matching '%s'.", valPattern);
end

HValCell = readall(valSds);
HVal = cat(1, HValCell{:});
HVal = permute(HVal, [2,3,4,1]);
fprintf("Loaded %d validation samples.\n", size(HVal,4));

% 5) Compute input/encoded sizes
inputSize   = size(HTrain(:,:,:,1));    % e.g. [subcarriers, antennas, 2]
encodedSize = 64;
compressionFactor = prod(inputSize) / encodedSize;
disp("Compression rate of the autoencoder is " + num2str(compressionFactor) + ":1");

% 6) Build the Encoder
embeddingSize = 16;

encConvBlock = [
    imageInputLayer(inputSize, Name="enc_input")
    convolution2dLayer([5,5], embeddingSize, Padding=[2,2,2,2], Name="enc_conv2D")
];

numPatches = (inputSize(1)*inputSize(2)) / (2*2);

flattenBlock = [
    convolution2dLayer([2,2], embeddingSize, Stride=[2,2], Name="enc_patches")
    functionLayer(@(X) dlarray(reshape(X, numPatches, embeddingSize, []), 'SCB'), ...
                  Formattable=true, Name="enc_flatten")
];

linearProj = helperCSIFormerLinearProjectionLayer(embeddingSize, Name="enc_linearProject");

net = dlnetwork([
    encConvBlock
    flattenBlock
    linearProj
]);

posEmbed = positionEmbeddingLayer(embeddingSize, numPatches, ...
    PositionDimension="spatial", Name="enc_posEmbed");
net = addLayers(net, posEmbed);
net = connectLayers(net, "enc_flatten", "enc_posEmbed");

add1 = additionLayer(2, Name="enc_add");
net = addLayers(net, add1);
net = connectLayers(net, "enc_linearProject", "enc_add/in1");
net = connectLayers(net, "enc_posEmbed",     "enc_add/in2");

figure("Name","Encoder—Initial"); plot(net); title("Encoder: Patch + PosEmbed + Res");

% 7) Add W-MSA and MLP blocks
windowSize = 8;

net = helperAddWMSALayersBlock(net, "enc_add", "enc_", windowSize);

figure("Name","Encoder—WMSA"); plot(net); title("After WMSA");

net = helperAddUnpatched2DConvLayersBlock(net, "enc_add3", "enc_UPC1_");
net = helperAddFlattenPatchLayersBlock(net, "enc_UPC1_conv2D", "enc_", numPatches, embeddingSize);

add4 = additionLayer(2, Name="enc_add4");
net = addLayers(net, add4);
net = connectLayers(net, "enc_add",      "enc_add4/in1");
net = connectLayers(net, "enc_flatten1", "enc_add4/in2");
net = helperAddUnpatched2DConvLayersBlock(net, "enc_add4", "enc_UPC2_");

add5 = additionLayer(2, Name="enc_add5");
net = addLayers(net, add5);
net = connectLayers(net, "enc_conv2D",     "enc_add5/in1");
net = connectLayers(net, "enc_UPC2_conv2D", "enc_add5/in2");

% 8) Final encoder output
encOutputBlock = [
    convolution2dLayer([3,3], 2, Padding=[1,1], Name="enc_conv2D_2")
    fullyConnectedLayer(encodedSize, Name="enc_output")
];
net = addLayers(net, encOutputBlock);
CSIFormerEnc = connectLayers(net, "enc_add5", "enc_conv2D_2");

figure("Name","Encoder—Final"); plot(CSIFormerEnc); title("Full Encoder Graph");

% 9) Create full autoencoder
CSIFormerNet = helperCSIFormerCreateNetwork(inputSize, encodedSize, ...
                                            embeddingSize, numPatches, windowSize);

% 10) Analyze
netInfo = analyzeNetwork(CSIFormerNet);
fprintf("Total number of learnables: %.3f M\n", netInfo.TotalLearnables / 1e6);

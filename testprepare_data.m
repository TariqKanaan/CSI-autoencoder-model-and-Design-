% Before cropping:
szBefore = size(HTrainReal);       % [58 32 2 20032]
disp("Before cropping: " + mat2str(szBefore));

% Crop to 32×32 in spatial dims
HTrainReal = HTrainReal(1:32, 1:32, :, :);  
HValReal   = HValReal(  1:32, 1:32, :, :);  

% Verify after cropping:
szAfterTrain = size(HTrainReal);    % should be [32 32 2 20032]
szAfterVal   = size(HValReal);      % e.g. [32 32 2 10016]
disp("After cropping (train): " + mat2str(szAfterTrain));
disp("After cropping (val):   " + mat2str(szAfterVal));

% Now define network with inputSize = [32 32 2]
inputSize = szAfterTrain(1:3);      % [32 32 2]
encodedSize  = 64;
embeddingSize = 16;
windowSize   = 8;

% numPatches = (32/2)*(32/2) = 16*16 = 256
numPatches = inputSize(1)*inputSize(2)/(2*2);  
CSIFormerNet = helperCSIFormerCreateNetwork( ...
    inputSize, encodedSize, embeddingSize, numPatches, windowSize);
%— After building CSIFormerNet …

% Determine if CSIFormerNet is already a dlnetwork
if isa(CSIFormerNet, 'dlnetwork')
    dlnet = CSIFormerNet;
else
    dlnet = dlnetwork(CSIFormerNet);
end
% 2. Initialize all learnable parameters
dlnet = initialize(dlnet);
% Now run a dummy forward pass
inputSize = size(HTrainReal(:,:,:,1));   % should be [32 32 2]
dummyIn  = rand(inputSize, 'single');
dlX      = dlarray(dummyIn, "SSCB");     % Spatial-Spatial-Channel-Batch

try
    dlY = predict(dlnet, dlX);
    disp("→ Forward pass OK: No reshape errors");
catch ME
    disp("! Forward-pass error: " + ME.message);
end

try
    dlY = predict(dlnet, dlX);
    disp("→ Forward pass OK: No reshape errors");
catch ME
    disp("! Forward‐pass error: " + ME.message);
end

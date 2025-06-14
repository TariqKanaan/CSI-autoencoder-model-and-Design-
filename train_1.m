trainNow= true
saveNetwork=true;

trainingMetric = @(x,t) helperNMSELossdB(x,t);
epochs = 10;
batchSize = 500;
initLearnRate = 8e-4; 

options = trainingOptions("adam", ...
    InitialLearnRate=initLearnRate, ...
    LearnRateSchedule="piecewise", ...
    LearnRateDropPeriod=30, ...  % Warm-up for first 30 epochs
    LearnRateDropFactor=0.1, ...
    MaxEpochs=epochs, ...
    MiniBatchSize=batchSize, ...
    Shuffle="every-epoch", ...
    ValidationData={HValReal,HValReal}, ...
    ValidationFrequency=200, ...
    OutputNetwork="best-validation-loss", ...
    Metrics=trainingMetric, ...
    Plots="training-progress", ...
    Verbose=false, ...
    VerboseFrequency=100);


if trainNow
    [trainedNet, trainInfo] = trainnet(HTrainReal,HTrainReal,CSIFormerNet,@(X,T) mse(X,T),options);%#ok

    if saveNetwork
        save("CSIFormerTrainedNetwork_" ...
            + string(datetime("now","Format","dd_MM_HH_mm")), 'trainedNet')
    end

else
    trainedNetName = "trainedNetEnc" + num2str(compressionFactor);
    load("CSIFormerTrainedNets.mat",trainedNetName)
end
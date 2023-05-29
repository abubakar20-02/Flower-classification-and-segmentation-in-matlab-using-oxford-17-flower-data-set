%%

% Read Me:
% To test the model, make sure to run the testModel function only. 
% The trainig and pre-processing have been commented out to avoid them running.
% If you need to train the model then uncomment the Pre-processing and training.
% The test data set has to be in the a folder called DataSet with a
% sub-folder called test, then place the test set inside ImagesRsz256.

%%

filePath = pwd;
modelName= "segmentnet.mat";
trainSplit=0.8;
inputSize = [256,256,3];
dataSetDir= fullfile(filePath, 'DataSet');
srcimageDir = fullfile(filePath, 'daffodilSeg');

%%

% %Preprocessing
% recreateDirectory(filePath,'DataSet');
% splitTrainTest(srcimageDir,dataSetDir,trainSplit);
%%

% %training
% train(modelName,inputSize,dataSetDir);

%%

%testing the model
testModel(modelName,filePath);

%%

function train(modelName,inputSize,dataSetDir)

    imds = imageDatastore(fullfile(dataSetDir, 'train/ImagesRsz256'));
    pixelLabelID = [1,3];
    classNames = ["flower", "background"];
    pxds = pixelLabelDatastore(fullfile(dataSetDir, 'train/LabelsRsz256'), classNames, pixelLabelID, 'FileExtensions', '.png', 'ReadFcn', @(x) uint8(replacePixels(imread(x), [2, 4], 3)));
    % Load pre-trained DeepLabv3+
    net = deeplabv3plusLayers(inputSize, 2, 'resnet18');
    augmenter = imageDataAugmenter( ...
        'RandXReflection', true, ...
        'RandRotation', [-15, 15], ...
        'RandXTranslation', [-5 5], ...
        'RandYTranslation', [-5 5]); 
    
    impxds = pixelLabelImageDatastore(imds, pxds, 'DataAugmentation',augmenter);
    % Set up training options
    opts = trainingOptions( ...
        'adam', ...
        'InitialLearnRate', 1e-4, ...
        'MaxEpochs', 30, ...
        'MiniBatchSize', 32, ...
        'Plots', 'training-progress',...
        'Shuffle', 'every-epoch', ...
        'VerboseFrequency', 2, ...
        'ExecutionEnvironment', 'gpu');
        
    % Train the network
    net = trainNetwork(impxds, net, opts);

    save(modelName, 'net')

end

%%

function testModel(Model,filePath)

    pixelLabelID = [1, 3];
    classNames = ["flower", "background"];
    
    load(Model, 'net')
    
    imgtestdir = fullfile(filePath, 'DataSet/test/ImagesRsz256');
    tsds = imageDatastore(imgtestdir);
    
    testLabelDir = fullfile(filePath, 'DataSet/test/LabelsRsz256');
    tsPxds = pixelLabelDatastore(testLabelDir, classNames, pixelLabelID, 'FileExtensions', '.png', 'ReadFcn', @(x) uint8(replacePixels(imread(x), [0, 2, 4], 3)));
    
    % Create a combined datastore for test images and their labels
    testData = combine(tsds,tsPxds);
    
    recreateDirectory(filePath, 'overlay');
    outDir = fullfile(filePath, 'overlay');
    pxdsResults = semanticseg(tsds,net,"WriteLocation", outDir);
    evalResult = evaluateSemanticSegmentation(pxdsResults, testData);
    
    disp(evalResult);
    
    figure;
    
    numTestImages = numel(tsds.Files);
    
    numImagesToDisplay = min(4, numTestImages);
    
    indices = 1:numImagesToDisplay;
    
    for i = 1:numImagesToDisplay
        % Display overlay
        subplot(2, 2, i);
        overlayOut = labeloverlay(readimage(tsds, indices(i)), readimage(pxdsResults, indices(i))); %overlay
        imshow(overlayOut);
    end

end

%%

function img = replacePixels(img, oldVals, newVal)

    for i = 1:numel(oldVals)
        img(img == oldVals(i)) = newVal;
    end

end

%%

function splitTrainTest(srcDir, dstDir, trainRatio)

    srcImg=fullfile(srcDir, 'ImagesRsz256');
    srcLabel=fullfile(srcDir, 'LabelsRsz256');

    Images = dir(fullfile(srcImg, '*.png')); 
    Labels = dir(fullfile(srcLabel, '*.png'));  
    
    indices = randperm(length(Images));
    
    % Determine the index at which to split the files into train and test
    splitIndex = round(trainRatio * length(Images));
    
    % Create directories for the training and testing sets
    if ~exist(fullfile(dstDir, 'train'), 'dir')
        mkdir(fullfile(dstDir, 'train'));
        mkdir(fullfile(dstDir, 'train/ImagesRsz256'));
        mkdir(fullfile(dstDir, 'train/LabelsRsz256'));
    end
    if ~exist(fullfile(dstDir, 'test'), 'dir')
        mkdir(fullfile(dstDir, 'test'));
        mkdir(fullfile(dstDir, 'test/ImagesRsz256'));
        mkdir(fullfile(dstDir, 'test/LabelsRsz256'));
    end

    % Move the files to the appropriate directories
    for i = 1:length(indices)
        if i <= splitIndex
            % This file is part of the training set

            copyfile(fullfile(Images(indices(i)).folder,Images(indices(i)).name), fullfile(dstDir, 'train/ImagesRsz256'));
            copyfile(fullfile(Labels(indices(i)).folder,Labels(indices(i)).name), fullfile(dstDir, 'train/LabelsRsz256'));
        else
            % This file is part of the testing set
            copyfile(fullfile(Images(indices(i)).folder,Images(indices(i)).name), fullfile(dstDir, 'test/ImagesRsz256'));
            copyfile(fullfile(Labels(indices(i)).folder,Labels(indices(i)).name), fullfile(dstDir, 'test/LabelsRsz256'));
        end
    end

end

%%

function recreateDirectory(rootDir, dirName)

    dirFullPath = fullfile(rootDir, dirName);

    % Check if the directory exists. If yes, delete it.
    if exist(dirFullPath, 'dir')
        rmdir(dirFullPath, 's'); % 's' option allows for non-empty directory removal
    end

    % Create the directory
    mkdir(rootDir, dirName);

end

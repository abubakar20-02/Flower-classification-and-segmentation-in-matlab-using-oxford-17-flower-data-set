%%

% Read Me:
% To test the model, make sure to run the testModel function only. 
% The trainig and pre-processing have been commented out to avoid them running.
% If you need to train the model then uncomment the Pre-processing and training.
% The test data set has to be in the a folder called ImageData with a
% sub-folder called test.

%%
filePath = pwd;

NumberofImagesPerClass = 80;
imageSize = [256,256];
trainTestValidationSplit = {0.7,0.1,0.2};
inputSize = [256 256 3];
modelName= "classnet.mat";

% The directory where your images are stored
srcDir = fullfile(filePath,'17flowers'); 
ImagePath = fullfile(filePath,'ImageData');
dstDir =ImagePath;
trainDir=fullfile(dstDir,'train');
testDir=fullfile(dstDir,'test');
valDir=fullfile(dstDir,'val');

flowerNames = {'Daffodil','Snowdrop','Lilly Valley','Bluebell','Crocus','Iris','Tigerlily','Tulip','Fritillary','Sunflower','Daisy','Colts Foot','Dandelion','Cowslip','Buttercup','Windflower','Pansy'};



%%

% %Preprocessing

% recreateDirectory(filePath,'ImageData');
% datasplit_sort(srcDir, dstDir, flowerNames,NumberofImagesPerClass,trainTestValidationSplit,ImagePath);
% resize_images(dstDir,imageSize)

%%

% %training

% train(modelName,inputSize);

%%
%testing the model
testModel(modelName);

%%
%%
%%
function recreateDirectory(rootDir, dirName)
    % Construct the full path of the directory
    dirFullPath = fullfile(rootDir, dirName);

    % Check if the directory exists. If yes, delete it.
    if exist(dirFullPath, 'dir')
        rmdir(dirFullPath, 's'); % 's' option allows for non-empty directory removal
    end

    % Create the directory
    mkdir(rootDir, dirName);
end
%%
function datasplit_sort(srcDir, dstDir, dirNames,NumberofImagesPerClass,trainTestValidationSplit,ImagePath)
    % Get a list of all JPEG files in the source directory
    srcFiles = dir(fullfile(srcDir, '*.jpg')); 

    % Ensure we have enough images
    assert(numel(srcFiles) >= NumberofImagesPerClass * numel(dirNames)); 

    % Calculate the number of train, validation, and test images per class
    numTrainImages = floor(NumberofImagesPerClass * trainTestValidationSplit{1});
    numValImages = floor(NumberofImagesPerClass * trainTestValidationSplit{2});

    % Create train, validation, and test directories
    trainDir = fullfile(dstDir, 'train');
    valDir = fullfile(dstDir, 'val');
    testDir = fullfile(dstDir, 'test');

    recreateDirectory(ImagePath,'train')
    recreateDirectory(ImagePath,'val')
    recreateDirectory(ImagePath,'test')

    % Loop over the new directories
    for i = 1:numel(dirNames)
        % Create the new directory for train, validation, and test
        recreateDirectory(trainDir, dirNames{i})
        recreateDirectory(valDir, dirNames{i})
        recreateDirectory(testDir, dirNames{i})

        imageIndices = (i-1)*NumberofImagesPerClass + 1 : i*NumberofImagesPerClass;
        % Randomize the order of the images
        randIndices = randperm(NumberofImagesPerClass);
        
        for j = 1:NumberofImagesPerClass
            idx = imageIndices(randIndices(j));
            
            % Copy the image to the new train, validation or test directory
            if j <= numTrainImages
                copyfile(fullfile(srcFiles(idx).folder, srcFiles(idx).name), fullfile(trainDir, dirNames{i}, srcFiles(idx).name));
            elseif j <= numTrainImages + numValImages
                copyfile(fullfile(srcFiles(idx).folder, srcFiles(idx).name), fullfile(valDir, dirNames{i}, srcFiles(idx).name));
            else
                copyfile(fullfile(srcFiles(idx).folder, srcFiles(idx).name), fullfile(testDir, dirNames{i}, srcFiles(idx).name));
            end
        end
    end
end

%% 
function resize_images(Dir,size)
    % Get a list of all JPEG files in the source directory and its subdirectories
    srcFiles = dir(fullfile(Dir, '**', '*.jpg')); 

    % Loop over the source files
    for i = 1:numel(srcFiles)
        img = imread(fullfile(srcFiles(i).folder, srcFiles(i).name));
        
        resized_img = imresize(img, size);
        
        % Create the destination directory if it doesn't exist
        destFolder = strrep(srcFiles(i).folder, Dir, Dir);
        if ~exist(destFolder, 'dir')
            mkdir(destFolder);
        end

        % Write the image to the destination directory
        imwrite(resized_img, fullfile(destFolder, srcFiles(i).name));
    end
end

%%
function train(modelName,inputSize)
    fullPath = pwd;
    ImagePath = fullfile(fullPath,'ImageData');
    % The directory where your new folders will be created
    dstDir =ImagePath;
    valDir=fullfile(dstDir,'val');
    trainDir=fullfile(dstDir,'train');

    imds_tr = imageDatastore(trainDir, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

    imds_val = imageDatastore(valDir, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

    layers=[
        imageInputLayer(inputSize)
        convolution2dLayer(5, 32)
        batchNormalizationLayer()
        reluLayer()    
        convolution2dLayer(3, 64,"Padding", "same")
        batchNormalizationLayer()
        reluLayer()
        maxPooling2dLayer(3, "Stride", 2)
        convolution2dLayer(5, 128,"Padding", "same")
        batchNormalizationLayer()
        reluLayer()
        maxPooling2dLayer(3, "Stride", 2)
        convolution2dLayer(3, 256,"Padding", "same")
        batchNormalizationLayer()
        reluLayer()
        maxPooling2dLayer(3, "Stride", 2)
        convolution2dLayer(3, 512,"Padding", "same")
        batchNormalizationLayer()
        reluLayer()
        dropoutLayer(0.4)
        globalMaxPooling2dLayer()
        fullyConnectedLayer(512)
        reluLayer()
        fullyConnectedLayer(256)
        reluLayer()
        fullyConnectedLayer(17) 
        softmaxLayer()
        classificationLayer()];
    
    options = trainingOptions('adam', ...
        'MaxEpochs',30,...,
        'MiniBatchSize', 32, ...,
        'ValidationData', imds_val, ...
        'ValidationFrequency',30, ...
        'InitialLearnRate',1e-4,...
        'Verbose', false, ...
        'Shuffle','every-epoch' ,...
        'Plots', 'training-progress', ...
        'ExecutionEnvironment', 'gpu');
    
    
    imageAugmenter = imageDataAugmenter( ...
        'RandRotation',[-30,30], ...
        'RandXTranslation',[-5 5], ...
        'RandYTranslation',[-5 5], ...
        'RandXReflection', true);
        
    augimds = augmentedImageDatastore(inputSize,imds_tr,'DataAugmentation',imageAugmenter);
    
    net = trainNetwork(augimds, layers,options);
    
    % Save the network to a .mat file
    save(modelName, 'net');
end

function testModel(Model)
    load(Model, 'net');
    fullPath = pwd;
    ImagePath = fullfile(fullPath,'ImageData');
    % The directory where your new folders will be created
    dstDir =ImagePath;
    testDir=fullfile(dstDir,'test');
    imds_ts = imageDatastore(testDir, ...
        'IncludeSubfolders', true, ...
        'LabelSource', 'foldernames');
    YPred = classify(net,imds_ts);
    scores = predict(net, imds_ts);
    YTest = imds_ts.Labels;
    
    accuracy = sum(YPred == YTest)/numel(YTest);
    fprintf('Accuracy: %.2f\n', accuracy);
    
    % Compute the confusion matrix
    confMat = confusionmat(YTest, YPred);
    
    
    % Optionally, you can display a confusion chart
    figure;
    confChart = confusionchart(YTest, YPred);
    
    % Get the number of classes
    numClasses = size(confMat, 1);
    classNames = unique(YTest);
    
        % Initialize vectors to hold precision, recall, and F1
    precision = zeros(numClasses, 1);
    recall = zeros(numClasses, 1);
    f1 = zeros(numClasses, 1);
    
    % Calculate precision, recall, and F1 for each class
    for i = 1:numClasses
        TP = confMat(i,i);
        FP = sum(confMat(:,i)) - TP;
        FN = sum(confMat(i,:)) - TP;
        
        % Prevent division by zero for classes with no predictions
        if (TP + FP) == 0
            precision(i) = NaN;
        else
            precision(i) = TP / (TP + FP);
        end
    
        % Prevent division by zero for classes with no actual instances
        if (TP + FN) == 0
            recall(i) = NaN;
        else
            recall(i) = TP / (TP + FN);
        end
        
        % Calculate F1 score, preventing division by zero
        if (precision(i) + recall(i)) == 0
            f1(i) = NaN;
        else
            f1(i) = 2 * ((precision(i) * recall(i)) / (precision(i) + recall(i)));
        end
    end
    
    % Create a table with the results
    results = table(classNames, precision, recall, f1, 'VariableNames', {'Class', 'Precision', 'Recall', 'F1'});
    
    % Display the table
    disp(results)

    figure;

    YTest_bin = full(ind2vec(double(YTest)',17)); % Convert labels to binary matrix
    YTest_bin = YTest_bin'; % Transpose to have the same dimensions as scores
    [X,Y,T,AUC] = perfcurve(YTest_bin(:), scores(:), 1);
    plot(X,Y)
    line([0 1], [0 1], 'Color', 'k', 'LineStyle', ':'); % Creates a black dotted line from (0,0) to (1,1)); % Creates a line with a slope of 1 and an intercept of 0
    xlabel('False positive rate') 
    ylabel('True positive rate')
    title('ROC Curve')

    figure
    idx = randperm(100,30);
    for i = 1:4
        j = idx(i);
        subplot(2,2,i);
        img_path = imds_ts.Files{j};
        Label = imds_ts.Labels(j);
        I_org = imread(img_path);
        [YPred,scores] = classify(net,I_org);
        imshow(I_org)
        title([char(YPred),' ' , num2str(max(scores))])
    end
end


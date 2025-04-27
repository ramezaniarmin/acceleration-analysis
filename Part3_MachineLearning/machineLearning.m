clc;
clearvars;
close all;

%% STEP 1: Load Files

dataFolderDaily    = 'Records';
dailyFileList      = dir(fullfile(dataFolderDaily, 'daily-activity*.csv'));
numDailyFiles      = length(dailyFileList);
dailyData          = cell(numDailyFiles, 1);

for i = 1:numDailyFiles
    dailyFileName  = dailyFileList(i).name;
    dailyFilePath  = fullfile(dataFolderDaily, dailyFileName);
    dailyRawData   = readmatrix(dailyFilePath);
    dailyData{i}   = dailyRawData;
end

dataFolderSeismic  = 'Records';
seismicFileList    = dir(fullfile(dataFolderSeismic, 'seismic-record*.csv'));
numSeismicFiles    = length(seismicFileList);
seismicData        = cell(numSeismicFiles, 1);

for i = 1:numSeismicFiles
    seismicFileName = seismicFileList(i).name;
    seismicFilePath = fullfile(dataFolderSeismic, seismicFileName);
    seismicRawData  = readmatrix(seismicFilePath);
    seismicData{i}  = seismicRawData;
end

disp('All daily activity and seismic record files loaded.');

%% STEP 2: Compute Centered W Signals

dailyW   = cell(numDailyFiles, 1);
seismicW = cell(numSeismicFiles, 1);

for i = 1:numDailyFiles
    rawData    = dailyData{i};
    X          = rawData(:,1);
    Y          = rawData(:,2);
    Z          = rawData(:,3);
    Xcentered  = X - mean(X);
    Ycentered  = Y - mean(Y);
    Zcentered  = Z - mean(Z);
    W          = (Xcentered + Ycentered + Zcentered) / 3;
    dailyW{i}  = W;
end

for i = 1:numSeismicFiles
    rawData    = seismicData{i};
    X          = rawData(:,1);
    Y          = rawData(:,2);
    Z          = rawData(:,3);
    Xcentered  = X - mean(X);
    Ycentered  = Y - mean(Y);
    Zcentered  = Z - mean(Z);
    W          = (Xcentered + Ycentered + Zcentered) / 3;
    seismicW{i}= W;
end

disp('Composite signals (W) for all records computed.');

%% STEP 3: P-Wave Arrival Time Detection

dt         = 1/100;
type       = 'SM';
pflag      = 'N';
Tn         = 0.01;
xi         = 0.6;
nbins      = 10;
modeSelect = 'to_peak';

pArrivals  = zeros(numSeismicFiles, 1);

for i = 1:numSeismicFiles
    signal     = seismicW{i};
    pIndex     = PphasePicker(signal, dt, type, pflag, Tn, xi, nbins, modeSelect);

    if isnan(pIndex) || pIndex == -1
        pIndex = 0;
    end

    pArrivals(i) = pIndex;
end

disp('P-wave arrival times detected.');

%% STEP 4: Segment Seismic Signals

fs                 = 100;
segmentDurationSec = 2;
overlapDurationSec = 1;
segmentSamples     = segmentDurationSec * fs;
overlapSamples     = overlapDurationSec * fs;
segmentsPerRecord  = 10;

totalSeismicSegments = 1800;
allSeismicSegments   = zeros(totalSeismicSegments, segmentSamples);
segmentCounter       = 1;

for i = 1:numSeismicFiles
    signal     = seismicW{i};
    
    if pArrivals(i) == 0
        startSample = 1;
    else
        startSample = round(pArrivals(i) * fs);
    end

    for j = 0:(segmentsPerRecord-1)
        idxStart = startSample + j * overlapSamples;
        idxEnd   = idxStart + segmentSamples - 1;

        if idxEnd <= length(signal)
            allSeismicSegments(segmentCounter, :) = signal(idxStart:idxEnd);
            segmentCounter = segmentCounter + 1;
        end
    end
end

disp([num2str(segmentCounter-1), ' seismic segments extracted.']);

%% STEP 5: Segment Daily Activity Signals

segmentsPerDailyRecord = 8;

totalDailySegments = 1800;
allDailySegments   = zeros(totalDailySegments, segmentSamples);
segmentCounter     = 1;

rng(0);

for i = 1:numDailyFiles
    signal   = dailyW{i};
    maxStart = length(signal) - segmentSamples + 1;

    if maxStart < segmentsPerDailyRecord
        continue;
    end

    startIndices = randperm(maxStart, segmentsPerDailyRecord);

    for j = 1:segmentsPerDailyRecord
        idxStart = startIndices(j);
        idxEnd   = idxStart + segmentSamples - 1;
        allDailySegments(segmentCounter, :) = signal(idxStart:idxEnd);
        segmentCounter = segmentCounter + 1;
    end
end

disp([num2str(segmentCounter-1), ' daily activity segments extracted.']);

%% STEP 6: Feature Extraction

numSeismic = size(allSeismicSegments,1);
numDaily   = size(allDailySegments,1);

seismicFeatures = zeros(numSeismic, 8);
dailyFeatures   = zeros(numDaily, 8);

for i = 1:numSeismic
    signal = allSeismicSegments(i,:);
    seismicFeatures(i,:) = extractFeatures(signal, fs);
end

for i = 1:numDaily
    signal = allDailySegments(i,:);
    dailyFeatures(i,:) = extractFeatures(signal, fs);
end

disp('Features for seismic record and daily activity segments extracted.');

%% STEP 7: Normalize Features

allFeatures         = [seismicFeatures; dailyFeatures];
normalizedFeatures  = zeros(size(allFeatures));

for i = 1:size(allFeatures,2)
    col = allFeatures(:,i);
    colMin = min(col);
    colMax = max(col);
    normalizedFeatures(:,i) = (col - colMin) / (colMax - colMin);
end

seismicFeaturesNorm = normalizedFeatures(1:numSeismic, :);
dailyFeaturesNorm   = normalizedFeatures(numSeismic+1:end, :);

disp('Features normalized to [0,1].');

%% STEP 8: Create Dataset

seismicLabels = ones(numSeismic, 1);
dailyLabels   = zeros(numDaily, 1);

featuresAll   = [seismicFeaturesNorm; dailyFeaturesNorm];
labelsAll     = [seismicLabels; dailyLabels];

dataset       = [featuresAll, labelsAll];

disp('Dataset ready for classification.');

%% STEP 9: Classification – Single Feature at a Time (5-Fold Cross Validation)

X = dataset(:, 1:end-1);
y = dataset(:, end);

numFeatures     = size(X,2);
numClassifiers  = 4;
accuracyMatrix  = zeros(numFeatures, numClassifiers);

cv = cvpartition(y, 'KFold', 5);

for f = 1:numFeatures
    accSVM = zeros(cv.NumTestSets,1);
    accKNN = zeros(cv.NumTestSets,1);
    accDT  = zeros(cv.NumTestSets,1);
    accLR  = zeros(cv.NumTestSets,1);

    for fold = 1:cv.NumTestSets
        trainIdx = training(cv, fold);
        testIdx  = test(cv, fold);

        XTrain = X(trainIdx, f);
        XTest  = X(testIdx, f);
        yTrain = y(trainIdx);
        yTest  = y(testIdx);

        if length(unique(yTrain)) < 2
            continue;
        end

        modelSVM = fitcsvm(XTrain, yTrain, 'KernelFunction', 'rbf', 'KernelScale', 0.3, 'BoxConstraint', 1);
        accSVM(fold) = mean(predict(modelSVM, XTest) == yTest) * 100;

        modelKNN = fitcknn(XTrain, yTrain, 'NumNeighbors', 10, 'Distance', 'euclidean');
        accKNN(fold) = mean(predict(modelKNN, XTest) == yTest) * 100;

        modelDT = fitctree(XTrain, yTrain, 'MaxNumSplits', 4);
        accDT(fold) = mean(predict(modelDT, XTest) == yTest) * 100;

        modelLR = fitclinear(XTrain, yTrain, 'Learner', 'logistic');
        accLR(fold) = mean(predict(modelLR, XTest) == yTest) * 100;
    end

    accuracyMatrix(f,:) = [mean(accSVM), mean(accKNN), mean(accDT), mean(accLR)];
end

accuracyMatrix = round(accuracyMatrix,2);

disp('Single Feature Accuracy:');
disp('     SVM       KNN        DT        LR');
disp(accuracyMatrix);

%% STEP 10: Feature Correlation Heatmap

featureNames = {'CAV', 'CWT', 'FFT', 'IQR', 'PeakDiff', 'Tp', 'STD', 'ZCR'};
correlationMatrix = corr(normalizedFeatures);

figure('Position', [400, 100, 1200, 900]);
imagesc(correlationMatrix);
colormap(gray);
colorbar;
clim([-1,1]);
axis square;

set(gca, 'XTick', 1:8, 'XTickLabel', featureNames, ...
         'YTick', 1:8, 'YTickLabel', featureNames, ...
         'TickLabelInterpreter', 'latex', 'FontSize', 14);

%% STEP 11: Two-Feature Combination Classification

selectedFeatures = [1,2,6,8];  % CAV, CWT, Tp, ZCR
combinations2    = nchoosek(selectedFeatures,2);
accuracyPairs    = zeros(size(combinations2,1), numClassifiers);

for idx = 1:size(combinations2,1)
    featurePair = combinations2(idx,:);

    accSVM = zeros(5,1);
    accKNN = zeros(5,1);
    accDT  = zeros(5,1);
    accLR  = zeros(5,1);

    for fold = 1:5
        trainIdx = training(cv, fold);
        testIdx  = test(cv, fold);

        XTrain = X(trainIdx, featurePair);
        XTest  = X(testIdx, featurePair);
        yTrain = y(trainIdx);
        yTest  = y(testIdx);

        if length(unique(yTrain)) < 2
            continue;
        end

        modelSVM = fitcsvm(XTrain, yTrain, 'KernelFunction', 'rbf', 'KernelScale', 0.3, 'BoxConstraint', 1);
        accSVM(fold) = mean(predict(modelSVM, XTest) == yTest) * 100;

        modelKNN = fitcknn(XTrain, yTrain, 'NumNeighbors', 10, 'Distance', 'euclidean');
        accKNN(fold) = mean(predict(modelKNN, XTest) == yTest) * 100;

        modelDT  = fitctree(XTrain, yTrain, 'MaxNumSplits', 4);
        accDT(fold) = mean(predict(modelDT, XTest) == yTest) * 100;

        modelLR  = fitclinear(XTrain, yTrain, 'Learner', 'logistic');
        accLR(fold) = mean(predict(modelLR, XTest) == yTest) * 100;
    end

    accuracyPairs(idx,:) = [mean(accSVM), mean(accKNN), mean(accDT), mean(accLR)];
end

accuracyPairs = round(accuracyPairs,1);

disp('Two-Feature Combinations Accuracy:');
disp('     SVM       KNN        DT        LR');
disp(accuracyPairs);

%% STEP 12: 3-Feature and 4-Feature Combination Classification

selectedFeatures = [1, 2, 6, 8];
featureNames = {'CAV', 'CWT', 'FFT', 'IQR', 'PeakDiff', 'Tp', 'STD', 'ZCR'};

combos3 = nchoosek(selectedFeatures, 3);
accuracy3 = zeros(size(combos3, 1), numClassifiers);

combo4 = selectedFeatures;

for idx = 1:size(combos3, 1)
    featureSet = combos3(idx,:);

    accSVM = zeros(5,1);
    accKNN = zeros(5,1);
    accDT  = zeros(5,1);
    accLR  = zeros(5,1);

    for fold = 1:5
        trainIdx = training(cv, fold);
        testIdx  = test(cv, fold);

        XTrain = X(trainIdx, featureSet);
        XTest  = X(testIdx, featureSet);
        yTrain = y(trainIdx);
        yTest  = y(testIdx);

        if length(unique(yTrain)) < 2
            continue;
        end

        modelSVM = fitcsvm(XTrain, yTrain, 'KernelFunction', 'rbf', 'KernelScale', 0.3, 'BoxConstraint', 1);
        accSVM(fold) = mean(predict(modelSVM, XTest) == yTest) * 100;

        modelKNN = fitcknn(XTrain, yTrain, 'NumNeighbors', 10, 'Distance', 'euclidean');
        accKNN(fold) = mean(predict(modelKNN, XTest) == yTest) * 100;

        modelDT = fitctree(XTrain, yTrain, 'MaxNumSplits', 4);
        accDT(fold) = mean(predict(modelDT, XTest) == yTest) * 100;

        modelLR = fitclinear(XTrain, yTrain, 'Learner', 'logistic');
        accLR(fold) = mean(predict(modelLR, XTest) == yTest) * 100;
    end

    accuracy3(idx,:) = [mean(accSVM), mean(accKNN), mean(accDT), mean(accLR)];
end

accSVM = zeros(5,1);
accKNN = zeros(5,1);
accDT  = zeros(5,1);
accLR  = zeros(5,1);

for fold = 1:5
    trainIdx = training(cv, fold);
    testIdx  = test(cv, fold);

    XTrain = X(trainIdx, combo4);
    XTest  = X(testIdx, combo4);
    yTrain = y(trainIdx);
    yTest  = y(testIdx);

    modelSVM = fitcsvm(XTrain, yTrain, 'KernelFunction', 'rbf', 'KernelScale', 0.3, 'BoxConstraint', 1);
    accSVM(fold) = mean(predict(modelSVM, XTest) == yTest) * 100;

    modelKNN = fitcknn(XTrain, yTrain, 'NumNeighbors', 10, 'Distance', 'euclidean');
    accKNN(fold) = mean(predict(modelKNN, XTest) == yTest) * 100;

    modelDT = fitctree(XTrain, yTrain, 'MaxNumSplits', 4);
    accDT(fold) = mean(predict(modelDT, XTest) == yTest) * 100;

    modelLR = fitclinear(XTrain, yTrain, 'Learner', 'logistic');
    accLR(fold) = mean(predict(modelLR, XTest) == yTest) * 100;
end

accuracy4 = [mean(accSVM), mean(accKNN), mean(accDT), mean(accLR)];

accuracy3 = round(accuracy3, 2);
accuracy4 = round(accuracy4, 2);

disp('Three-Feature Combinations Accuracy:');
disp('     SVM       KNN        DT        LR');
disp(accuracy3);

disp(' ');
disp('Accuracy for All 4 Selected Features:');
disp('     SVM       KNN        DT        LR');
disp(accuracy4);
%% STEP 13: 3D Scatter Plot

combo1 = [2, 6, 8];  % CWT, Tp, ZCR
combo2 = [1, 6, 8];  % CAV, Tp, ZCR

earthquakeIdx = labelsAll == 1;
humanIdx      = labelsAll == 0;

figure('Position', [400, 100, 1200, 600]);

subplot(1,2,1);
scatter3(X(earthquakeIdx, combo1(1)), X(earthquakeIdx, combo1(2)), X(earthquakeIdx, combo1(3)), 20, 'k', 'filled');
hold on;
scatter3(X(humanIdx, combo1(1)), X(humanIdx, combo1(2)), X(humanIdx, combo1(3)), 20, [0.5 0.5 0.5], 'filled');
xlabel('$\mathrm{CWT}$', 'Interpreter', 'latex', 'FontSize', 16);
ylabel('$T_p$', 'Interpreter', 'latex', 'FontSize', 16);
zlabel('$\mathrm{ZCR}$', 'Interpreter', 'latex', 'FontSize', 16);
legend({'Earthquake', 'Human Activity'}, 'Interpreter', 'latex', 'Location', 'best', 'FontSize', 14, 'Color', [0.94 0.94 0.94]);
grid on;
axis tight;

subplot(1,2,2);
scatter3(X(earthquakeIdx, combo2(1)), X(earthquakeIdx, combo2(2)), X(earthquakeIdx, combo2(3)), 20, 'k', 'filled');
hold on;
scatter3(X(humanIdx, combo2(1)), X(humanIdx, combo2(2)), X(humanIdx, combo2(3)), 20, [0.5 0.5 0.5], 'filled');
xlabel('$\mathrm{CAV}$', 'Interpreter', 'latex', 'FontSize', 16);
ylabel('$T_p$', 'Interpreter', 'latex', 'FontSize', 16);
zlabel('$\mathrm{ZCR}$', 'Interpreter', 'latex', 'FontSize', 16);
legend({'Earthquake', 'Human Activity'}, 'Interpreter', 'latex', 'Location', 'best', 'FontSize', 14, 'Color', [0.94 0.94 0.94]);
grid on;
axis tight;

%% STEP 14: Earthquake Recall

combosRecall = {[1,6,8], [2,6,8]};
comboNames   = {'CAV-Tp-ZCR', 'CWT-Tp-ZCR'};
classifierNames = {'SVM', 'KNN', 'DT', 'LR'};

recalls = zeros(length(combosRecall), numClassifiers);

for c = 1:length(combosRecall)
    featureSet = combosRecall{c};

    for clf = 1:numClassifiers
        recallFold = zeros(cv.NumTestSets,1);

        for fold = 1:cv.NumTestSets
            trainIdx = training(cv, fold);
            testIdx  = test(cv, fold);

            XTrain = featuresAll(trainIdx, featureSet);
            XTest  = featuresAll(testIdx, featureSet);
            yTrain = labelsAll(trainIdx);
            yTest  = labelsAll(testIdx);

            if clf == 1
                model = fitcsvm(XTrain, yTrain, 'KernelFunction', 'rbf', 'KernelScale', 0.3, 'BoxConstraint', 1);
            elseif clf == 2
                model = fitcknn(XTrain, yTrain, 'NumNeighbors', 10, 'Distance', 'euclidean');
            elseif clf == 3
                model = fitctree(XTrain, yTrain, 'MaxNumSplits', 4);
            elseif clf == 4
                model = fitclinear(XTrain, yTrain, 'Learner', 'logistic');
            end

            yPred = predict(model, XTest);

            TP = sum((yTest == 1) & (yPred == 1));
            FN = sum((yTest == 1) & (yPred == 0));
            recallFold(fold) = TP / (TP + FN);
        end

        recalls(c, clf) = mean(recallFold);
    end
end

for c = 1:2
    fprintf('Recall for combination %s:\n', comboNames{c});
    for clf = 1:4
        fprintf('%s\t\t%.2f\n', classifierNames{clf}, recalls(c, clf)*100);
    end
end

%% STEP 15: ROC Curve (for Best Feature Set)

selectedFeaturesROC = featuresAll(:, [2,6,8]);
yROC = labelsAll;

scoresAll = zeros(size(yROC,1), 1);
labelsAllPred = zeros(size(yROC,1), 1);

startIdx = 1;

for fold = 1:cv.NumTestSets
    trainIdx = training(cv, fold);
    testIdx  = test(cv, fold);

    XTrain = selectedFeaturesROC(trainIdx,:);
    XTest  = selectedFeaturesROC(testIdx,:);
    yTrain = yROC(trainIdx);
    yTest  = yROC(testIdx);

    modelSVM = fitcsvm(XTrain, yTrain, 'KernelFunction', 'rbf', 'KernelScale', 0.3, 'BoxConstraint', 1, ...
        'Standardize', true, 'ClassNames', [0,1]);
    modelSVM = fitPosterior(modelSVM);

    [~, score] = predict(modelSVM, XTest);

    foldSize = length(yTest);

    scoresAll(startIdx:startIdx+foldSize-1) = score(:,2);
    labelsAllPred(startIdx:startIdx+foldSize-1) = yTest;

    startIdx = startIdx + foldSize;
end

[fpRate, tpRate, T, AUC] = perfcurve(labelsAllPred, scoresAll, 1);

[~, bestIdx] = min(sqrt(fpRate.^2 + (1 - tpRate).^2));
bestPoint = [fpRate(bestIdx), tpRate(bestIdx)];

figure('Position', [400, 100, 1200, 900]);
hold on;
set(gca, 'FontName', 'Times New Roman', 'FontSize', 14);

plot(fpRate, tpRate, 'k-', 'LineWidth', 2);
area(fpRate, tpRate, 'FaceColor', [0.8 0.8 0.8], 'FaceAlpha', 0.5, 'EdgeColor', 'none');
plot([0 1], [0 1], 'k--');
plot(bestPoint(1), bestPoint(2), 'ko', 'MarkerFaceColor', [0.2 0.2 0.2], 'MarkerSize', 8);
text(bestPoint(1)+0.02, bestPoint(2), sprintf('(%.2f, %.2f)', bestPoint(1), bestPoint(2)), ...
    'FontSize', 14, 'Interpreter', 'latex', 'Color', 'black');

xlabel('$\mathrm{False\ Positive\ Rate}$', 'Interpreter', 'latex', 'FontSize', 18);
ylabel('$\mathrm{True\ Positive\ Rate}$', 'Interpreter', 'latex', 'FontSize', 18);

text(0.5, 0.5, sprintf('Positive class: 1\nAUC = %.2f', AUC), ...
    'HorizontalAlignment', 'center', 'FontSize', 16, 'Interpreter', 'latex');

legend({'Area Under Curve (AUC)', 'ROC Curve', 'Random Classifier', 'Best Point'}, ...
    'Location', 'SouthEast', 'Interpreter', 'latex', 'Color', [0.94 0.94 0.94], 'FontSize', 14);

grid on;
axis square;
xlim([0 1]);
ylim([0 1]);

%% Function: Feature Extraction & Tp Calculation

function features = extractFeatures(signal, fs)

    alpha   = 0.99;
    epsVal  = 1e-12;

    dt      = 1 / fs;
    CAV     = trapz(dt, abs(signal));

    wt      = cwt(signal, fs);
    avgCWT  = mean(abs(wt(:)));

    fftVals = abs(fft(signal));
    avgFFT  = mean(fftVals .^ 2);

    IQRval  = iqr(signal);

    peakDiff = max(signal) - min(signal);

    TpSeries = computeTpSeries(signal, fs, alpha, epsVal);
    Tp       = mean(TpSeries);

    stdVal  = std(signal);

    ZCR     = sum(abs(diff(sign(signal)))) / 2;

    features = [CAV, avgCWT, avgFFT, IQRval, peakDiff, Tp, stdVal, ZCR];
end

function TpSeries = computeTpSeries(acc, Fs, alpha, epsVal)

    t     = (1:length(acc))' / Fs;
    vel   = cumtrapz(t, acc);
    Vdot  = gradient(vel) * Fs;

    V     = zeros(length(acc),1);
    D     = zeros(length(acc),1);
    Tp    = zeros(length(acc),1);

    V(1)  = mean(vel(1:3).^2);
    D(1)  = mean(Vdot(1:3).^2);
    Tp(1) = 2*pi*sqrt(V(1)/(D(1)+epsVal));

    for i = 2:length(acc)
        V(i)  = alpha * V(i-1) + vel(i)^2;
        D(i)  = alpha * D(i-1) + Vdot(i)^2;
        Tp(i) = 2*pi*sqrt(V(i)/(D(i)+epsVal));
    end

    TpSeries = Tp(2:end);
end

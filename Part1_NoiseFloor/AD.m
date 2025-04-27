clc; 
clearvars;
close all;

%% Sampling & Analysis Settings
samplingRate   = 100;
dt             = 1 / samplingRate;
maxNumTau      = 100;
commonTauCount = 100;
simSamples     = 30000;

%% File & Test Configuration
dataFolder     = 'Records';
fileList       = dir(fullfile(dataFolder, 'iPhone8-test*.csv'));
numTests       = numel(fileList);

%% Compute Allan Variance (One axis)
tausAll        = cell(numTests, 1);
allanVarsAll   = cell(numTests, 1);

for testIndex = 1:numTests
    filename   = fullfile(dataFolder, fileList(testIndex).name);
    rawData    = readmatrix(filename);
    numSamples = size(rawData, 1);

    % Tau vector for this test
    maxTau         = 2^floor(log2(numSamples / 2));
    tauSteps       = unique(ceil(logspace(log10(1), log10(maxTau), maxNumTau).'));
    tauThisTest    = tauSteps * dt;

    % Signal: integrate acceleration to velocity (One axis)
    signal         = cumsum(rawData(:, 2)) * dt;
    allanVarLocal  = NaN(numel(tauSteps), 1);

    for i = 1:numel(tauSteps)
        m = tauSteps(i);
        n = numSamples - 2 * m;

        if n <= 0
            continue;
        end

        y1 = signal(1:n);
        y2 = signal(1+m:n+m);
        y3 = signal(1+2*m:n+2*m);

        diffs = y3 - 2*y2 + y1;
        allanVarLocal(i) = sum(diffs.^2) / (2 * m^2 * dt^2 * n);
    end

    tausAll{testIndex}      = tauThisTest;
    allanVarsAll{testIndex} = allanVarLocal;
end

%% Interpolation on Allan Variance
minTau        = min(cellfun(@(x) min(x), tausAll));
maxTau        = max(cellfun(@(x) max(x), tausAll));
tol           = 1;
commonTau     = logspace(log10(minTau), log10(maxTau - tol), commonTauCount);
allanVarsInterp = NaN(numTests, commonTauCount);

for testIndex = 1:numTests
    tau_current = tausAll{testIndex};
    var_current = allanVarsAll{testIndex};

    validIdx = (commonTau >= min(tau_current)) & (commonTau <= max(tau_current));
    allanVarsInterp(testIndex, validIdx) = interp1(tau_current, var_current, commonTau(validIdx), 'linear');
end

% Final Allan deviation using median Allan variance
medianAllanVar = median(allanVarsInterp, 1, 'omitnan');
allanDev       = sqrt(medianAllanVar);

%% Noise Parameter Identification (WN, RW, BI)
logTau       = log10(commonTau);
logAllanDev  = log10(allanDev);
slope        = diff(logAllanDev) ./ diff(logTau);

% White Noise (slope = -0.5)
slopeWN      = -0.5;
idxWN        = findClosestSlope(slope, slopeWN);
interceptWN  = logAllanDev(idxWN) - slopeWN * logTau(idxWN);
whiteNoise   = 10^(slopeWN * log10(1) + interceptWN);
WNLine       = whiteNoise ./ sqrt(commonTau);
tauWN        = 1;

% Random Walk (slope = +0.5)
slopeRW      = 0.5;
idxRW        = findClosestSlope(slope, slopeRW);
interceptRW  = logAllanDev(idxRW) - slopeRW * logTau(idxRW);
randomWalk   = 10^(slopeRW * log10(3) + interceptRW);
RWLine       = randomWalk .* sqrt(commonTau / 3);
tauRW        = 3;

% Bias Instability (slope = 0)
idxBI        = findClosestSlope(slope, 0);
adevAtBI     = allanDev(idxBI);
biasInstabilityFactor = sqrt(2 * log(2) / pi);  % ≈ 0.664
biasInstability = adevAtBI / biasInstabilityFactor;
tauBI        = commonTau(idxBI);
adevBI       = adevAtBI;
biLine       = biasInstability * biasInstabilityFactor * ones(size(commonTau));

%% Plot: Allan Deviation and Noise Lines
figure;
loglog(commonTau, allanDev, 'r', 'LineWidth', 2); hold on;

WNColor = [0.2 0.2 0.2];
RWColor = [0.5 0.5 0.5];
BIColor = [0.7 0.7 0.7];

loglog(commonTau, WNLine, '--', 'Color', WNColor, 'LineWidth', 1.5);
loglog(commonTau, RWLine, ':', 'Color', RWColor, 'LineWidth', 1.5);
loglog(commonTau, biLine,  '-.', 'Color', BIColor, 'LineWidth', 1.5);

loglog([tauWN, tauRW, tauBI], [whiteNoise, randomWalk, adevBI], ...
       'ko', 'MarkerFaceColor', 'yellow');

xlabel('Averaging Time (s)');
ylabel('Allan Deviation (m/s)');
title('Allan Deviation & Noise Characteristics');
legend('Allan Deviation','White Noise','Random Walk','Bias Instability','Noise IDs','Location','best');
grid on;
xlim([1e-2, 1e+2]);
ylim([1e-5, 1e-1]);

text(tauWN * 1.05, whiteNoise * 1.2, 'WN', 'Color', [0.1 0.1 0.1], 'FontSize', 11, 'FontWeight', 'bold');
text(tauRW * 1.05, randomWalk * 1.2, 'RW', 'Color', [0.1 0.1 0.1], 'FontSize', 11, 'FontWeight', 'bold');
text(tauBI * 1.05, adevBI * 1.2, '0.664BI', 'Color', [0.1 0.1 0.1], 'FontSize', 11, 'FontWeight', 'bold');

%% Simulated Accelerometer Using Identified Parameters
params = accelparams( ...
    'AxesMisalignment', 0, ...
    'NoiseDensity', whiteNoise, ...
    'RandomWalk', randomWalk, ...
    'BiasInstability', biasInstability);

imuModel = imuSensor('SampleRate', samplingRate, 'Accelerometer', params);
orientation = quaternion.ones(simSamples, 1);
zeroAcc     = zeros(simSamples, 3);
zeroAngVel  = zeros(simSamples, 3);
simAccel    = imuModel(zeroAcc, zeroAngVel, orientation);

dataSim     = simAccel(:, 2);  % Simulate One axis
signalSim   = cumsum(dataSim) * dt;
mVals       = unique(round(commonTau * samplingRate));
N           = length(signalSim);

simAllanVar = NaN(size(mVals));
for i = 1:length(mVals)
    m = mVals(i);
    n = N - 2 * m;
    if n <= 0
        continue;
    end

    y1 = signalSim(1:n);
    y2 = signalSim(1+m:n+m);
    y3 = signalSim(1+2*m:n+2*m);

    diffs = y3 - 2*y2 + y1;
    simAllanVar(i) = sum(diffs.^2) / (2 * m^2 * dt^2 * n);
end

simAllanDev = sqrt(simAllanVar);
simTau      = mVals / samplingRate;
simAllanVarInterp = interp1(simTau, simAllanVar, commonTau, 'linear', 'extrap');
simAllanDevInterp = sqrt(simAllanVarInterp);

%% Plot: Real vs Simulated Comparison
figure;

realColor = [0 0 0];
simColor  = [1 0 0];

loglog(commonTau, allanDev, '-', 'Color', realColor, 'LineWidth', 2, 'DisplayName', 'Real'); hold on;
loglog(commonTau, simAllanDevInterp, '--', 'Color', simColor, 'LineWidth', 2, 'DisplayName', 'Simulated');

xlabel('Averaging Time (s)');
ylabel('Allan Deviation (m/s)');
title('Comparison of Allan Deviation: Real vs. Simulated');
legend('show', 'Location', 'best');
grid on;
xlim([1e-2, 1e+2]);
ylim([1e-5, 1e-1]);

%% Function: Find Closest Slope Match
function idx = findClosestSlope(slopeVec, targetSlope)
    smoothed = movmean(slopeVec, 10);
    [~, idx] = min(abs(smoothed - targetSlope));
end

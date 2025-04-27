clc;
clearvars;
close all;

%% File & Test Configuration
dataFolder = 'Records';
fileName   = fullfile(dataFolder, 'earthquake-test1.csv');
rawData    = readmatrix(fileName);

% Sampling
Fs = 100;
Ts = 1/Fs;

% Acceleration
refAcc    = rawData(:,1) - mean(rawData(:,1));
phone1Acc = rawData(:,2) - mean(rawData(:,2));
phone2Acc = rawData(:,3) - mean(rawData(:,3));

%% P-phase Picking
pickerType        = 'na';
plotFlag          = 'N';
sampleInterval    = Ts;
oscillatorPeriod  = 0.01;
dampingRatio      = 0.6;
histBins          = 50;
segmentOption     = 'to_peak';

[pickLocRef, snrRef]       = PphasePicker(refAcc, sampleInterval, pickerType, plotFlag, oscillatorPeriod, dampingRatio, histBins, segmentOption);
[pickLocPhone1, snrPhone1] = PphasePicker(phone1Acc, sampleInterval, pickerType, plotFlag, oscillatorPeriod, dampingRatio, histBins, segmentOption);
[pickLocPhone2, snrPhone2] = PphasePicker(phone2Acc, sampleInterval, pickerType, plotFlag, oscillatorPeriod, dampingRatio, histBins, segmentOption);

% P-pick Time Differences
diffPhone1 = abs(pickLocPhone1 - pickLocRef);
diffPhone2 = abs(pickLocPhone2 - pickLocRef);
avgDiff    = mean([diffPhone1, diffPhone2]);

fprintf('\nAbsolute Difference in P-wave Arrival Times:\n');
fprintf('  Phone 1 - Reference: %.3f s\n', diffPhone1);
fprintf('  Phone 2 - Reference: %.3f s\n', diffPhone2);
fprintf('  Average Difference:  %.3f s\n', avgDiff);

%% Tp Calculation
offsetAfterPick_sec = 2;
windowLength_sec    = 4;
alpha = 0.99;
epsVal = 1e-12;

[TpRef, tRef, precisionRef, indexRef, segStartRef, segStopRef] = computeTpSeries(refAcc, pickLocRef, Fs, alpha, epsVal, offsetAfterPick_sec, windowLength_sec);
[TpPhone1, t1, precisionPhone1, indexPhone1, segStart1, segStop1] = computeTpSeries(phone1Acc, pickLocPhone1, Fs, alpha, epsVal, offsetAfterPick_sec, windowLength_sec);
[TpPhone2, t2, precisionPhone2, indexPhone2, segStart2, segStop2] = computeTpSeries(phone2Acc, pickLocPhone2, Fs, alpha, epsVal, offsetAfterPick_sec, windowLength_sec);

fprintf('\nEstimated Tp Values:\n');
fprintf('  Reference: %.4f s at %.2f s\n', precisionRef, tRef(indexRef));
fprintf('  Phone 1:   %.4f s at %.2f s\n', precisionPhone1, t1(indexPhone1));
fprintf('  Phone 2:   %.4f s at %.2f s\n', precisionPhone2, t2(indexPhone2));

% Log(Tp) Differences
logTpRef    = log(precisionRef);
logTpPhone1 = log(precisionPhone1);
logTpPhone2 = log(precisionPhone2);

logDiffPhone1 = abs(logTpPhone1 - logTpRef);
logDiffPhone2 = abs(logTpPhone2 - logTpRef);
avgLogDiff    = mean([logDiffPhone1, logDiffPhone2]);

fprintf('\nAbsolute Difference in log(Tp):\n');
fprintf('  Phone 1 - Reference: %.3f\n', logDiffPhone1);
fprintf('  Phone 2 - Reference: %.3f\n', logDiffPhone2);
fprintf('  Average log(Tp) Difference: %.3f\n', avgLogDiff);

%% Plot: Tp
figure('Position', [400 100 1200 900]);
set(gcf, 'Color', [0.94 0.94 0.94]);
tiledlayout(3, 1, 'TileSpacing', 'compact', 'Padding', 'compact');

nexttile;
plot(tRef, TpRef, 'Color', [0.3 0.3 0.3], 'LineWidth', 1.2); hold on;
xline(pickLocRef, 'Color', [0 0 0], 'Label', 'P-pick', 'FontSize', 12, 'FontWeight', 'bold', 'LineWidth', 1.5);
xline(tRef(segStartRef), 'Color', [0 0 0], 'LineStyle', '--', 'Label', 'Start', 'FontSize', 12, 'FontWeight', 'bold', 'LineWidth', 1.5);
xline(tRef(segStopRef), 'Color', [0 0 0], 'Label', 'End', 'FontSize', 12, 'FontWeight', 'bold', 'LineWidth', 1.5);
plot(tRef(indexRef), precisionRef, 'o', 'Color', [0 0 0], 'MarkerFaceColor', [0 0 0], 'MarkerSize', 8);
legend({'Reference Accelerometer'}, 'Interpreter', 'latex', 'Location', 'best', 'FontSize', 18, 'Color', [0.94 0.94 0.94]);
ylabel('$T_p$ (s)', 'Interpreter', 'latex', 'FontSize', 18, 'FontWeight', 'bold');
set(gca, 'TickLabelInterpreter', 'latex', 'FontSize', 16, 'LineWidth', 1.2); grid on;

nexttile;
plot(t1, TpPhone1, 'Color', [0.3 0.3 0.3], 'LineWidth', 1.2); hold on;
xline(pickLocPhone1, 'Color', [0 0 0], 'Label', 'P-pick', 'FontSize', 12, 'FontWeight', 'bold', 'LineWidth', 1.5);
xline(t1(segStart1), 'Color', [0 0 0], 'LineStyle', '--', 'Label', 'Start', 'FontSize', 12, 'FontWeight', 'bold', 'LineWidth', 1.5);
xline(t1(segStop1), 'Color', [0 0 0], 'Label', 'End', 'FontSize', 12, 'FontWeight', 'bold', 'LineWidth', 1.5);
plot(t1(indexPhone1), precisionPhone1, 'o', 'Color', [0 0 0], 'MarkerFaceColor', [0 0 0], 'MarkerSize', 8);
legend({'iPhone 6 No.1'}, 'Interpreter', 'latex', 'Location', 'best', 'FontSize', 18, 'Color', [0.94 0.94 0.94]);
ylabel('$T_p$ (s)', 'Interpreter', 'latex', 'FontSize', 18, 'FontWeight', 'bold');
set(gca, 'TickLabelInterpreter', 'latex', 'FontSize', 16, 'LineWidth', 1.2); grid on;

nexttile;
plot(t2, TpPhone2, 'Color', [0.3 0.3 0.3], 'LineWidth', 1.2); hold on;
xline(pickLocPhone2, 'Color', [0 0 0], 'Label', 'P-pick', 'FontSize', 12, 'FontWeight', 'bold', 'LineWidth', 1.5);
xline(t2(segStart2), 'Color', [0 0 0], 'LineStyle', '--', 'Label', 'Start', 'FontSize', 12, 'FontWeight', 'bold', 'LineWidth', 1.5);
xline(t2(segStop2), 'Color', [0 0 0], 'Label', 'End', 'FontSize', 12, 'FontWeight', 'bold', 'LineWidth', 1.5);
plot(t2(indexPhone2), precisionPhone2, 'o', 'Color', [0 0 0], 'MarkerFaceColor', [0 0 0], 'MarkerSize', 8);
legend({'iPhone 6 No.2'}, 'Interpreter', 'latex', 'Location', 'best', 'FontSize', 18, 'Color', [0.94 0.94 0.94]);
xlabel('Time (s)', 'Interpreter', 'latex', 'FontSize', 18, 'FontWeight', 'bold');
ylabel('$T_p$ (s)', 'Interpreter', 'latex', 'FontSize', 18, 'FontWeight', 'bold');
set(gca, 'TickLabelInterpreter', 'latex', 'FontSize', 16, 'LineWidth', 1.2); grid on;

%% B-Δ & C-Δ Fitting
meanPickLoc = mean([pickLocRef, pickLocPhone1, pickLocPhone2]);
[tRefEnv, absAccRef, envRef, fittedRef, linearRef, B_ref, C_ref] = computeEnvelopeFits(refAcc, meanPickLoc, Fs);
[t1Env, absAcc1, env1, fitted1, linear1, B_1, C_1]               = computeEnvelopeFits(phone1Acc, meanPickLoc, Fs);
[t2Env, absAcc2, env2, fitted2, linear2, B_2, C_2]               = computeEnvelopeFits(phone2Acc, meanPickLoc, Fs);

% Differences in log(B) and log(C)
logDiff_B1 = abs(log(B_1) - log(B_ref));
logDiff_B2 = abs(log(B_2) - log(B_ref));
avgLogDiff_B = mean([logDiff_B1, logDiff_B2]);

logDiff_C1 = abs(log(C_1) - log(C_ref));
logDiff_C2 = abs(log(C_2) - log(C_ref));
avgLogDiff_C = mean([logDiff_C1, logDiff_C2]);

fprintf('\nAbsolute Differences in log(B) Parameter:\n');
fprintf('  Phone 1 - Reference: %.3f\n', logDiff_B1);
fprintf('  Phone 2 - Reference: %.3f\n', logDiff_B2);
fprintf('  Average log(B) Difference: %.3f\n', avgLogDiff_B);

fprintf('\nAbsolute Differences in log(C) Parameter:\n');
fprintf('  Phone 1 - Reference: %.3f\n', logDiff_C1);
fprintf('  Phone 2 - Reference: %.3f\n', logDiff_C2);
fprintf('  Average log(C) Difference: %.3f\n', avgLogDiff_C);

%% Plot: B-Δ and C-Δ
figure('Position', [400 100 1200 900]);
set(gcf, 'Color', [0.94 0.94 0.94]);
tiledlayout(2, 1, 'TileSpacing', 'compact', 'Padding', 'compact');

% B-Δ
nexttile;
plot(tRefEnv, absAccRef, 'Color', [0.6 0.6 0.6], 'DisplayName', 'Absolute Value'); hold on;
plot(tRefEnv, envRef, 'Color', [0.3 0.3 0.3], 'LineWidth', 1.5, 'DisplayName', 'Envelope');
plot(tRefEnv, fittedRef, '--', 'Color', [0.1 0.1 0.1], 'LineWidth', 1.5, ...
    'DisplayName', 'Fitted Curve ($B \cdot t \cdot e^{-At}$)');
set(gca, 'YScale', 'log', 'TickLabelInterpreter', 'latex', 'FontSize', 14, 'LineWidth', 1.2);
ylabel('Amplitude (m/s$^2$)', 'Interpreter', 'latex', 'FontSize', 16, 'FontWeight', 'bold');
legend('Interpreter', 'latex', 'Location', 'best', 'FontSize', 16, 'Color', [0.94 0.94 0.94]);
grid on; xlim([0 3]);

% C-Δ
nexttile;
tShort = tRefEnv(~isnan(linearRef));
plot(tShort, absAccRef(1:length(tShort)), 'Color', [0.6 0.6 0.6], 'DisplayName', 'Absolute Value'); hold on;
plot(tShort, envRef(1:length(tShort)), 'Color', [0.3 0.3 0.3], 'LineWidth', 1.5, 'DisplayName', 'Envelope');
plot(tShort, linearRef(1:length(tShort)), '--', 'Color', [0.1 0.1 0.1], 'LineWidth', 1.5, ...
    'DisplayName', 'Fitted Curve ($C \cdot t$)');
set(gca, 'YScale', 'log', 'TickLabelInterpreter', 'latex', 'FontSize', 14, 'LineWidth', 1.2);
xlabel('Time (s)', 'Interpreter', 'latex', 'FontSize', 16, 'FontWeight', 'bold');
ylabel('Amplitude (m/s$^2$)', 'Interpreter', 'latex', 'FontSize', 16, 'FontWeight', 'bold');
legend('Interpreter', 'latex', 'Location', 'best', 'FontSize', 16, 'Color', [0.94 0.94 0.94]);
grid on; xlim([0 0.5]);

%% Function: Tp Calculation
function [Tp, t, precision, index, segmentStart, segmentStop] = computeTpSeries(acc, pickLoc, Fs, alpha, epsVal, offsetAfterPick_sec, windowLength_sec)
    t     = (1:length(acc))' / Fs;
    vel   = cumtrapz(t, acc);
    Vdot  = gradient(vel) * Fs;
    V     = zeros(length(acc), 1);
    D     = zeros(length(acc), 1);
    Tp    = zeros(length(acc), 1);
    V(1)  = mean(vel(1:3).^2);
    D(1)  = mean(Vdot(1:3).^2);
    Tp(1) = 2 * pi * sqrt(V(1) / (D(1) + epsVal));
    for i = 2:length(acc)
        V(i)  = alpha * V(i - 1) + vel(i)^2;
        D(i)  = alpha * D(i - 1) + Vdot(i)^2;
        Tp(i) = 2 * pi * sqrt(V(i) / (D(i) + epsVal));
    end
    Tp           = Tp(2:end);
    t            = t(2:end);
    start        = round(pickLoc * Fs);
    segmentStart = start + round(offsetAfterPick_sec * Fs);
    segmentStop  = start + round(windowLength_sec * Fs);
    segmentStop  = min(segmentStop, length(Tp));
    [precision, index] = max(Tp(segmentStart:segmentStop));
    index              = index + segmentStart - 1;
end

%% Function: Fitting B-Δ and C-Δ
function [t, absAcc, env, fittedCurve, linearFit, B, C] = computeEnvelopeFits(acc, pickLoc, Fs)
    Ts = 1 / Fs;
    startS = round(pickLoc * Fs);
    stopS  = startS + round(3 * Fs);
    accSeg = acc(startS:stopS);
    t      = (0:length(accSeg)-1)' * Ts;
    absAcc = abs(accSeg);

    noiseStd = std(accSeg);
    epsilon  = 0.01 * noiseStd;
    absAcc(absAcc < epsilon) = epsilon;

    env = zeros(size(absAcc));
    env(1) = absAcc(1);
    for i = 2:length(absAcc)
        env(i) = max(env(i-1), absAcc(i));
    end

    % C-Δ
    tShort   = t(t <= 0.5);
    envShort = env(t <= 0.5);
    C        = sum(tShort .* envShort) / sum(tShort.^2);
    linearFit = C * tShort;
    linearFit = [linearFit; nan(length(t) - length(tShort), 1)];

    % B-Δ
    modelFun   = @(params, t) params(2) * t .* exp(-params(1) * t);
    initParams = [1, max(env)];
    lb = [-Inf, 0]; ub = [Inf, Inf];
    opts = optimoptions('lsqcurvefit','Display','off');
    paramsOpt = lsqcurvefit(modelFun, initParams, t, env, lb, ub, opts);
    A = paramsOpt(1); B = paramsOpt(2);
    fittedCurve = B * t .* exp(-A * t);

    env(env < epsilon) = epsilon;
    fittedCurve(fittedCurve < epsilon) = epsilon;
    linearFit(linearFit < epsilon) = epsilon;
end
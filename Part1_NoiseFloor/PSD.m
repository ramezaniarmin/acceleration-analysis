clc;
clearvars;
close all;

%% Sampling & Analysis Settings
samplingRate   = 100;
fixedDuration  = 40;
segmentLength  = fixedDuration * samplingRate;
window         = hann(segmentLength);
overlap        = round(0.5 * segmentLength);
nfft           = segmentLength;

%% File & Test Configuration
dataFolder     = 'Records';
fileList       = dir(fullfile(dataFolder, 'iPhoneX-test*.csv'));
numTests       = numel(fileList);

%% Data Processing & PSD Computation
psdAllTests    = nan(nfft/2 + 1, 3, numTests);
freq           = [];

for testIndex = 1:numTests
    filename   = fullfile(dataFolder, fileList(testIndex).name);
    rawData    = readmatrix(filename);
    
    % Ensure the file has the expected 3-axis
    if size(rawData, 2) < 3
        warning('File "%s" does not have at least 3 columns.', filename);
        continue;
    end
    
    % Process each axis
    for axisIdx = 1:3
        acc = rawData(:, axisIdx);
        acc = acc - mean(acc);
        
        if length(acc) < segmentLength
            warning('Test %d, axis %d: Not enough samples (%d < %d)', testIndex, axisIdx, length(acc), segmentLength);
            psd = nan(nfft/2 + 1, 1);
        else
            try
                [psd, f] = pwelch(acc, window, overlap, nfft, samplingRate);
                if isempty(freq)
                    freq = f;
                elseif ~isequal(freq, f)
                    warning('Frequency vector mismatch in Test %d, axis %d.', testIndex, axisIdx);
                end
            catch ME_pwelch
                warning('Test %d, axis %d: pwelch failed: %s', testIndex, axisIdx, ME_pwelch.message);
                psd = nan(nfft/2 + 1, 1);
            end
        end
        
        psdAllTests(:, axisIdx, testIndex) = psd;
    end
end

if isempty(freq)
    error('No valid frequency vector obtained.');
end

%% Aggregate PSD Results
medianPsdX = median(psdAllTests(:, 1, :), 3, 'omitnan');
medianPsdY = median(psdAllTests(:, 2, :), 3, 'omitnan');
medianPsdZ = median(psdAllTests(:, 3, :), 3, 'omitnan');
medianPsdAcrossAxes = mean([medianPsdX, medianPsdY, medianPsdZ], 2, 'omitnan');

if all(isnan(medianPsdAcrossAxes))
    error('Median PSD is all NaNs.');
end

%% Outlier Removal and Smoothing
% Apply Hampel filter to remove outliers from the median PSD.
% This prepares the data for effective smoothing.
hampelWindow    = 20;
hampelThreshold = 3;
HampelPsd       = hampel(medianPsdAcrossAxes, hampelWindow, hampelThreshold);

% Apply Savitzky-Golay filter for broad trend extraction.
% The use of a large window is intentional to average over fine fluctuations,
% yielding a smooth single-line representation of the PSD.
sgolayOrder  = 1;
sgolayWindow = 1001;
SmoothedPsd  = sgolayfilt(HampelPsd, sgolayOrder, sgolayWindow);

% Optional: Convert to log scale and smooth again for further refinement.
logPsd         = 10 * log10(SmoothedPsd);
SmoothedLogPsd = sgolayfilt(logPsd, 1, 1001);

%% Plot: PSDs Comparision
% Figure 1: Compare raw, Hampel-filtered, and smoothed PSDs
figure(1)
semilogx(freq, medianPsdAcrossAxes, 'Color', [0.7 0.7 0.7], 'DisplayName', 'Raw PSD');
hold on;
semilogx(freq, HampelPsd, 'b', 'DisplayName', 'Hampel Filtered PSD');
semilogx(freq, SmoothedPsd, 'r', 'LineWidth', 1, 'DisplayName', 'Smoothed PSD');
hold off;
title('PSD Across Tests (Median-Based)');
xlabel('Frequency (Hz)');
ylabel('Amplitude (dB)');
legend('Location', 'best');
grid on;

% Figure 2: Compare Log PSD and Smoothed Log PSD
figure(2)
semilogx(freq, logPsd, 'b--', 'DisplayName', 'Log PSD');
hold on;
semilogx(freq, SmoothedLogPsd, 'k', 'LineWidth', 1, 'DisplayName', 'Smoothed Log PSD');
hold off;
xlim([0.01 100]);
ylim([-70 -50]);
title('Log PSD Comparison');
xlabel('Frequency (Hz)');
ylabel('Amplitude (dB)');
legend('Location', 'best');
grid on;

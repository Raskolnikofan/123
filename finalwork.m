function main
    hFig = figure('Name', '图像处理工具', 'NumberTitle', 'off', 'MenuBar', 'none', ...
        'Position', [100, 100, 800, 600]);

    uicontrol('Style', 'pushbutton', 'String', '打开图像', ...
        'Position', [20, 550, 100, 30], 'Callback', @openImage);

    uicontrol('Style', 'pushbutton', 'String', '灰度直方图', ...
        'Position', [140, 550, 100, 30], 'Callback', @showHistogram);

    uicontrol('Style', 'pushbutton', 'String', '直方图均衡化', ...
        'Position', [260, 550, 100, 30], 'Callback', @histEqualization);

    uicontrol('Style', 'pushbutton', 'String', '直方图匹配', ...
        'Position', [380, 550, 100, 30], 'Callback', @histMatching);

    uicontrol('Style', 'pushbutton', 'String', '对比度增强', ...
        'Position', [500, 550, 100, 30], 'Callback', @contrastEnhancement);

    uicontrol('Style', 'pushbutton', 'String', '几何变换', ...
        'Position', [620, 550, 100, 30], 'Callback', @geometricTransform);

    uicontrol('Style', 'pushbutton', 'String', '加噪与滤波', ...
        'Position', [20, 500, 100, 30], 'Callback', @noiseAndFilter);

    uicontrol('Style', 'pushbutton', 'String', '边缘提取', ...
        'Position', [140, 500, 100, 30], 'Callback', @edgeDetection);

    uicontrol('Style', 'pushbutton', 'String', '目标提取', ...
        'Position', [260, 500, 100, 30], 'Callback', @objectExtraction);

    uicontrol('Style', 'pushbutton', 'String', '特征提取', ...
        'Position', [380, 500, 100, 30], 'Callback', @featureExtraction);

    global img;
    img = [];
end

function openImage(~, ~)
    global img;
    [file, path] = uigetfile({'*.jpg;*.png;*.bmp', 'Image Files (*.jpg, *.png, *.bmp)'});
    if isequal(file, 0)
        return;
    end
    img = imread(fullfile(path, file));
    figure, imshow(img), title('原始图像');
end

function showHistogram(~, ~)
    global img;
    if isempty(img)
        errordlg('请先打开图像！', '错误');
        return;
    end
    grayImg = rgb2gray(img);
    figure, imhist(grayImg), title('灰度直方图');
end

function histEqualization(~, ~)
    global img;
    if isempty(img)
        errordlg('请先打开图像！', '错误');
        return;
    end
    grayImg = rgb2gray(img);
    eqImg = histeq(grayImg);
    figure, imshow(eqImg), title('直方图均衡化');
end

function histMatching(~, ~)
    global img;
    if isempty(img)
        errordlg('请先打开图像！', '错误');
        return;
    end
    [file, path] = uigetfile({'*.jpg;*.png;*.bmp', 'Image Files (*.jpg, *.png, *.bmp)'});
    if isequal(file, 0)
        return;
    end
    targetImg = imread(fullfile(path, file));
    grayImg = rgb2gray(img);
    targetGray = rgb2gray(targetImg);
    matchedImg = imhistmatch(grayImg, targetGray);
    figure, imshow(matchedImg), title('直方图匹配');
end

function contrastEnhancement(~, ~)
    global img;
    if isempty(img)
        errordlg('请先打开图像！', '错误');
        return;
    end
    grayImg = rgb2gray(img);
    linearImg = imadjust(grayImg);
    logImg = im2double(grayImg);
    logImg = log(1 + logImg);
    expImg = im2double(grayImg);
    expImg = exp(expImg) - 1;
    figure, 
    subplot(1, 3, 1), imshow(linearImg), title('线性变换');
    subplot(1, 3, 2), imshow(logImg, []), title('对数变换');
    subplot(1, 3, 3), imshow(expImg, []), title('指数变换');
end

function geometricTransform(~, ~)
    global img;
    if isempty(img)
        errordlg('请先打开图像！', '错误');
        return;
    end
    scaledImg = imresize(img, 0.5);
    rotatedImg = imrotate(img, 45);
    figure, 
    subplot(1, 2, 1), imshow(scaledImg), title('缩放变换');
    subplot(1, 2, 2), imshow(rotatedImg), title('旋转变换');
end

function noiseAndFilter(~, ~)
    global img;
    if isempty(img)
        errordlg('请先打开图像！', '错误');
        return;
    end
    prompt = {'噪声类型（gaussian, salt & pepper, uniform）：', ...
              '噪声参数1（均值或密度或强度范围下限）：', ...
              '噪声参数2（方差或强度范围上限，可为空）：'};
    dlgTitle = '输入噪声参数';
    numLines = 1;
    defaultAns = {'gaussian', '0', '0.01'};
    answer = inputdlg(prompt, dlgTitle, numLines, defaultAns);

    if isempty(answer)
        return;
    end

    noiseType = lower(answer{1});
    param1 = str2double(answer{2});
    param2 = str2double(answer{3});

    switch noiseType
        case 'gaussian'
            if isnan(param1) || isnan(param2)
                errordlg('请正确输入高斯噪声的均值和方差！', '错误');
                return;
            end
            noisyImg = imnoise(img, 'gaussian', param1, param2); % 均值param1，方差param2

        case 'salt & pepper'
            if isnan(param1)
                errordlg('请正确输入椒盐噪声的密度！', '错误');
                return;
            end
            noisyImg = imnoise(img, 'salt & pepper', param1); % 噪声密度param1

        case 'uniform'
            if isnan(param1) || isnan(param2)
                errordlg('请正确输入均匀噪声的范围下限和上限！', '错误');
                return;
            end
            noisyImg = im2double(img) + (rand(size(img)) * (param2 - param1) + param1);
            noisyImg = im2uint8(mat2gray(noisyImg)); % 调整到有效像素范围

        otherwise
            errordlg('未知噪声类型！请重新输入。', '错误');
            return;
    end
    spatialFiltered = medfilt2(rgb2gray(noisyImg));

    freqFiltered = freqFilter(noisyImg);

    figure, 
    subplot(1, 3, 1), imshow(noisyImg), title(['加噪图像 (' noiseType ')']);
    subplot(1, 3, 2), imshow(spatialFiltered), title('空域滤波（中值滤波）');
    subplot(1, 3, 3), imshow(freqFiltered, []), title('频域滤波');
end

function filteredImg = freqFilter(img)
    grayImg = rgb2gray(img);

    F = fft2(double(grayImg));
    Fshift = fftshift(F);
    [M, N] = size(grayImg);
    D0 = 30;
    [U, V] = meshgrid(1:N, 1:M);
    D = sqrt((U - N/2).^2 + (V - M/2).^2);
    H = double(D <= D0);

    G = Fshift .* H;

    G = ifftshift(G);
    filteredImg = real(ifft2(G));
end




function edgeDetection(~, ~)
    global img;
    if isempty(img)
        errordlg('请先打开图像！', '错误');
        return;
    end
    grayImg = rgb2gray(img);
    robertEdge = edge(grayImg, 'roberts');
    prewittEdge = edge(grayImg, 'prewitt');
    sobelEdge = edge(grayImg, 'sobel');
    laplacianEdge = edge(grayImg, 'log');
    figure, 
    subplot(2, 2, 1), imshow(robertEdge), title('Roberts 算子');
    subplot(2, 2, 2), imshow(prewittEdge), title('Prewitt 算子');
    subplot(2, 2, 3), imshow(sobelEdge), title('Sobel 算子');
    subplot(2, 2, 4), imshow(laplacianEdge), title('拉普拉斯算子');
end

function objectExtraction(~, ~)
    global img;
    if isempty(img)
        errordlg('请先打开图像！', '错误');
        return;
    end
    grayImg = rgb2gray(img);
    bwImg = imbinarize(grayImg);
    figure, 
    subplot(1, 2, 1), imshow(grayImg), title('灰度图');
    subplot(1, 2, 2), imshow(bwImg), title('二值化提取目标');
end

function featureExtractionCallback(~, ~)
    try
        img = evalin('base', 'img');
        grayImg = rgb2gray(img); 

        if size(grayImg, 1) < 32 || size(grayImg, 2) < 32
            errordlg('图像尺寸过小，请选择至少 32x32 的图像进行特征提取。', '错误');
            return;
        end

        lbpFeatures = extractLBPFeatures(grayImg);

        [hogFeatures, visualization] = extractHOGFeatures(grayImg);

        figure;
        subplot(1, 2, 1); bar(lbpFeatures); title('LBP 特征'); 
        subplot(1, 2, 2); plot(visualization); title('HOG 特征'); 
    catch ME
        errordlg(['特征提取失败：', ME.message], '错误');
    end
end


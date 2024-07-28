clear;
clc;
close all;

s1 = importdata("trainset.txt");
s2 = importdata("testset.txt");

%% 提取原始数据： 取偶数行，去除数据编号,有时候不需要
datatrain = s1(2:2:end,:);
datatest = s2(2:2:end,:);
datatrainp = datatrain(1:2750,:);
datatrainn = datatrain(2751:end,:);

% 无打乱顺序选取
% 0.9
% datatrainp = datatrain(1:2475,:);
% datatrainn = datatrain(2751:5225,:);
% 0.8
% datatrainp = datatrain(1:2200,:);
% datatrainn = datatrain(2751:4950,:);
% 0.7
% datatrainp = datatrain(1:1925,:);
% datatrainn = datatrain(2751:4675,:);
% datatrain = [datatrainp;datatrainn];

datatestp = datatest(1:750,:);
datatestn = datatest(751:end,:);
numTrain = size(datatrainp,1) + size(datatrainn,1);
numTest = size(datatest,1);
lenthSeq = size(datatest{1},2)-1;
% 自编码
pkCode_Train = pkCode(datatrain);
pkCode_Test = pkCode(datatest);
% PSDAAP编码
[PSDAAPtrainp, PSDAAPtrainn, PSDAAPtestp, PSDAAPtestn] = PSDAAP(datatrainp,datatrainn,datatestp,datatestn);
PSDAAPtrain = [PSDAAPtrainp;PSDAAPtrainn];
PSDAAPtrain = [PSDAAPtrain,zeros(numTrain,1)]';
PSDAAPtest = [PSDAAPtestp;PSDAAPtestn];
PSDAAPtest = [PSDAAPtest,zeros(numTest,1)]';
PSDAAPtrain = reshape(PSDAAPtrain,[lenthSeq 1 numTrain]);
PSDAAPtest = reshape(PSDAAPtest,[lenthSeq 1 numTest]);
%AAC编码
[AACtrainp,AACtrainn] = AAC(datatrainp,datatrainn);
[AACtestp,AACtestn] = AAC(datatestp,datatestn);
AACtrain = [AACtrainp;AACtrainn];
AACtrain = [AACtrain,zeros(numTrain,9)]';
AACtest = [AACtestp;AACtestn];
AACtest = [AACtest,zeros(numTest,9)]';
AACtrain = reshape(AACtrain,[lenthSeq 1 numTrain]);
AACtest = reshape(AACtest,[lenthSeq 1 numTest]);
% PWAA编码
[PWAAtrainp,PWAAtrainn] = PWAA(datatrainp,datatrainn);
[PWAAtestp,PWAAtestn] = PWAA(datatestp,datatestn);
PWAAtrain = [PWAAtrainp;PWAAtrainn];
PWAAtrain = [PWAAtrain,zeros(numTrain,9)]';
PWAAtest = [PWAAtestp;PWAAtestn];
PWAAtest = [PWAAtest,zeros(numTest,9)]';
PWAAtrain = reshape(PWAAtrain,[lenthSeq 1 numTrain]);
PWAAtest = reshape(PWAAtest,[lenthSeq 1 numTest]);

% BPB编码
[BPBtrainp, BPBtrainn, BPBtestp, BPBtestn] = BPB(datatrainp,datatrainn,datatestp,datatestn);
BPBtrain = [BPBtrainp;BPBtrainn]';
BPBtest = [BPBtestp;BPBtestn]';
BPBtrain = reshape(BPBtrain,[lenthSeq 2 numTrain]);
BPBtest = reshape(BPBtest,[lenthSeq 2 numTest]);

% AAindex编码
[iindex1,iindex2,iindex3,iindex4] = AAindex(datatrainp,datatrainn,datatestp,datatestn);
AAindextrain = [iindex1;iindex2]';
AAindextest = [iindex3;iindex4]';
AAindextrain = reshape(AAindextrain(1:540,:),[30 18 numTrain]);
AAindextest = reshape(AAindextest(1:540,:),[30 18 numTest]);

% BeOnehot编码
[BeOnehot1, BeOnehot2] = BeOnehot(datatrain,datatest);
BeOnehottrain = reshape(BeOnehot1,[numTrain,30,20]);
BeOnehottrain = permute(BeOnehottrain,[2 3 1]);
BeOnehottest = reshape(BeOnehot2,[numTest,30,20]);
BeOnehottest = permute(BeOnehottest,[2 3 1]);

% EGAAC编码
[EGAACtrain1,EGAACtest1] = EGAAC(datatrain,datatest);
EGAACtrain = reshape(EGAACtrain1,[numTrain,27,5]);
EGAACtrain = cat(2,EGAACtrain,zeros(numTrain,3,5));
EGAACtrain = permute(EGAACtrain,[2 3 1]);
EGAACtest = reshape(EGAACtest1,[numTest,27,5]);
EGAACtest = cat(2,EGAACtest,zeros(numTest,3,5));
EGAACtest = permute(EGAACtest,[2 3 1]);

%% 合并编码
train_data1 = cat(2,pkCode_Train,AACtrain,PWAAtrain,PSDAAPtrain); %pkCode_Train,PSDAAPtrain,PWAAtrain,AACtrain,EGAACtrain
test_data1 = cat(2,pkCode_Test,AACtest,PWAAtest,PSDAAPtest); %pkCode_Test,PSDAAPtest,PWAAtest,AACtest,EGAACtest
train_data2 = cat(2,train_data1,AAindextrain,BPBtrain); %pkCode_Train,PSDAAPtrain,PWAAtrain,AACtrain,EGAACtrain
test_data2 = cat(2,test_data1,AAindextest,BPBtest); %pkCode_Test,PSDAAPtest,PWAAtest,AACtest,EGAACtest
train_data3 = cat(2,train_data1,AAindextrain,EGAACtrain); %pkCode_Train,PSDAAPtrain,PWAAtrain,AACtrain,EGAACtrain
test_data3 = cat(2,test_data1,AAindextest,EGAACtest); %pkCode_Test,PSDAAPtest,PWAAtest,AACtest,EGAACtest
train_data31 = cat(2,train_data1,BPBtrain,AAindextrain,EGAACtrain); %pkCode_Train,PSDAAPtrain,PWAAtrain,AACtrain,EGAACtrain
test_data31 = cat(2,test_data1,BPBtest,AAindextest,EGAACtest); %pkCode_Test,PSDAAPtest,PWAAtest,AACtest,EGAACtest
train_data = {train_data2,train_data3,train_data31};
test_data = {test_data2,test_data3,test_data31};
model_RF = {};
for j = 1:5
    j
    for i = 1:size(train_data,2)
        i
        %% method 2 3 4 转换数据格式，以适应神经网络需要的输入
        %训练数据
        XTrain1 = permute(train_data{i},[2 1 3]); %转置
        rng('default');
        t1 = rng;
        idx_Train = randperm(size(XTrain1,3),size(XTrain1,3)*0.8);
        rng(t1);
        idx1(i,:,j) = idx_Train;
        XTrain2 = XTrain1(:,:,idx_Train);%样本打乱顺序的
        XTrain = num2cell(XTrain2,[1 2]);% 三维矩阵转换成cell，一个矩阵变成一个cell
        XTrain = squeeze(XTrain); %把一维的压缩掉
        YTrain1 = categorical([ones(size(datatrainp,1),1);zeros(size(datatrainn,1),1)]);%原始类标签，前一半正，后一半负
        YTrain = YTrain1(idx_Train,:); %打乱后的类标签
        
        % 测试数据
        XTest1 = permute(test_data{i},[2 1 3]);
        rng('default');
        t2 = rng;
        idx_Test = randperm(size(XTest1,3),size(XTest1,3));    
        rng(t2);
        idx2(i,:,j) = idx_Test;
        XTest2 = XTest1(:,:,idx_Test);
        XTest = num2cell(XTest2,[1 2]);
        XTest = squeeze(XTest);
        YTest = [ones(size(datatestp,1),1);zeros(size(datatestn,1),1)];
        YTest = YTest(idx_Test,:);
        %% 神经网络初始化
        inputSize = size(XTrain{1},1);
        numHiddenUnits1 = 110;
        numHiddenUnits2 = 110;
        numClasses = 2;
        dropout = 0.2;
        maxEpochs = 30;
        miniBatchSize = floor(0.005*size(XTrain,1));   
        %构建神经网络
        layers = [ ...
            sequenceInputLayer(inputSize)%输入层        
            bilstmLayer(numHiddenUnits1,'OutputMode','sequence')% 神经网络结构，四个一套，自己设置        
            batchNormalizationLayer %数据归一化
            reluLayer
            dropoutLayer(dropout)
            bilstmLayer(numHiddenUnits2,'OutputMode','last')% 神经网络结构，四个一套，自己设置 
            batchNormalizationLayer %数据归一化
            reluLayer
            dropoutLayer(dropout)
            fullyConnectedLayer(numClasses)%全连接层
            softmaxLayer %激活函数
            classificationLayer];   %分类层 
        options = trainingOptions('adam', ...%梯度下降函数
            'MaxEpochs',maxEpochs, ...%最大训练轮次
            'L2Regularization',1.0000e-03,...%正则化        
            'MiniBatchSize',miniBatchSize, ...%最小批次
            'GradientThreshold',2, ...
            "ExecutionEnvironment",'GPU',...%可选
            'Shuffle','every-epoch',...%打乱数据，混洗每一批次数据
            'Verbose',false, ...%命令窗口显示训练细节与否
            'LearnRateDropPeriod',90,... 
            'LearnRateDropFactor',0.1,...
            'GradientDecayFactor',0.85,...        
            'Plots','training-progress');%显示训练进度
        %开始训练
        net_bilstm = trainNetwork(XTrain,YTrain,layers,options);
        analyzeNetwork(net_bilstm);
        
        %计算独立测试数据各分类指标
        [YPred_bilstm,scores_bilstm(:,:,i,j)] = classify(net_bilstm,XTest,'MiniBatchSize',miniBatchSize);
        YPred1 = double(YPred_bilstm);
        YPred2(:,i) = YPred1-1;
        [SN1(:,i,j), SP1(:,i,j), ACC1(:,i,j),MCC1(:,i,j),Precision1(:,i,j),F1_score1(:,i,j)] = metrics(YPred2(:,i), YTest); 
    end    
    YPred3 = round(sum(YPred2,2)./size(YPred2,2));
    [SN3(:,j), SP3(:,j), ACC3(:,j),MCC3(:,j),Precision3(:,j),F1_score3(:,j)] = metrics(YPred3, YTest); 
    
    %% method 6
    traindata_cnn = [pkCode_Train,AACtrain];    
    traindata_cnn = reshape(traindata_cnn,[size(traindata_cnn,1) size(traindata_cnn,2) 1 size(traindata_cnn,3)]);
    %     YTrain_cnn = categorical([ones(size(datatrainp,1),1);zeros(size(datatrainn,1),1)]);
    %     rng('default');
    %     t1_cnn = rng;
    %     idx_Train_cnn = randperm(size(traindata_cnn,4),size(traindata_cnn,4));
    traindata_cnn = traindata_cnn(:,:,:,idx_Train);
    %     YTrain_cnn = YTrain_cnn(idx_Train,:);
    %     rng(t1_cnn);
    %     idx1_cnn(j,:) = idx_Train_cnn;

    testdata_cnn = [pkCode_Test,AACtest];    
    testdata_cnn = reshape(testdata_cnn,[size(testdata_cnn,1) size(testdata_cnn,2) 1 size(testdata_cnn,3)]);
    testdata_cnn = testdata_cnn(:,:,:,idx_Test);
    
    % 神经网络初始化
    inputSize = [size(testdata_cnn,1) size(testdata_cnn,2) size(testdata_cnn,3)];
    numClasses = 2;
    dropout = 0.3;
    maxEpochs = 30;
    miniBatchSize = floor(0.002*size(traindata_cnn,4));
    
    layers = [ ...
        imageInputLayer(inputSize)%输入层
        convolution2dLayer([2 2],32)% 神经网络结构，四个一套，自己设置
        maxPooling2dLayer([3 3])
        batchNormalizationLayer
        reluLayer
        dropoutLayer(dropout)
        convolution2dLayer([3 3],16)% 神经网络结构，四个一套，自己设置
        maxPooling2dLayer([3 3])
        batchNormalizationLayer
        reluLayer
        dropoutLayer(dropout)
    %         convolution2dLayer([3 3],64)% 神经网络结构，四个一套，自己设置
    %         maxPooling2dLayer([2 2])
    %         batchNormalizationLayer
    %         reluLayer
    %         dropoutLayer(dropout)
        fullyConnectedLayer(numClasses)%全连接层
        softmaxLayer %激活函数
        classificationLayer];   %分类层 
    options = trainingOptions('adam', ...%梯度下降函数
        'MaxEpochs',maxEpochs, ...%最大训练轮次
        'L2Regularization',1.0000e-03,...%正则化
        'InitialLearnRate',0.001,...
        'MiniBatchSize',miniBatchSize, ...%最小批次
        'GradientThreshold',1, ...
        "ExecutionEnvironment",'GPU',...%可选
        'Shuffle','every-epoch',...%打乱数据，混洗每一批次数据
        'Verbose',false, ...%命令窗口显示训练细节与否
        'LearnRateDropPeriod',2,...
        'LearnRateDropFactor',0.1,...
        'Plots','training-progress');%显示训练进度
    %开始训练
    net_cnn = trainNetwork(traindata_cnn,YTrain,layers,options);
%     analyzeNetwork(net_cnn);
    %计算独立测试数据各分类指标 
    [YPred_cnn,scores_CNN(:,:,j)] = classify(net_cnn,testdata_cnn,'MiniBatchSize',miniBatchSize);  
    YPred1_cnn = double(YPred_cnn);
    YPred2_cnn = YPred1_cnn-1;
    [Recall_cnn(:,j), SP_cnn(:,j), ACC_cnn(:,j),MCC_cnn(:,j),Precision_cnn(:,j),F1_score_cnn(:,j)] = metrics(YPred2_cnn, YTest);

    %% method 5--编码
    AACtrain_x = [AACtrainp;AACtrainn];
    AACtrain_x = AACtrain_x(idx_Train,:);
    AACtest_x = [AACtestp;AACtestn];
    AACtest_x = AACtest_x(idx_Test,:);
    BeOnehottrain_x = BeOnehot1(idx_Train,:);
    BeOnehottest_x = BeOnehot2(idx_Test,:);
    
    train_x = [AACtrain_x,BeOnehottrain_x]; 
    test_x = [AACtest_x,BeOnehottest_x]; 
         
    model_tree = TreeBagger(600,train_x,YTrain,'Method','classification');
    model_RF{j} = model_tree;
    [pre_tree,score_tree(:,:,j)] = predict(model_RF{j},test_x);
    pre1 = double(cell2mat(pre_tree))-48;
    [SN_RF(:,j),SP_RF(:,j), ACC_RF(:,j),MCC_RF(:,j),Precision_RF(:,j),F1_score_RF(:,j)] = metrics(pre1,YTest);
    
    % 5种方法集成
    YPred_final1 =[YPred2,pre1,YPred2_cnn];
    YPredlabel = round(sum(YPred_final1,2)./size(YPred_final1,2));
    [Recall_final(:,j),SP_final(:,j),ACC_final(:,j),MCC_final(:,j),Precision_final(:,j),F1_score_final(:,j)] = metrics(YPredlabel,YTest);    
end

% 程序运行5遍的平均结果
ACC_mean = mean(ACC_final)  % ACC：0.818
Precision_mean = mean(Precision_final)  % Precision：0.81
Recall_mean = mean(Recall_final)  % recall：0.854
F1_score_mean = mean(F1_score_final)  %F-score：0.825

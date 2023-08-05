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

datatestp = datatest(1:750,:);
datatestn = datatest(751:end,:);
numTrain = size(datatrain,1);
numTest = size(datatest,1);
lenthSeq = size(datatest{1},2)-1;
%% 自编码
pkCode_Train = pkCode(datatrain);
pkCode_Test = pkCode(datatest);
% PSDAAP编码
[PSDAAPtrainp, PSDAAPtrainn, PSDAAPtestp, PSDAAPtestn] = PSDAAP(datatrainp,datatrainn,datatestp,datatestn);
PSDAAPtrain = [PSDAAPtrainp;PSDAAPtrainn]';
PSDAAPtrain = [PSDAAPtrain;zeros(1,numTrain)];
PSDAAPtest = [PSDAAPtestp;PSDAAPtestn]';
PSDAAPtest = [PSDAAPtest;zeros(1,numTest)];
PSDAAPtrain = reshape(PSDAAPtrain,[lenthSeq 1 numTrain]);
PSDAAPtest = reshape(PSDAAPtest,[lenthSeq 1 numTest]);
%AAC编码
[AACtrainp,AACtrainn] = AAC(datatrainp,datatrainn);
[AACtestp,AACtestn] = AAC(datatestp,datatestn);
AACtrain = [AACtrainp;AACtrainn];
AACtrain = [AACtrain,zeros(numTrain,9)];
AACtest = [AACtestp;AACtestn];
AACtest = [AACtest,zeros(numTest,9)];
AACtrain = reshape(AACtrain,[lenthSeq 1 numTrain]);
AACtest = reshape(AACtest,[lenthSeq 1 numTest]);
% PWAA编码
[PWAAtrainp,PWAAtrainn] = PWAA(datatrainp,datatrainn);
[PWAAtestp,PWAAtestn] = PWAA(datatestp,datatestn);
PWAAtrain = [PWAAtrainp;PWAAtrainn];
PWAAtrain = [PWAAtrain,zeros(numTrain,9)];
PWAAtest = [PWAAtestp;PWAAtestn];
PWAAtest = [PWAAtest,zeros(numTest,9)];
PWAAtrain = reshape(PWAAtrain,[lenthSeq 1 numTrain]);
PWAAtest = reshape(PWAAtest,[lenthSeq 1 numTest]);

%% 合并编码
train_data = cat(2,pkCode_Train(:,1:5,:),PSDAAPtrain,PWAAtrain,AACtrain);
test_data = cat(2,pkCode_Test(:,1:5,:),PSDAAPtest,PWAAtest,AACtest);

%% 转换数据格式，以适应神经网络需要的输入
%训练数据
XTrain1 = permute(train_data,[2 1 3]); %转置
idx_Train = randperm(size(XTrain1,3),size(XTrain1,3));
XTrain2 = XTrain1(:,:,idx_Train);
XTrain = num2cell(XTrain2,[1 2]);
XTrain = squeeze(XTrain);
YTrain1 = categorical([ones(size(datatrainp,1),1);zeros(size(datatrainn,1),1)]);
YTrain = YTrain1(idx_Train,:);

% 测试数据
XTest1 = permute(test_data,[2 1 3]);
idx_Test = randperm(size(XTest1,3),size(XTest1,3));
XTest2 = XTest1(:,:,idx_Test);
XTest = num2cell(XTest2,[1 2]);
XTest = squeeze(XTest);
YTest1 = [ones(size(datatestp,1),1);zeros(size(datatestn,1),1)];
YTest = YTest1(idx_Test,:);
%% 神经网络初始化
inputSize = size(XTrain{1},1);
numHiddenUnits1 = 30;
numHiddenUnits2 = 20;
numClasses = 2;
dropout = 0.1;
maxEpochs = 30;
miniBatchSize = floor(0.005*size(XTrain,1));
%% 交叉验证
times = 10;
Folds = 10;
for i = 1:times
    cvp = cvpartition(YTrain,'KFold',Folds);
    for k = 1:Folds
        XTrain_data = XTrain(cvp.training(k),:);
        YTrain_data = YTrain(cvp.training(k),:);
        XValidation = XTrain(cvp.test(k),:);
        YValidation = YTrain(cvp.test(k),:);
        %构建神经网络
        layers = [ ...
            sequenceInputLayer(inputSize)%输入层
            bilstmLayer(numHiddenUnits1,'OutputMode','sequence')% 神经网络结构，四个一套，自己设置
            batchNormalizationLayer %数据归一化
            reluLayer
            dropoutLayer(dropout)
            bilstmLayer(numHiddenUnits2,'OutputMode','last')% 神经网络结构，四个一套，自己设置
            batchNormalizationLayer
            reluLayer
            dropoutLayer(dropout)
            fullyConnectedLayer(numClasses)%全连接层
            softmaxLayer %激活函数
            classificationLayer];   %分类层 
        options = trainingOptions('adam', ...%梯度下降函数
            'MaxEpochs',maxEpochs, ...%最大训练轮次
            'L2Regularization',1.0000e-03,...%正则化
            'ValidationData',{XValidation,YValidation}, ...%验证数据
            'MiniBatchSize',miniBatchSize, ...%最小批次
            'GradientThreshold',1, ...
            "ExecutionEnvironment",'GPU',...%可选
            'Shuffle','every-epoch',...%打乱数据，混洗每一批次数据
            'Verbose',false, ...%命令窗口显示训练细节与否
            'Plots','training-progress');%显示训练进度
        %开始训练
        net = trainNetwork(XTrain_data,YTrain_data,layers,options);
        %计算验证数据各分类指标
        YPred_Validation = classify(net,XValidation,'MiniBatchSize',miniBatchSize);
        YPred_Validation1 = double(YPred_Validation);
        YPred_Validation2 = YPred_Validation1 - 1;
        [SNV(:,k,i), SPV(:,k,i), ACCV(:,k,i),MCCV(:,k,i),PrecisionV(:,k,i),F1_scoreV(:,k,i)] = metrics(YPred_Validation2, double(YValidation)-1);
        %计算独立测试数据各分类指标
        YPred = classify(net,XTest,'MiniBatchSize',miniBatchSize);
        YPred1 = double(YPred);
        YPred = YPred1-1;
        [SN(:,k,i), SP(:,k,i), ACC(:,k,i),MCC(:,k,i),Precision(:,k,i),F1_score(:,k,i)] = metrics(YPred, YTest);
    end
end
%% 保存结果
result_Val = [ACCV;F1_scoreV;PrecisionV;SNV];
result_Test = [ACC;F1_score;Precision;SN];
save('biLstm(BPB2T)_net_10_20221102.mat','net','miniBatchSize');
save('biLstm(BPB2T)_result20221102_10.mat','result_Val','result_Test');

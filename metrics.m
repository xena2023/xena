function [SN, SP, ACC,MCC,Precision,F1_score] = metrics(predict_label, label)
%     predict_stat = tabulate(predict_label);
%     test_stat = tabulate(label);
    a = predict_label + label;
    TP = length(find(a == 2));
    TN = length(find(a == 0));
    b = predict_label - label;
    FN = length(find(b == -1));
    FP = length(find(b == 1));
    SN =TP/(TP+FN); %Recall
    SP = TN/(FP+TN);
    ACC = (TP+TN)/(TP+FN+FP+TN);
    MCC = (TP*TN - FP*FN)/((TP+FN)*(TP+FP)*(TN+FN)*(TN+FP))^0.5;
    Precision = TP/(TP+FP);
    F1_score = (2*Precision*SN)/(Precision+SN);
end
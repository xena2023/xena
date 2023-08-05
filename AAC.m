function [AAC1,AAC2] =AAC(positive,negative)
    positive = positive';
    negative = negative';
    Np = length(positive);  %number of positive samples
    Nn = length(negative);  %number of negative samplesd
    AA = 'ACDEFGHIKLMNPQRSTVWYX';
    AAC1 = zeros(Np,21);  %PPT1为Np*4的全0矩阵
    for m = 1:Np
        M = length(positive{1,m});  %两个正样本序列构成一个1*2的矩阵,当m=1时,表示的是第一行第一列,即要求第一条序列的长度；
         for j = 1:M
            t = positive{1,m}(j);  %依次遍历每一条序列中的所有碱基；
            k = strfind(AA,t);  %strfind表示在AA中寻找t,当t=A时,k=1;当t=C时,k=2;当t=G时,k=3;当t=T时,k=4;
            AAC1(m,k)=AAC1(m,k)+1;
         end
        AAC1(m,:) = AAC1(m,:)/M;
    end

    AAC2 = zeros(Nn,21);
    for m = 1:Nn
        M = length(negative{1,m});
        for j = 1:M
           t = negative{1,m}(j);
           k = strfind(AA,t);
           AAC2(m,k) = AAC2(m,k)+1;
        end
        AAC2(m,:) = AAC2(m,:)/M;
    end
end  


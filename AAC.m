function [AAC1,AAC2] =AAC(positive,negative)
    positive = positive';
    negative = negative';
    Np = length(positive);  %number of positive samples
    Nn = length(negative);  %number of negative samplesd
    AA = 'ACDEFGHIKLMNPQRSTVWYX';
    AAC1 = zeros(Np,21);  %PPT1ΪNp*4��ȫ0����
    for m = 1:Np
        M = length(positive{1,m});  %�������������й���һ��1*2�ľ���,��m=1ʱ,��ʾ���ǵ�һ�е�һ��,��Ҫ���һ�����еĳ��ȣ�
         for j = 1:M
            t = positive{1,m}(j);  %���α���ÿһ�������е����м����
            k = strfind(AA,t);  %strfind��ʾ��AA��Ѱ��t,��t=Aʱ,k=1;��t=Cʱ,k=2;��t=Gʱ,k=3;��t=Tʱ,k=4;
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


function [twi1,twi2,twi3,twi4]=PSDAAP(positive,negative,Test_p,Test_n)
    positive = positive';
    negative = negative';
    Test_p = Test_p';
    Test_n = Test_n';
    Np=length(positive);%number of positive samples
    M=length(positive{1,1});
    x=(M+1)/2;
    %中间位置的“T”或“S”去掉
    for i=1:Np
        positive{1,i}(x)='';
        s=positive{1,i};
        positive{1,i}=s;
    end
    Nn=length(negative);%number of negative samples
    %中间位置的“T”去掉
    for i=1:Nn
        negative{1,i}(x)='';
        s=negative{1,i};
        negative{1,i}=s;
    end
    AA='ACDEFGHIKLMNPQRSTVWYX';
    M=length(positive{1,1});
    F1=zeros(21^2,M-1);%记录positive dataset中每个氨基酸在每个位置出现的频率
    F2=zeros(21^2,M-1);%记录negative dataset中每个氨基酸在每个位置出现的频率
    for m=1:Np
        for j=1:M-1
            t1=positive{1,m}(j);
            k1=strfind(AA,t1);
            t2=positive{1,m}(j+1);
            k2=strfind(AA,t2);
            F1(21*(k1-1)+k2,j)=F1(21*(k1-1)+k2,j)+1;
        end
    end
    F1=F1/Np;
    for m=1:Nn
        for j=1:M-1
            t1=negative{1,m}(j);
            k1=strfind(AA,t1);
            t2=negative{1,m}(j+1);
            k2=strfind(AA,t2);
            F2(21*(k1-1)+k2,j)=F2(21*(k1-1)+k2,j)+1;
        end
    end
    F2=F2/Nn;
    F=F1-F2;
    twi1=zeros(Np,M-1); %positive dataset的特征矩阵
    twi2=zeros(Nn,M-1); %negative dataset的特征矩阵
    for m=1:Np
        for j=1:M-1
            t1=positive{1,m}(j);
            k1=strfind(AA,t1);
            t2=positive{1,m}(j+1);
            k2=strfind(AA,t2);
            twi1(m,j)=F(21*(k1-1)+k2,j);

        end
    end
    for m=1:Nn
        for j=1:M-1
            t1=negative{1,m}(j);
            k1=strfind(AA,t1);
            t2=negative{1,m}(j+1);
            k2=strfind(AA,t2);
            twi2(m,j)=F(21*(k1-1)+k2,j);

        end
    end

    n3=length(Test_p);
    n4=length(Test_n);
    for i=1:n3
        Test_p{1,i}(x)='';
        s=Test_p{1,i};
        Test_p{1,i}=s;
    end
    for i=1:n4
        Test_n{1,i}(x)='';
        s=Test_n{1,i};
        Test_n{1,i}=s;
    end
    twi3=zeros(n3,M-1); %positive dataset的特征矩阵
    twi4=zeros(n4,M-1); %negative dataset的特征矩阵
    for m=1:n3
        for j=1:M-1
            t1=Test_p{1,m}(j);
            k1=strfind(AA,t1);
            t2=Test_p{1,m}(j+1);
            k2=strfind(AA,t2);
            twi3(m,j)=F(21*(k1-1)+k2,j);
        end
    end
    for m=1:n4
        for j=1:M-1
            t1=Test_n{1,m}(j);
            k1=strfind(AA,t1);
            t2=Test_n{1,m}(j+1);
            k2=strfind(AA,t2);
            twi4(m,j)=F(21*(k1-1)+k2,j);
        end
    end
end

function [C1,C2] = PWAA(positive,negative)
    % 取自zhuyan论文11（21页）
    Np=length(positive);%number of positive samples
    Nn=length(negative);%number of negative samples
    L1=length(positive{1,1});%每个肽段的长度，正负样本一样长
    AA='ACDEFGHIKLMNPQRSTVWYX';
    L=(L1+1)/2;
    L2=L-1;
    for i=1:Np
        for k=1:21
            for j=1:L1
                t=AA(k);
                s1=positive{i,1}(j);
                if s1~=t
                    a(i,j)=0;
                else
                    a(i,j)=1;
                end
            end
            for ii=1:L1
                b(1,ii)=a(i,ii)*((ii-L)+(abs(ii-L))/L2);
            end
            bb(i,k)=sum(b(:));
            C1(i,k)=1/(L2*L)*bb(i,k);
        end
    end
    for i=1:Nn
        for k=1:21
            for j=1:L1
                t=AA(k);
                s1=negative{i,1}(j);
                if s1~=t
                    a(i,j)=0;
                else
                    a(i,j)=1;
                end
            end
            for ii=1:L1
                b(1,ii)=a(i,ii)*((ii-L)+(abs(ii-L))/L2);
            end
            bb(i,k)=sum(b(:));
            C2(i,k)=1/(L2*(L2+1))*bb(i,k);
        end
end

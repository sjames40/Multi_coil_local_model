function ODCT = genODCT1(n,J, center,DLflag)
% generates (n x J) iODCT dictionary, and can only handle any value of J.
% (Check if implementation is the best possible)
% Setting center = 1 subtracts the mean from each atom. Also generates DCT
% for transform domain applications, upon setting DLflag to 0,

tflg = 0;
if n>J
    tmp = n; n = J; J = tmp; tflg = 1;
end

if DLflag == 1
    ODCT = idct(eye(ceil(sqrt(J)))); % IDCT for Dictionary Learning
else
    ODCT = dct(eye(ceil(sqrt(J)))); % DCT for Transform Learning
end

ODCT = ODCT(1:ceil(sqrt(n)),:);
ODCT = kron(ODCT,ODCT);

if tflg ==1
    ODCT = ODCT';
    tmp = J; J = n; n = tmp;
end

%%
if center == 1
    ODCT=[ODCT(:,1) ODCT(:,2:end)-repmat(mean(ODCT(:,2:end)),n,1)];
end
%%
for i=1:J
   ODCT(:,i)=ODCT(:,i)/norm(ODCT(:,i)); 
end

ODCT = ODCT(:,1:J);



end
function [out] = random_gen(i)
while(1)
   randNumber = randi([0 1],1,2);
   if(randNumber(1)~=randNumber(2))
       break;
   end
end
out=zeros(2);
if(i==1)
    out(1,1:2)=randNumber;
    out(2,1:2)=randNumber;
elseif (i==2)
    out(1,1:2)=randNumber;
    out(2,1:2)=~randNumber;
end
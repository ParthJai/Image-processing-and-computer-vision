%The program implemants visual encryption.
%In case if user enters rgb/grayscale image then function
%graythresh automatically calculates the level value.
%If user wants to select level value manually
%Uncomment the whole commented part and comment out line 27.
clc;
clear all; 

waitfor(warndlg(sprintf('Visual cryptography works on binary images. Any RGB or grayscale image will be converted to binary image.Do you want to continue?\n Press cancel in next dialog box to exit.')));
[filename pathname]=uigetfile({'*.jpg'},'File Selector');
fullpathname=strcat(pathname,filename);
if (filename==0)
    return;
end
choice=questdlg('Did you select RGB/grayscale image or binary image?','Image menu','RGB/grayscale','binary','binary');


switch choice
    case 'RGB/grayscale'
        %waitfor(msgbox('As you selected RGB/grayscale image you need to define a threshold between 0 and 1. All pixels having value greater than threshold will be assigned 1 and less than threshold will be zero.Try experimenting with different values to see what is best.','Convert into binary image'));
        %while(1)
           %prompt={'Enter threshold value between 0 and 1:'};
           %dlg_title='Input for threshold';
           %num_lines=1;
           %def={'0.3'};
           %answer = inputdlg(prompt,dlg_title,num_lines,def);
           %answer=cell2mat(answer);
           %answer=str2num(answer);
           i=imread(fullpathname);
           answer = graythresh(i)
           i=im2bw(i,answer);
           figure;imshow(i);
           %choice1=questdlg('Do you want to change the threshold value?','select threshold','yes','no','no');
           %switch choice1
            %   case 'yes'
             %      continue;
              % case 'no'
               %    break;
           %end
        %end
    case 'binary'
        
end

s=size(i);
a=s(1);
b=s(2);
part1=zeros(a,b);
part2=zeros(a,b);
[x y]=find(i);
len=length(x);


tic;
for I=1:len
    a1=x(I);b1=y(I);
    part_pixels=random_gen(1);
    part1(a1,b1:b1+1)=part_pixels(1,1:2);
    part2(a1,b1:b1+1)=part_pixels(2,1:2);
end


toc;
[x y]=find(i==0);
len=length(x);
tic;
for I=1:len
    a1=x(I);b1=y(I);
    part_pixels=random_gen(2);
    part1(a1,b1:b1+1)=part_pixels(1,1:2);
    part2(a1,b1:b1+1)=part_pixels(2,1:2);
end
toc;


decrypt=bitor(part1,part2);
decrypt=~decrypt;
figure;imshow(decrypt);title('decrypted image');
figure;imshow(part1);title('part1');
figure;imshow(part2);title('part2');
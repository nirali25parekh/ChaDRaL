%{
ogImage=imread('F:\PROGRAMMING\Stegnography\final\baboon.png');
encryptedImage=imread('F:\PROGRAMMING\Stegnography\final\baboon_encrypt.png');
n=size(ogImage);
M=n(1);
N=n(2);
MSE = sum(sum((ogImage-encryptedImage).^2))/(M*N);
fprintf('\nMSE: %7.2f ', MSE);
PSNR = 10*log10(256*256/MSE);

fprintf('\nPSNR: %9.7f dB', PSNR);
%}

clc;
filename1 = 'F:\PROGRAMMING\Stegnography\final\peppers.png'
filename2 = 'F:\PROGRAMMING\Stegnography\final\peppers_recovered.png'
image1=imread(filename1);
image2=imread(filename2);
   
[row,col] = size(image1)
size_host = row*col;
o_double = double(image1);
w_double = double(image2);
s=0;
for j = 1:size_host; % the size of the original image
  s = s+(w_double(j) - o_double(j))^2 ; 
end
mse=s/size_host;
psnr =10*log10((255)^2/mse);
display 'MSE', mse
display 'PSNR',psnr


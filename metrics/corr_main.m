%usage [k1,k2,k3,k4,k5,k6]=resim_korelasyon('lennagri.bmp','lenagrisifreli1.bmp',0);
%k1,k2,k3 Original Image correlation coefficient
%k4,k5,k6 encrypted image correlation coefficient
%color  ==> 0 gray , 1 RGB
%kyatayO,kdikeyO,kkosegenO,kyatayI,kdikeyI,kkosegenI  correlation coefficients

%{
[k1,k2,k3,k4,k5,k6]=resim_korelasyon('lena.png','lena_encrypt.png',1);
display "coeff" k1 k2 k3 k4 k5 k6;
%}

filename1 = 'F:\PROGRAMMING\Stegnography\metrics\frymire_encrypt.png'
image1=imread(filename1);

rxy = AdjancyCorrPixel(image1);
display 'corr',rxy
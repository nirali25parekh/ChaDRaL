
clc;
filename1 = 'F:\PROGRAMMING\Stegnography\final\peppers_encrypt.png'
filename2 = 'F:\PROGRAMMING\Stegnography\final\peppers.png'
img_a=imread(filename1);
img_b=imread(filename2);
   
%{
num_of_pix = size(img_a, 1) * size(img_b, 2);

uaci_score = sum( abs( double(img_a(:)) - double(img_b(:)) ) ) / num_of_pix / largest_allowed_val;
npcr_score = sum( double( img_a(:) ~= img_b(:) ) ) / num_of_pix;

display 'UACI', uaci_score
display 'NPCR',npcr_score
%}

results = NPCR_and_UACI( img_a, img_b, 1, 255 );

display 'results' results
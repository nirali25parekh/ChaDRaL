function  r_xy=AdjancyCorrPixel( P )

x1 = double(P(1:end-1,1:end-1,1) );
y1 = double(P(2:end,2:end,1));
randIndex1 = randperm(numel(x1));
randIndex1 = randIndex1(1:3000);
x = x1(randIndex1);
y = y1(randIndex1);
r_xy = corrcoef(x,y);
scatter(x,y);
xlabel('Pixel gray value on location (x,y)')
ylabel('Pixel gray value on location (x+1,y+1)')
end


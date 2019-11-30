% This function is to implement QASD algorithm.
% Input each pair of a reference image and the corresponding distorted image,
% returns a quality score.
function Score = QASD(imageRef, imageDis, Dictionary, blockSize, Sparsity)
% if nargin<1
%     quit
% end
imageRef = imread( imageRef );
imageDis = imread( imageDis );
blockSize=str2double(blockSize);
Sparsity=str2double(Sparsity);
Ir = double( rgb2gray( imageRef ) );
Id = double( rgb2gray( imageDis ) );
blockDim = blockSize^2;
[sizeY,sizeX] = size(Ir);
load(Dictionary);
%% Divide input images into blocks, calculate sparse coefficients
gridY = 1 : blockSize : sizeY-mod(sizeY,blockSize);
gridX = 1 : blockSize : sizeX-mod(sizeX,blockSize);
Y = length(gridY);  X = length(gridX);
Xr = zeros(blockDim, Y*X);
Xd = zeros(blockDim, Y*X);
A_thM=zeros(Sparsity,Y*X); % sparse coefficient matrix of reference image
B_thM=zeros(Sparsity,Y*X); % sparse coefficient matrix of distorted image
k = 0;
for i = gridY;
    for j = gridX
        k = k+1;
        imagePatchr=Ir(i:i+blockSize-1, j:j+blockSize-1);
        imagePatchd=Id(i:i+blockSize-1, j:j+blockSize-1);
        Xr(:,k)=imagePatchr(:);
        Xd(:,k)=imagePatchd(:);
        
        A_th = OMP(Dictionary,Xr(:,k),Sparsity);
        [SN,~,WS]=find(A_th);
        L0=length(SN);
        if L0==0
            break;
        elseif L0<Sparsity
            A_thM(1:L0,k)=WS;
            Dictionary_R=Dictionary(:,SN);
            B_th = OMP(Dictionary_R,Xd(:,k),L0);
            B_thM(1:L0,k)=full(B_th);
        else
            A_thM(:,k)=WS;
            Dictionary_R=Dictionary(:,SN);
            B_th = OMP(Dictionary_R,Xd(:,k),L0);
            B_thM(:,k)=full(B_th);
        end
    end
end
%% Calculate feature map similarity score
featureMapr=reshape( sqrt(sum(A_thM.^2)) , [ X Y ]);
featureMapr=featureMapr';
% featureMapr=mat2gray(imresize( mat2gray(featureMapr) , [ sizeY/2 sizeX/2 ], 'bicubic' ));
featureMapr=mat2gray(imresize( mat2gray(featureMapr) , [ sizeY sizeX ], 'bicubic' ));

featureMapd=reshape( sqrt(sum(B_thM.^2)) , [ X Y ]);
featureMapd=featureMapd';
% featureMapd=mat2gray(imresize( mat2gray(featureMapd) , [ sizeY/2 sizeX/2 ], 'bicubic' ));
featureMapd=mat2gray(imresize( mat2gray(featureMapd) , [ sizeY sizeX ], 'bicubic' ));

constForEner = 1.15;%fixed
featureSimMatrix = (2 * featureMapr .* featureMapd + constForEner) ./ (featureMapr.^2 + featureMapd.^2 + constForEner);

weight = max(featureMapr, featureMapd);

featureSimScore = featureSimMatrix.* weight;
featureSimScore = sum(sum(featureSimScore)) / sum(weight(:));
%% Calculate luminance similarity score
mXr = mean(Xr);
mXd = mean(Xd);

% select mean value pairs
Tm = 1.25;%fixed
mXe = abs(mXr-mXd);
mmXe = Tm*median(mXe);
mXd = mXd(:,mXe>=mmXe);
mXr = mXr(:,mXe>=mmXe);

meanXr = mXr-mean(mXr);
meanXd = mXd-mean(mXd);

constForLumi = 0.001;%fixed
luminanceSimScore = (sum(meanXr.*meanXd)+constForLumi) / (sqrt(sum(meanXr.^2)*sum(meanXd.^2))+constForLumi);
%%
% Obtain Y, Cb, Cr components
Y1 = 0.257 * double(imageRef(:,:,1)) + 0.504 * double(imageRef(:,:,2)) + 0.098 * double(imageRef(:,:,3))+16/255;
Y2 = 0.257 * double(imageDis(:,:,1)) + 0.504 * double(imageDis(:,:,2)) + 0.098 * double(imageDis(:,:,3))+16/255;
Cr1 = 0.439 * double(imageRef(:,:,1)) - 0.368 * double(imageRef(:,:,2)) - 0.071 * double(imageRef(:,:,3))+128/255;
Cr2 = 0.439 * double(imageDis(:,:,1)) - 0.368 * double(imageDis(:,:,2)) - 0.071 * double(imageDis(:,:,3))+128/255;
Cb1 = -0.148 * double(imageRef(:,:,1)) - 0.291 * double(imageRef(:,:,2)) + 0.439 * double(imageRef(:,:,3))+128/255;
Cb2 = -0.148 * double(imageDis(:,:,1)) - 0.291 * double(imageDis(:,:,2)) + 0.439 * double(imageDis(:,:,3))+128/255;

% Downsample the image
minDimension = min(sizeY,sizeX);
F = max(1,round(minDimension / 256));
aveKernel = fspecial('average',F);

aveCr1 = conv2(Cr1, aveKernel,'same');
aveCr2 = conv2(Cr2, aveKernel,'same');
Cr1 = aveCr1(1:F:sizeY,1:F:sizeX);
Cr2 = aveCr2(1:F:sizeY,1:F:sizeX);

aveCb1 = conv2(Cb1, aveKernel,'same');
aveCb2 = conv2(Cb2, aveKernel,'same');
Cb1 = aveCb1(1:F:sizeY,1:F:sizeX);
Cb2 = aveCb2(1:F:sizeY,1:F:sizeX);

aveY1 = conv2(Y1, aveKernel,'same');
aveY2 = conv2(Y2, aveKernel,'same');
Y1 = aveY1(1:F:sizeY,1:F:sizeX);
Y2 = aveY2(1:F:sizeY,1:F:sizeX);
%% Calculate gradient similarity score
dx = [3 0 -3; 10 0 -10;  3  0 -3]/16;
dy = [3 10 3; 0  0   0; -3 -10 -3]/16;

IxY1 = conv2(Y1, dx, 'same');
IyY1 = conv2(Y1, dy, 'same');
gradientMap1 = sqrt(IxY1.^2 + IyY1.^2);

IxY2 = conv2(Y2, dx, 'same');
IyY2 = conv2(Y2, dy, 'same');
gradientMap2 = sqrt(IxY2.^2 + IyY2.^2);

[sizeYR, sizeXR]=size(Y1);
featureMaprS=mat2gray(imresize( featureMapr , [ sizeYR sizeXR ], 'bicubic' ));
featureMapdS=mat2gray(imresize( featureMapd , [ sizeYR sizeXR ], 'bicubic' ));
weightS = max(featureMaprS, featureMapdS);

constForGM = 89;%fixed
gradientSimMatrix = (2*gradientMap1.*gradientMap2 + constForGM) ./(gradientMap1.^2 + gradientMap2.^2 + constForGM);
gradientSimScore = gradientSimMatrix.* weightS;
gradientSimScore = sum(sum(gradientSimScore)) / sum(weightS(:));
%% Calculate chromatic similarity score
constForChrom = 110;%fixed
ISimMatrix = (2 * Cr1 .* Cr2 + constForChrom) ./ (Cr1.^2 + Cr2.^2 + constForChrom);
QSimMatrix = (2 * Cb1 .* Cb2 + constForChrom) ./ (Cb1.^2 + Cb2.^2 + constForChrom);
chromaticSimMatrix=ISimMatrix .* QSimMatrix;
chromaticSimScore = chromaticSimMatrix.* weightS;
chromaticSimScore = sum(sum(chromaticSimScore)) / sum(weightS(:));
%% Obtain final score (pooling)
alpha = 0.25;%fixed
beta = 0.03;%fixed
gamma = 0.65;%fixed
Score=featureSimScore*(gradientSimScore.^alpha)*(chromaticSimScore.^beta)*(luminanceSimScore.^gamma);
% fid=fopen('quality_score.txt', 'w');
% fprintf( fopen('quality_score.txt', 'w'),'%f\n', Score);
% fclose(fid);
% fprintf( '%f\n', Score);
return
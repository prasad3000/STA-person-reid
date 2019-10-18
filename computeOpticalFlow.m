clc;
clear all;

rootDir = fullfile('G:', 'git_hub', 'STA', 'iLIDS-VID');

for person = 1:319
    disp(person)
    for cam = 1:2
        camName = {'cam_a', 'cam_b'};
        
        dataDir = fullfile(rootDir, 'i-LIDS-VID', 'sequences',['cam', num2str(cam)], ['person', sprintf('%03i', person)]);
        files = dir(dataDir);
        
        saveDir = fullfile(rootDir, 'i-LIDS-VID-OF-HVP', 'sequences',['cam', num2str(cam)], ['person', sprintf('%03i', person)]);
        
        if ~exist(saveDir)
            mkdir(saveDir);
        end
        
        seqFiles = {};
        for f = 1:length(files)
            if length(files(f).name) > 4 && ~isempty(findstr(files(f).name, '.png'))
                seqFiles = [seqFiles files(f).name];
            end
        end
        
        optical = opticalFlowLK('NoiseThreshold',0.009);
        %optical = vision.OpticalFlow('Method','Lucas-Kanade','OutputValue', 'Horizontal and vertical components in complex form');
        
        for f = 1:length(seqFiles)
                seqImg = imread(fullfile(dataDir,seqFiles{f}));
                optFlow = estimateFlow(optical, double(rgb2gray(seqImg)));
                
                fig = figure;
                imshow(seqImg) 
                hold on
                plot(optFlow,'DecimationFactor',[5 5],'ScaleFactor',10)
                hold off
                
                saveFile = fullfile(saveDir,seqFiles{f});
                saveas(fig, saveFile);
                
                clear fig;
                close all;
                
                
                %optFlow = step(optical,double(rgb2gray(seqImg)));
                
                %separate optFlow into mag and phase components
                %R = abs(optFlow);
                %theta = angle(optFlow);                
                
                %threshold to remove pixels with large magnitude values
                %ofThreshold = 50;
                %R = min(R,ofThreshold);
                %R = max(R,-1*ofThreshold);                
                
                %convert back to complex form
                %Z = R.*exp(1i*theta);                

                %H = imag(optFlow);
                %V = real(optFlow);
                %M = abs(optFlow);
                
                %H = H + 127;
                %V = V + 127;
                %M = M + 127;
                %P = theta + 127;

                %imgDims = size(seqImg);
                %tmpImg = zeros(imgDims);
                %tmpImg(:,:,1) = H;
                %tmpImg(:,:,2) = V;
                %tmpImg(:,:,3) = 0;
                
                %tmpImg(tmpImg < 0) = 0;
                %tmpImg(tmpImg > 255) = 255;

                %tmpImg = tmpImg ./ 255;            

                %save optical flow image to file
                %saveFile = fullfile(saveDir,seqFiles{f});
                %imwrite(tmpImg,saveFile);
        end
            
    end
end
        
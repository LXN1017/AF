function [outfilename,fr] = AllFrameFaceDet(infilename)
%第一步：检测参考帧人脸
% Create a cascade detector object.
faceDetector = vision.CascadeObjectDetector();

vidFile =  fullfile('..\data', infilename);
outfilename = [infilename(1:end-4),'_All_Faced.avi'];
outName =  fullfile('..\result',outfilename);
vid = VideoReader(vidFile);
fr = vid.FrameRate;
len = vid.NumberOfFrames;
% temp = struct('cdata', zeros(vidHeight, vidWidth, nChannels, 'uint8'), 'colormap', []);
% temp.cdata = read(vid, 1);
% [videoFrame,~] = frame2im(temp);
% Read a video frame and run the detector.
videoFileReader = vision.VideoFileReader(vidFile);
videoFrame      = step(videoFileReader);
bbox            = step(faceDetector, videoFrame);
center = zeros(len,2);
center(1,:) = bbox(1:2)+floor(bbox(3:4)/2);
size = bbox(3:4);
% Draw the returned bounding box around the detected face.
boxInserter  = vision.ShapeInserter('BorderColor','Custom',...
    'CustomBorderColor',[255 255 0]);
videoOut = step(boxInserter, videoFrame,bbox);
figure, imshow(videoOut), title('Detected face');
drawnow
%%输出文件创建
vidOut = VideoWriter(outName);
vidOut.FrameRate = fr;
open(vidOut)
%%第一帧图像裁剪
faceImage    = imcrop(videoFrame,bbox);
writeVideo(vidOut,im2uint8(faceImage));
n=1
%第二步：检测其他帧人脸
while ~isDone(videoFileReader)
    n=n+1
    % Extract the next video frame
    videoFrame = step(videoFileReader);
    bbox = step(faceDetector, videoFrame);
    center(n,:) = bbox(1:2)+floor(bbox(3:4)/2);
    bbox(3:4) = size;
    %bbox(1:2) = center(n,:) - floor(bbox(3:4)/2);
  
    % Insert a bounding box around the object being tracked
   % videoOut = step(boxInserter, videoFrame, bbox);
    faceImage    = imcrop(videoFrame,bbox);
    writeVideo(vidOut,im2uint8(faceImage));

end

% Release resources
close(vidOut);
release(videoFileReader);
clear vid;
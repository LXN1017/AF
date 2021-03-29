function [outfilename,fr] = FirstFrameFaceDet(infilename,handles)
%第一步：检测参考帧人脸
% Create a cascade detector object.
faceDetector = vision.CascadeObjectDetector();
% Read a video frame and run the detector.
vidFile =  fullfile('data', infilename);
outfilename = [infilename(1:end-4),'_1stFaced.avi'];
outName =  fullfile('result',outfilename);
vid = VideoReader(vidFile);
fr = round(vid.FrameRate);
len = vid.NumberOfFrames;
% temp = struct('cdata', zeros(vidHeight, vidWidth, nChannels, 'uint8'), 'colormap', []);
% temp.cdata = read(vid, 1);
% [videoFrame,~] = frame2im(temp);
videoFileReader = vision.VideoFileReader(vidFile);
videoFrame      = step(videoFileReader);
bbox            = step(faceDetector, videoFrame);
center = zeros(1,2);
center(1,:) = bbox(1:2)+floor(bbox(3:4)/2);
% Draw the returned bounding box around the detected face.
boxInserter  = vision.ShapeInserter('BorderColor','Custom',...
    'CustomBorderColor',[255 255 0]);
videoOut = step(boxInserter, videoFrame,bbox);
%figure,imshow(videoOut), title('Detected face');
    %%显示人脸检测前后图像%%
    axes(handles.axes3);
    imshow(videoFrame);
drawnow
%%输出文件创建
vidOut = VideoWriter(outName);
vidOut.FrameRate = fr;
open(vidOut)
%%第一帧图像裁剪
faceImage    = imcrop(videoFrame,bbox);
    axes(handles.axes4);
    imshow(faceImage);
writeVideo(vidOut,im2uint8(faceImage));
h=waitbar(0,'开始人脸检测...','Name','正在人脸跟踪...');
%第二步：裁剪其他帧人脸
% Create a video player object for displaying video frames.
% videoInfo    = info(videoFileReader);
% videoPlayer  = vision.VideoPlayer('Position',[300 300 videoInfo.VideoSize+30]);
n=1;
% Track the face over successive video frames until the video is finished.
while ~isDone(videoFileReader)
    n=n+1;
    % Extract the next video frame
    videoFrame = step(videoFileReader);
    % Insert a bounding box around the object being tracked
    % videoOut = step(boxInserter, videoFrame, bbox);
    faceImage    = imcrop(videoFrame,bbox);
    % Display the annotated video frame using the video player object
    % step(videoPlayer, faceImage);
    writeVideo(vidOut,im2uint8(faceImage));
    h=waitbar(0.05+n*(0.85/len),h,[num2str(floor(100*(0.05+n*(0.85/len)))),'%']);
end

% Release resources
close(vidOut);
h=waitbar(0.9,h,[num2str(90),'%']);
release(videoFileReader);
h=waitbar(1,h,[num2str(1),'%']);
clear vid;
close(h)
 axes(handles.axes3);
    cla
 axes(handles.axes4);
    cla
drawnow
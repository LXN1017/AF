function [outName,cankao] = faceRec_DRMF1(infilename)
%addpath(genpath('.'));
% Create a cascade detector object.
infilename=[infilename,'.avi'];
vidFile =  fullfile('.\data', infilename);
outfilename = [infilename(1:end-4),'_DRMF_Faced.avi'];
outName =  fullfile('.\result',outfilename);
vid = VideoReader(vidFile);
vidHeight = vid.Height;
vidWidth = vid.Width;
nChannels = 3;
fr = vid.FrameRate;
len = vid.NumberOfFrames;
temp = struct('cdata', zeros(vidHeight, vidWidth, nChannels, 'uint8'), 'colormap', []);

%%输出文件创建
vidOut = VideoWriter(outName);
vidOut.FrameRate = fr;
open(vidOut)

%抠出角落作为参考信号
startIndex=31;
endIndex=len;
k=0;
cankaotu=zeros(60,endIndex-startIndex+1,3);
for i=startIndex:endIndex
    temp.cdata = read(vid, i); 
    [rgbframe,~] = frame2im(temp);   
    k=k+1;
    rgbframe = im2double(rgbframe);
    gao=60;
    kuan=100;
    cankaotu(:,k,:)=mean(rgbframe(1:gao,1:kuan,:),2);
end
    cankao=mean(cankaotu);
%     figure;
%     plot(cankao(1,:,3));
%     title('背景光照信号');   
%%第一帧图像裁剪
faceDetector = vision.CascadeObjectDetector();
videoFileReader = vision.VideoFileReader(vidFile);
videoFrame      = step(videoFileReader);
bbox            = step(faceDetector, videoFrame);


% Draw the returned bounding box around the detected face.
videoshow = insertShape(videoFrame, 'Rectangle', bbox);
 figure; 
imshow(videoshow); title('Detected face');
drawnow
faceImage    = imcrop(videoFrame,bbox);
writeVideo(vidOut,im2uint8(faceImage));

% Convert the first box into a list of 4 points
% This is needed to be able to visualize the rotation of the object.
bboxPoints = bbox2points(bbox(1, :));  %将第一个框转换成四点坐标，为了计算下面的旋转角度
% Detect feature points in the face region.
%points = detectMinEigenFeatures(rgb2gray(videoFrame), 'ROI', bbox);

% Display the detected points.
% figure, imshow(videoFrame), hold on, title('Detected features');
% plot(points);
bbox_method = 1;
visualize = 0;
n=1;
str = int2str(n);
data.name = str;
data.img = im2double(videoFrame);    
data.bbox = []; % Face Detection Bounding Box [x;y;w;h]
data.points = []; % MAT containing 66 Landmark Locations
data.pose = []; % POSE information [Pitch;Yaw;Roll]
clm_model='model/DRMF_Model.mat';
load(clm_model); 
pause off
data=DRMF(clm_model,data,bbox_method,visualize);   
points = data.points;
% Create a point tracker and enable the bidirectional error constraint to
% make it more robust in the presence of noise and clutter.
pointTracker = vision.PointTracker('MaxBidirectionalError', 2);

% Initialize the tracker with the initial point locations and the initial
% video frame.
%points = points.Location;
initialize(pointTracker, points, videoFrame);
videoPlayer  = vision.VideoPlayer('Position',...
    [100 100 [size(videoFrame, 2), size(videoFrame, 1)]+30]);
% Make a copy of the points to be used for computing the geometric
% transformation between the points in the previous and the current frames
oldPoints = points;
[l,h,z] = size(videoshow);
while ~isDone(videoFileReader)
    % get the next frame
    n=n+1;
    videoFrame = step(videoFileReader);

    % Track the points. Note that some points may be lost.
    [points, isFound] = step(pointTracker, videoFrame);
    visiblePoints = points(isFound, :);
    oldInliers = oldPoints(isFound, :);

    if size(visiblePoints, 1) >= 2 % need at least 2 points

        % Estimate the geometric transformation between the old points
        % and the new points and eliminate outliers
        [xform, oldInliers, visiblePoints] = estimateGeometricTransform(...
            oldInliers, visiblePoints, 'similarity', 'MaxDistance', 4);

        % Apply the transformation to the bounding box points
        bboxPoints = transformPointsForward(xform, bboxPoints);
        [centerX,centerY,theta]= node( bboxPoints(1,:),bboxPoints(3,:),bboxPoints(2,:),bboxPoints(4,:));
        [videoFrame,bbox_rot]= prerotate(videoFrame,theta,centerX,centerY,bbox,bboxPoints);
        videoFrame_rot=imrotate(videoFrame,theta,'bilinear','crop');
        bboxnew = bbox;
        [Y,X,~]=size(videoFrame_rot);
        bboxnew(1:2) = round([X Y]/2-bbox(3:4)/2);%%确定裁剪人脸的起点，这里的取整待讨论
        % Insert a bounding box around the object being tracked
        faceImage    = imcrop(videoFrame_rot,bboxnew);
        writeVideo(vidOut,im2uint8(faceImage));
        %%扩展videoFrame_rot已同一尺寸显示
        videoFrame_show = ones(l,h,z);
        SX = bbox_rot(1);  SY = bbox_rot(2);
        videoFrame_show(SY:SY+Y-1,SX:SX+X-1,:)=videoFrame_rot;
        bboxshow(1,1:2) = bboxnew(1,1:2) + bbox_rot(1,1:2);
        bboxshow(1,3:4) =  bboxnew(1,3:4);

        %%在videoFrame_show上画矩形并显示
        bboxnewPoints = bbox2points(bboxshow(1, :));
        bboxPolygon = reshape(bboxnewPoints', 1, []);
        videoFrame_show = insertShape(videoFrame_show, 'Polygon', bboxPolygon, ...
            'LineWidth', 1);

        % Display tracked points
        videoFrame_show = insertMarker(videoFrame_show, visiblePoints, '+', ...
            'Color', 'red');

        % Reset the points
        oldPoints = visiblePoints;
        setPoints(pointTracker, oldPoints);
    end

    % Display the annotated video frame using the video player object
    step(videoPlayer, videoFrame_show);
end
% Clean up
close(vidOut);
release(videoFileReader);
release(videoPlayer);
release(pointTracker);


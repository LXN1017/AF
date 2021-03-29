function [outfilename,fr] = faceRec_DRMF(infilename,handles)
addpath(genpath('.'));
% Create a cascade detector object.
faceDetector = vision.CascadeObjectDetector();
faceDetector_nose = vision.CascadeObjectDetector('Nose');
vidFile =  fullfile(infilename);
outfilename = [infilename(1:end-4),'_DRMF_Faced.avi'];
outName =  fullfile('.\result',outfilename);
outName1 =  fullfile('.\result',['1' outfilename]);
vid = VideoReader(vidFile);
fr = round(vid.FrameRate);

% Read a video frame and run the face detector.
videoFileReader = vision.VideoFileReader(vidFile);
videoFrame      = step(videoFileReader);
%%
%选择
%bbox            = step(faceDetector, videoFrame);
%bbox=face1(videoFrame);
rebbox            = step(faceDetector, videoFrame);
rebbox_nose            = step(faceDetector_nose, videoFrame);
[bx by] = size(rebbox);
[bxn byn] = size(rebbox_nose);
if(bx>=1)
    %人脸有两个，选取框大的。
    [Pa Pb]=find(rebbox(:,3:4)>150);
    bbox = rebbox(Pa(1),:);
else if(bxn == 1)
        %鼻子只有一个。
        bbox = [rebbox_nose(1)-100 rebbox_nose(2)-100 300 300];
    else
        rebbox1 = face1(videoFrame);
        [bx1 by1] = size(rebbox1);
        if(bx1>1)
            [Pa1 Pb1]=find(rebbox1(:,3:4)>160);
            bbox = rebbox1(Pa1(1),:);
        else
            bbox=rebbox1; 
        end
    end 
end


%%


% Draw the returned bounding box around the detected face.
videoshow = insertShape(videoFrame, 'Rectangle', bbox);
 figure; 
imshow(videoshow); title('Detected face');
drawnow
%%输出文件创建
vidOut = VideoWriter(outName);
vidOut.FrameRate = fr;
open(vidOut)
vidOut1 = VideoWriter(outName1);
vidOut1.FrameRate = fr;
open(vidOut1)
%%第一帧图像裁剪
faceImage    = imcrop(videoFrame,bbox);
writeVideo(vidOut,im2uint8(faceImage));

% Convert the first box into a list of 4 points
% This is needed to be able to visualize the rotation of the object.
bboxPoints = bbox2points(bbox(1, :));
% Detect feature points in the face region.
%points = detectMinEigenFeatures(rgb2gray(videoFrame), 'ROI', bbox);

% Display the detected points.
% figure, imshow(videoFrame), hold on, title('Detected features');
% plot(points);
bbox_method = 2;
visualize = 0;
n=1;
str = int2str(n);
data.name = str;
data.img = im2double(videoFrame);    
data.bbox = bbox; % Face Detection Bounding Box [x;y;w;h]
data.points = []; % MAT containing 66 Landmark Locations
data.pose = []; % POSE information [Pitch;Yaw;Roll]
clm_model='model/DRMF_Model.mat';
load(clm_model);
%close figure 2;
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
%     if(n>600)
%         break;
%     end
    videoFrame = step(videoFileReader);
    a=videoFrame;

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
        if(Y >= bboxnew(4)+1)
        faceImage    = imcrop(videoFrame_rot,bboxnew);
        writeVideo(vidOut,im2uint8(faceImage));
        end
        %%扩展videoFrame_rot已同一尺寸显示
        videoFrame_show = ones(l,h,z)*0.5;
        SX = bbox_rot(1);  SY = bbox_rot(2);
        videoFrame_show=a;
       % videoFrame_show(SY:SY+Y-1,SX:SX+X-1,:)=videoFrame_rot;
        bboxshow(1,1:2) = bboxnew(1,1:2) + bbox_rot(1,1:2);
        bboxshow(1,3:4) =  bboxnew(1,3:4);

        %%在videoFrame_show上画矩形并显示
        bboxnewPoints = bbox2points(bboxshow(1, :));
        bboxPolygon = reshape(bboxnewPoints', 1, []);
        videoFrame_show = insertShape(videoFrame_show, 'Polygon', bboxPolygon, ...
            'LineWidth', 3,'color','red');

        % Display tracked points
        videoFrame_show = insertMarker(videoFrame_show, visiblePoints, '+', ...
            'Color', 'green','size',3);

        % Reset the points
        oldPoints = visiblePoints;
        setPoints(pointTracker, oldPoints);
        writeVideo(vidOut1,im2uint8(videoFrame_show));
    end

    % Display the annotated video frame using the video player object
    step(videoPlayer, videoFrame_show);
end

% Clean up
close(vidOut);
close(vidOut1);
release(videoFileReader);
clear vid;
release(videoPlayer);
release(pointTracker);
%  axes(handles.axes3);
%     cla
%  axes(handles.axes4);
%     cla
% drawnow

function [faceImage,bboxnew] = prerotate(I,theta,X,Y,bbox,bboxpoints)
%以X，Y为中心点截取较大的图片
%I:input image
%theta:single
%X,Y:center point
%l:ROI的边长
bboxnew = zeros(1,4);
%l = round(bbox(1,3)*0.04);
if abs(theta) <=1
   if mod(bbox(1,3),2) == 0
       bboxnew(1,3) = bbox(1,3) + 10;
   else
       bboxnew(1,3) = bbox(1,3) + 11;
   end
   if mod(bbox(1,4),2) == 0
       bboxnew(1,4) = bbox(1,4) + 10;
   else
       bboxnew(1,4) = bbox(1,4) + 11;
   end
   bboxnew(1:2) = round([X Y]-bboxnew(3:4)/2);
   faceImage    = imcrop(I,bboxnew);
else if abs(theta) <=5
   % dis = round(sqrt(bbox(1,3)^2+bbox(1,4)^2));
       if mod(bbox(1,3),2) == 0
           bboxnew(1,3) = bbox(1,3) + 40;
       else
           bboxnew(1,3) = bbox(1,3) + 41;
       end
       if mod(bbox(1,4),2) == 0
           bboxnew(1,4) = bbox(1,4) + 40;
       else
           bboxnew(1,4) = bbox(1,4) + 41;
       end
       bboxnew(1:2) = round([X Y]-bboxnew(3:4)/2);
       faceImage    = imcrop(I,bboxnew);
    else
       bboxnew(1,3) = min([round(X),round(640-X),round(Y),round(480-Y)])*2-2;
       bboxnew(1,4) = bboxnew(1,3);
       bboxnew(1:2) = round([X Y]-bboxnew(3:4)/2);
       faceImage    = imcrop(I,bboxnew); 
    end
end

    
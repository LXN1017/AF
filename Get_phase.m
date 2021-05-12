%this code is to get the phase representation of a video 

vid = VideoReader('example.avi');
 vidHeight = vid.Height;
 vidWidth = vid.Width;
 nChannels = 3;
 fr = vid.FrameRate;
 len = vid.NumberOfFrames;
 temp = struct('cdata', zeros(vidHeight, vidWidth, nChannels, 'uint8'), 'colormap', []); 
 startIndex = 1;   %起始帧
 endIndex =len; 

tem_phase=zeros(vidHeight,vidWidth,len);
for i=startIndex:1:endIndex
       temp.cdata = read(vid, i);
       [rgbframe,~] = frame2im(temp);
       rgbframe = im2double(rgbframe);   
       
       fre=fft2(rgbframe(:,:,2));
       fre_nor=fre./abs(fre);
       tem=ifft2(fre_nor);
       tem_phase(:,:,i)=tem;  %时域相位
 end    

% tem_phase=tem_phase-min(min(min(tem_phase)));
% tem_phase=tem_phase/max(max(max(tem_phase)));
% aviobj=VideoWriter('diver.avi');
% aviobj.FrameRate = 30;
% open(aviobj)
% for i=1:330
%     frame=tem_phase(:,:,i);
%     writeVideo(aviobj,frame);
% end
% close(aviobj)
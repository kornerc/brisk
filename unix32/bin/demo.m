% Brisk Demo: showing off the Matlab interface
% Written by Stefan Leutenegger, 08/2011

% Initialize look-up tables etc.
brisk('init','threshold',60,'octaves',4);

% measure time:
tic;

% Load a first image
im1=imread('../../images/img1.ppm');
brisk('loadImage',im1);
% detect keypoints
keypoints=brisk('detect');
% get the descriptors
[keypoints1, descriptors1]=brisk('describe');

% Load the second image
im2=imread('../../images/img2.ppm');
brisk('loadImage',im2);
% detect keypoints
brisk('detect');
% get the descriptors
[keypoints2, descriptors2]=brisk('describe');

% Match them
matches=brisk('radiusMatch',descriptors1,descriptors2,90);

% measure time:
toc;

% free memory
brisk('terminate');

% show
if size(im1,1)>size(im2,1)
    imshow([im1 [im2; zeros(size(im1,1)-size(im2,1),size(im2,2),3)]]);
else
    if size(im1,1)<size(im2,1)
        imshow([[im1; zeros(size(im2,1)-size(im1,1),size(im1,2),3)] im2]);
    else
        imshow([im1 im2]);
    end
end
hold on;
dx=size(im1,2); %offset to right image
% keypoints in the first image: red circles
for i=1:size(keypoints1,1)
    radius=keypoints1(i,3)/2;
    [x,y,z] = cylinder(radius,20);
    plot(keypoints1(i,1)+x(1,:),keypoints1(i,2)+y(1,:),'r')
    % direction line
    angle=keypoints1(i,4)/180*pi;
    radial=radius*[cos(angle),sin(angle)];
    plot([keypoints1(i,1);keypoints1(i,1)+radial(1)],...
        [keypoints1(i,2);keypoints1(i,2)+radial(2)],'r');
end
% keypoints in the second image: red circles
for i=1:size(keypoints2,1)
    radius=keypoints2(i,3)/2;
    [x,y,z] = cylinder(radius,20);
    plot(keypoints2(i,1)+x(1,:)+dx,keypoints2(i,2)+y(1,:),'r');
    % direction line
    angle=keypoints2(i,4)/180*pi;
    radial=radius*[cos(angle),sin(angle)];
    plot([keypoints2(i,1)+dx;keypoints2(i,1)+radial(1)+dx],...
        [keypoints2(i,2);keypoints2(i,2)+radial(2)],'r');
end
% matches
for i=1:size(matches,1)
    if matches(i,1)~=0
        j=matches(i,1);
        radius1=keypoints1(i,3)/2;
        radius2=keypoints2(j,3)/2;
        % green circles
        [x1,y1,z1] = cylinder(radius1,20);
        [x2,y2,z2] = cylinder(radius2,20);
        plot(keypoints1(i,1)+x1(1,:),keypoints1(i,2)+y1(1,:),'g')
        plot(keypoints2(j,1)+x2(1,:)+size(im1,2),...
            keypoints2(j,2)+y2(1,:),'g')
        % direction lines
        angle1=keypoints1(i,4)/180*pi;
        radial1=radius1*[cos(keypoints1(i,4)),sin(keypoints1(i,4))];
        angle2=keypoints2(j,4)/180*pi;
        radial2=radius2*[cos(keypoints2(j,4)),...
            sin(keypoints2(j,4))];
        plot([keypoints1(i,1);keypoints1(i,1)+radial1(1)],...
            [keypoints1(i,2);keypoints1(i,2)+radial1(2)],'g');
        plot([keypoints2(j,1)+dx;keypoints2(j,1)+radial2(1)+dx],...
            [keypoints2(j,2);keypoints2(j,2)+radial2(2)],'g');
        % green match line
        plot([keypoints1(i,1);keypoints2(matches(i,1),1)+size(im1,2)],...
            [keypoints1(i,2);keypoints2(matches(i,1),2)],'g');
    end
end
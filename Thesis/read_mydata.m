function [ mydata ] = read_mydata_action3d_general(folder)
% Reads all skeletons in the folder and returns them in a mydata object:
% mydata{,1} = actionnumber
% mydata{,2} = subject number
% mydata{,3} = repetition number
% mydata{,4} = nframes
% mydata{,5} = skeleton array (20*330*8 = JOINTS * nframes * XYZ)
% mydata{,6} = feature vectors (for future computation)

% Index all available Skeletons
addpath(folder)

ff = fopen('JiangExperimentFileList.txt');
gg = textscan(ff,'%q');
index_list = gg{1};
N=size(index_list,1);

mydata=cell(N,7);
for i=1:N
     myskeleton=strcat(index_list{i},'_skeleton3D.txt');
    
    % Store action, subject and repetition number
    mydata{i,1} = str2num(myskeleton(2:3));
    %if mydata{i,1} == 11
    %    mydata{i,1} = 8;
    %end
    mydata{i,2} = str2num(myskeleton(6:7));
    mydata{i,3} = str2num(myskeleton(10:11));
    
    % Open file
    fp = fopen(myskeleton);
    B = fscanf(fp,'%f') ;
    
    l=size(B,1)/4;
    B2 = reshape(B,4,l);
    B2 = B2';
    B3 = reshape(B2,20,l/20,4);
    mydata{i,4} = size(B3,2);
    mydata{i,5} = B3;
end

% Remove elements with less than 3 repetitions
check = zeros(20,10);
for i=1:20
    for j=1:10
        ind = (cell2mat(mydata(:,1))==i) & (cell2mat(mydata(:,2))==j);
        check(i,j) = sum(ind);
        if check(i,j) ~= 3
            mydata(ind,:) = [];
        end        
    end
end

% Throw away some malicious videos
N=size(mydata,1);
set1 = [1:3:N ; 2:3:N ; 3:3:N];
throwaway = [set1(:,[2 36 101, 109, 116,117,145,155,175])];
throw1 = mydata(throwaway(:),:);
mydata(throwaway(:),:) = [];
N=size(mydata,1);
set1 = [1:3:N ; 2:3:N ; 3:3:N];
throwaway2 = [set1(:,[69,79,89,107,11,157,158])];
throw2 = mydata(throwaway2(:),:);
mydata(throwaway2(:),:) = [];
N=size(mydata,1);
set1 = [1:3:N ; 2:3:N ; 3:3:N];
throwaway3 = [set1(:,[53,97])];
throw3 = mydata(throwaway2(:),:);
mydata(throwaway3(:),:) = [];

% Remove ambiguous action classes
for i=[1,4,5,7,8] 
mydata((cell2mat(mydata(:,1))==i),:) = [];
end

end


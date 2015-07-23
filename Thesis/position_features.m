function [ mydata ] = position_features(mydata,center, rotated,rescale,head,angles,legs)
% Function to derive feature vectors from raw skeleton sequences
% Raw sequences stores in mydata{i,5}
% Processed vectors in mydata{i,6}

% OPTIONS:
% normalize = TRUE normalizes the body lenghts
% head = TRUE adds the head coordinates to the feature vector
% angles = TRUE adds the yaw,pitch,roll angles of the torso to feature
% legs = TRUE tries to detect and correct ill leg extraction

N = size(mydata,1);

for j=1:N 
    % Select video & initialize storage
    video = mydata{j,5};
    nframe = size(video,2);
    
    P=24;
    if head
        P = P+3;
    end
    if angles
        P = P+3;
    end
    out = zeros(P,nframe);
    repos_skelet = zeros(20,nframe,3);
    
    for i=1:nframe
        skelet = squeeze(video(:,i,1:3)); % pick frame
        
        % Mean center
        if center
        new_origin = mean(skelet(1:7,1:3)); 
        skelet = skelet - repmat(new_origin,20,1);
        end
        
        if rotated | angles
        % Rotation
        [~,~,axes1]=svd(skelet(1:7,1:3));
            if dot(axes1(:,1),(skelet(3,:)-skelet(7,:)))<0 % Reorientate
                axes1(:,1) = (-1)*axes1(:,1);
            end
            if dot(axes1(:,2),(skelet(2,:)-skelet(1,:)))<0 
                axes1(:,2) = (-1)*axes1(:,2);
            end
            axes1(:,3) = cross(axes1(:,2),axes1(:,1));
            axes1 = axes1(:,[2 1 3]); % correct rotation matrix
        
            if rotated
                skelet = ((axes1^-1)*(skelet'))'; % Reposition skeleton
            end
            if angles
                [r1 r2 r3] = dcm2angle(axes1,'YZX');
            end
        end
 
        % Sometimes the legs are confused by the tracker: correct this.
        if legs
        if (skelet(15,1) < skelet(14,1)) & abs(skelet(15,1) - skelet(14,1))> 0.3*abs(skelet(6,1) - skelet(5,1))
            switch_store = skelet([15 17 19],1);
            skelet([15 17 19],1) = skelet([14 16 18],1);
            skelet([14 16 18],1) = switch_store;
        end
        end
        
        % Store the repositioned skeleton, for possible plotting
        repos_skelet(:,i,:) = skelet;
 
        % Built normalized skeleton:
        if rescale
        newskelet = zeros(16,3);
        newskelet(1,:) = [-0.2, 0.3, 0];
        newskelet(2,:) = [0.2, 0.3, 0];
        newskelet(3,:) = [0, 0.3, 0];
        newskelet(4,:) = [0, 0, 0];
        newskelet(5,:) = [0, -0.25, 0];
        newskelet(6,:) = [-0.15, -0.4, 0];
        newskelet(7,:) = [0.15, -0.4, 0];
        newskelet(8,:) = 0.25*((skelet(20,:)-skelet(3,:))/norm(skelet(20,:)-skelet(3,:)))+skelet(3,:);
        newskelet(9,:) = 0.25*((skelet(8,:)-skelet(1,:))/norm(skelet(8,:)-skelet(1,:)))+skelet(1,:);
        newskelet(10,:) = 0.25*((skelet(10,:)-skelet(8,:))/norm(skelet(10,:)-skelet(8,:)))+newskelet(9,:);
        newskelet(11,:) = 0.25*((skelet(9,:)-skelet(2,:))/norm(skelet(9,:)-skelet(2,:)))+skelet(2,:);
        newskelet(12,:) = 0.25*((skelet(11,:)-skelet(9,:))/norm(skelet(11,:)-skelet(9,:)))+newskelet(11,:);
        newskelet(13,:) = 0.45*((skelet(14,:)-skelet(5,:))/norm(skelet(14,:)-skelet(5,:)))+skelet(5,:);
        newskelet(14,:) = 0.45*((skelet(16,:)-skelet(14,:))/norm(skelet(16,:)-skelet(14,:)))+newskelet(13,:);
        newskelet(15,:) = 0.45*((skelet(15,:)-skelet(6,:))/norm(skelet(15,:)-skelet(6,:)))+skelet(6,:);
        newskelet(16,:) = 0.45*((skelet(17,:)-skelet(15,:))/norm(skelet(17,:)-skelet(15,:)))+newskelet(15,:);
        end
        
        % Built feature vector
        if ~rescale
            x = [skelet([ 8 10 9 11 14 16 15 17],1);skelet([ 8 10 9 11 14 16 15 17],2);skelet([ 8 10 9 11 14 16 15 17],3)];
            if head
                x = [x ; skelet(20,:)'];
            end
            if angles
                x = [x ; r3 ; r2 ; r1];
            end
        else
            x = [newskelet(9:16,1); newskelet(9:16,2); newskelet(9:16,3)];
            if head
                x = [x ; newskelet(8,:)'];
            end
            if angles
                x = [x ; r3 ; r2 ; r1];
            end
        end
        
        out(:,i) = x;

        % Store frame skeleton as feature vector in video matrix
        %out(:,i) = 
        %out2(:,i) = [skelet([ 8 10 9 11 14 16 15 17 20],1);skelet([ 8 10 9 11 14 16 15 17 20],2);skelet([ 8 10 9 11 14 16 15 17 20],3)];
        %out2(:,i) = [skelet([ 8 10 9 11 14 16 15 17 20],1);skelet([ 8 10 9 11 14 16 15 17 20],2);skelet([ 8 10 9 11 14 16 15 17 20],3); r3; r2; r1];
        %out2(:,i) = [skelet([ 8:20],1);skelet([ 8:20],2);skelet([8:20],3); r3; r4];
    end
    
    % Store video feature matrix
    away = sum(isnan(out))>0;
    away2 = sum(out == 0)>0;
    away = logical(away) | logical(away2);
    out(:,logical(away))=[];
    mydata{j,4} = size(out,2);
    mydata{j,6} = out;
    mydata{j,7} = repos_skelet;
    if rescale
        mydata{j,8} = newskelet;
    end
end

end


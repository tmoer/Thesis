function[] = plot_skeleton(mydata,action,pauze,raw)
% General video plotting function
% 'action' is the action number to be plotted
% 'pauze' is the break between consequetive frames
% 'rotation' indicates whether the raw or processed (rotated and centered)
% skeletons have to be used

% True joint connections
J=[20     1     2     1     8    10     2     9    11     3     4     7     7     5     6    14    15    16    17;
    3     3     3     8    10    12     9    11    13     4     7     5     6    14    15    16    17    18    19];

% Select action sequences belonging to correct action number
if raw
    X=mydata(logical(cell2mat(mydata(:,1))==action),5);
    fprintf('Raw skeleton sequences');
else
    X=mydata(logical(cell2mat(mydata(:,1))==action),7);
    fprintf('Processed skeleton sequences');
end
n=size(X,1);

% Plot skeletons
for p=1:n
    fprintf('Sequence %d for action %d \n',p,action);
    Y=X{p}; % select array
    for s=1:size(Y,2);
        S=squeeze(Y(:,s,1:3));
        h=plot3(S(:,1),S(:,3),S(:,2),'r.','Linewidth',2);
        %rotate(h,[0 45], -180);
        if ~raw
            axis([-0.75 0.75 -1 1 -1.25 0.75])
        else
            axis([-0.75 0.75 2.5 3 -1.5 0.5])
        end
        set(gca,'DataAspectRatio',[1 1 1]);
        set(gca,'YTickLabel',[]);
        set(gca,'XTickLabel',[]);
        set(gca,'ZTickLabel',[]);

        for j=1:19
            c1=J(1,j);
            c2=J(2,j);
            line([S(c1,1) S(c2,1)], [S(c1,3) S(c2,3)], [S(c1,2) S(c2,2)],'Linewidth',2);
        end
        pause()
    end
end
end
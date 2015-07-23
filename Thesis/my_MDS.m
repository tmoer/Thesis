function [out] = my_MDS( LL_distance1, pred_table_novel1 )
% Create MDS plots for all novelty splits in pred_table_novel1 
out = [];

for i=1:size(pred_table_novel1,1)/366
    figure()
    pred = pred_table_novel1((i-1)*366+(1:366),:,8);
    LL = LL_distance1(:,(i-1)*366+(1:366),8);
    new = pred(:,2) == 100;
    label = pred(new,7);
    LL = LL(:,new);
    LL(LL==-Inf) = -100;
    D = squareform(pdist(LL','euclidean'));
    [coor e] = cmdscale(D);
    leftout = unique(pred(pred(:,1)==100,7));
    hold on
    h1 = plot(coor(label == leftout(1),1),coor(label == leftout(1),2),'r+','markers',10,'Linewidth',4);
    h3 = plot(coor(label == leftout(3),1),coor(label == leftout(3),2),'bx','markers',10,'Linewidth',4);
    h2 = plot(coor(label == leftout(2),1),coor(label == leftout(2),2),'g*','markers',10,'Linewidth',4);
    h4 = plot(coor(~ismember(label,leftout),1),coor(~ismember(label,leftout),2),'k.','markers',12);
    title('Multidimensional scaling')
end

end


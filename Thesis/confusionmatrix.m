function [ error ] = confusionmatrix(truth, class)
% Confusionmatrix functionality, modified from 
% http://stackoverflow.com/questions/21215352/matlab-confusion-matrix

mat = confusionmat(truth,class);
error = sum(diag(mat))/sum(sum(mat));
mat = mk_stochastic(mat);  
imagesc(mat);            %# Create a colored plot of the matrix values
colormap(flipud(gray));  %# Change the colormap to gray (so higher values are
                         %#   black and lower values are white)

textStrings = num2str(mat(:),'%0.2f');  %# Create strings from the matrix values
textStrings = strtrim(cellstr(textStrings));  %# Remove any space padding

% Remove 0.00
idx = find(strcmp(textStrings(:), '0.00'));
textStrings(idx) = {'   '};

% Make plot
[x,y] = meshgrid(1:15);   %# Create x and y coordinates for the strings
hStrings = text(x(:),y(:),textStrings(:),...      %# Plot the strings
                'HorizontalAlignment','center');
midValue = mean(get(gca,'CLim'));  %# Get the middle value of the color range
textColors = repmat(mat(:) > midValue,1,3);  %# Choose white or black for the
                                             %#   text color of the strings so
                                             %#   they can be easily seen over
                                             %#   the background color
set(hStrings,{'Color'},num2cell(textColors,2));  %# Change the text colors

% Add labels
set(gca, 'XAxisLocation', 'top')
set(gca,'XTick',1:15,...                         %# Change the axes tick marks
        'XTickLabel',strread(num2str(1:15),'%s')',...  %#   and tick labels
        'YTick',1:15,...
        'YTickLabel',strread(num2str(1:15),'%s')',...
        'TickLength',[0 0]);

end


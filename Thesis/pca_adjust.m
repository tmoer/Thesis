function [ train, test, train_new ] = pca_adjust(train, test, dim, plot_it, train_new)
% Calculates PCA of first 'dim' dimensions based on train
% Returns train and test with adjusted dimensionalities

featuremat = cell2mat(train(:,6)')';
N = size(featuremat,1);

% Mean center
means = mean(featuremat);
featuremat = featuremat - repmat(means,N,1);

% Perform PCA
[coefs,scores,variances] = princomp(featuremat);
prin_axes = coefs(:,1:dim);
if plot_it
    figure()
    plot(variances./sum(variances),'LineWidth',3);
    xlabel('Principal component')
    ylabel('Variance explained (%)')
end

% Adjust train and test dataset
for i=1:size(train,1)
    train{i,6} = (train{i,6}'*prin_axes)';
end
for i=1:size(test,1)
    test{i,6} = (test{i,6}'*prin_axes)';
end

if nargin == 5
    for i=1:size(train_new,1)
        train_new{i,6} = (train_new{i,6}'*prin_axes)';
    end
end

end


function [out_store] = hmm_KP(mydata, epochs, K_range, P_range, plot_it)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

N = size(mydata,1);
out_store = zeros(numel(K_range),numel(P_range),epochs*3);

for l=1:numel(K_range)
    K=K_range(l);
        for j=1:numel(P_range)
            dim= P_range(j);
            [precision pred_table opt_iter A mu Sigma ] = hmm_optimal('clustered', epochs, mydata, K, dim);
            out_store(l,j,:) = precision;
        end
end

% Make plot if desired
if plot_it
    means_store = mean(out_store,3);
    std_store = zeros(numel(K_range),numel(P_range));
    for i=1:numel(K_range)
        for j=1:numel(P_range)
            std_store(i,j) = std(squeeze(out_store(i,j,:)));
        end
    end

    h1 = errorbar(means_store(:,1),std_store(:,1),'g:','LineWidth',2);
    hold on
    ax = gca;
    ax.XLim=[0 (numel(K_range)+1)];
    ax.YLim=[0.5 1];
    ax.XTick = 1:numel(K_range);
    ax.XTickLabel = num2cell(K_range);
    h3 = errorbar(means_store(:,5),std_store(:,5),'b--','LineWidth',2);
    h2 = errorbar(means_store(:,3),std_store(:,3),'r-','LineWidth',2);
    
    xlabel('Number of keyposes (K)')
    ylabel('Test set accuracy (%)')
    legend([h3 h2 h1], 'P=30','P=10','P=3','Location','southoutside','Orientation','horizontal')
    print('K_PCA', '-dpng')
end

end


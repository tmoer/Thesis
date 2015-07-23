function [confusion_novel, tau, error] = determine_tau( pred_table, plot_it, antitype)
% Function to determine optimal tau from a prediction table

N_anti = numel(antitype);
tau = [];
confusion_novel = [];
error = [];

for i = 1:N_anti
    % Create visualization of novelty detection:
    antimodel = antitype{i};
    
    % For nicer plotting
    switch antimodel
    case 'none'
        model_name = 'None'
    case 'sum'
        model_name = 'P(S|X)'
    case 'entropy'
        model_name = 'Entropy'
    case 'filler'
        model_name = 'Filler'
    case 'flat'
        model_name = 'Flat'
    case 'anti_full'
        model_name = 'Full anti-model'
    case 'anti_matej'
        model_name = 'Reweighted anti-model'
    case 'combination'
        model_name = 'Combination of filler and anti-model'
    case 'combination2'
        model_name = 'Combination of flat and anti-model'
    end
    
    truth = pred_table(:,1,i);
    class = pred_table(:,2,i);
    value = pred_table(:,3,i);
    if strcmp(antimodel,'sum')
        value = pred_table(:,4,i);
    else if strcmp(antimodel,'entropy')
            value = pred_table(:,5,i);
        end
    end
    
    % ROC to determine optimal tau:
    %correct = (truth == class);
    correct = (truth ~= 100);
    [truepos falsepos cutoff] = roc(correct',value');
    totalerror = (1-truepos) + falsepos;
    [error1 t] = min(totalerror);
    truepick = truepos(t); falsepick = falsepos(t);
    if (t==1)
        t=2; % heuristic solution to prevent error if t is the first pick
    end
    tau1 = cutoff((t-1));
    
    % Novelty confusion table
    class(value < tau1) = 100;
    confusion = confusionmat(class == 100,truth ==100);
    precision = sum(diag(confusionmat(truth(truth~=100),class(truth~=100))));
    confusion = [precision, confusion(1,2); (confusion(1,1)-precision) 0; confusion(2,:)];
    confusion_novel1 = mk_stochastic(confusion')';

    % Built plots
    if plot_it
        f=figure();
        subplot(1,2,1);
        plot(falsepos, truepos);
        axis square; title('ROC-curve'); ylabel('true positive (known)'); xlabel('false positive (known)');
        hold on; plot([0 1],[0 1],'k:'); plot(falsepick,truepick,'pk','Linewidth',2); hold off;
        text(1.1,1.3,sprintf('Background model: %s',model_name),'HorizontalAlignment','center','VerticalAlignment', 'top')
        % text(1,-0.4,num2str(confusion_novel1,2));

        subplot(1,2,2);
        known = (truth~=100);
        [f xi] = ksdensity(value(known)); plot(xi,f); axis square; title('Histogram'); xlabel('P(X|S) - P(X|background)'); ylabel('density');lim = axis; axis(lim);
        [f xi] = ksdensity(value(~known)); hold on; plot(xi,f,'red'); legend('known','new','Location','northwest');
        plot([t t], [0 1],'k--');  hold off;
    end
    
    confusion_novel = cat(3,confusion_novel,confusion_novel1);
    error = cat(1,error,error1);
    tau = cat(1,tau,tau1);  


end





end


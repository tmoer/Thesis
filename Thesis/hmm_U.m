function [ store ] = hmm_U(U_seq, epochs, mydata, K, dim, starttype, n_EM, plot_it)
%
store = [];

for i=1:numel(U_seq)
    [train_precision test_precision pred_table train_loglik] = ...
        hmm_sequence('EBW', epochs, mydata, K, dim, starttype, n_EM, false, U_seq(i));
    store = cat(4,store,cat(3,train_precision, test_precision, train_loglik));
end


if plot_it
    precious = squeeze(mean(store(:,:,2,:),2));
    figure();
    subplot(3,1,1)
    hold on
    for i = 1:4
        plot(1:30,precious(:,i),'Linewidth',2);
    end
    title('Test error')

    precious = squeeze(mean(store(:,:,1,:),2));
    subplot(3,1,2)
    hold on
    for i = 1:4
    plot(1:30,precious(:,i),'Linewidth',2);
    end
    title('Train error')

    precious = squeeze(mean(store(:,:,3,:),2));
    subplot(3,1,3)
    hold on
    for i = 1:4
        plot(1:30,precious(:,i),'Linewidth',2);
    end
    title('Log likelihood')
    legend('U=0.1','U=1.5','U=5','U=20','Location','southoutside','Orientation','horizontal')
end


end


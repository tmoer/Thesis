function [train_precision test_precision pred_table train_loglik] ...
    = hmm_sequence(model, epochs, mydata, K, dim, starttype, n_EM, plot_it, U)

% Wrapper to assess test set hmm performance as function of number of
% iterations. 
% model: indicates which model to assess 'EM' or 'EBW'

N=size(mydata,1)
train_loglik = [];
train_precision = [];
test_precision = [];
pred_table = [];
    
for rep=1:epochs
    known = split_index(N,'normal',mydata);
    
    for i=1:3
        [ train test ] = split_set(mydata, 'normal', i, known); 
        [ train test ] = pca_adjust(train, test, dim, false); 
        [A_init, mu_init, Sigma_init, actions] = models_init(train, K, starttype);
        
        tic
        switch model
            case 'EM'
        [A mu Sigma A_store mu_store  Sigma_store loglik_store]  = train_EM(train, ...
            A_init, mu_init, Sigma_init, actions, n_EM, starttype);
            case 'EBW'
                [A, mu, Sigma, A_store, mu_store, Sigma_store, loglik_store] = train_EBW(train, A_init, ...
            mu_init, Sigma_init, actions, n_EM, starttype, U);
        end
        toc
        
        tic
        % Classify
        [ LL_frame ] = frame_lik(test, A_store, mu_store, Sigma_store);
        [pred_table1 precision1] = classify(test, actions, LL_frame);
        [ LL_frame2 ] = frame_lik(train, A_store, mu_store, Sigma_store);
        [pred_table2 precision2] = classify(train, actions, LL_frame2);
        toc
        
        % Store results of this iteration
        train_loglik = cat(2,train_loglik,loglik_store);
        train_precision = cat(2,train_precision,precision2);
        test_precision = cat(2,test_precision,precision1);
        pred_table = cat(1,pred_table,pred_table1);
    end 
end


% Make direct plot of result
if plot_it
    train_precision2 = mean(train_precision,2);
    test_precision2 = mean(test_precision,2);
    train_loglik2 = mean(train_loglik,2);
    
    f1 = figure();
    [ax h2 h1] =  plotyy(1:n_EM,train_precision2,1:n_EM,train_loglik2);
    hold on
    [h3] = plot(1:n_EM,test_precision2);
    
    set(h1,'Color','g','Linestyle','--','LineWidth',2)
    set(h2,'Color','b','Linestyle',':','LineWidth',2)
    set(h3,'Color','r','Linestyle','-','LineWidth',4)
    
    ylim(ax(1),[floor(min([train_precision2 ; test_precision2])*10)/10, ...
        ceil(max([train_precision2 ; test_precision2])*10)/10])
    xlabel('Iterations')
    ylabel(ax(1),'Recognition accuracy (%)') % left y-axis
    ylabel(ax(2),'Log likelihood') % right y-axis
    set(ax,{'ycolor'},{'k';'g'})
    legend([h3 h2 h1], 'Test error','Training error','Log-likelihood' ,'Location','southoutside','Orientation','horizontal')
    print(cat(2,'sequence',model,starttype), '-dpng')
    
end


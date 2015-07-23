function [A mu Sigma opt_iter] = train_optimal(model, train, K, starttype, max_EM, plot_it,U)
% Wrapper to identify an optimal trained model
% model indicates 'clustered' 'EM' or 'EBW'

[A_init mu_init Sigma_init actions] = models_init(train, K, starttype);        
% split up train in model and validation
made_it = 1;
while made_it
    IND = randperm(round(size(train,1)/2),round(size(train,1)/5)); % NOTE, the 5 may be added as parameter to this function
    train_validation = train(IND,:);
    train_model = train(~ismember(1:size(train,1),IND),:);
    [aaa ~] = hist(cell2mat(train_model(:,1)),actions);
    if all(aaa>=2)
        made_it = 0;
    else
        made_it = 1;
    end
end

switch model
    case 'EM'
        [A mu Sigma A_store mu_store  Sigma_store loglik_store]  = train_EM(train_model, A_init, ...
            mu_init, Sigma_init, actions, max_EM, starttype);
        
    case 'EBW'
        [A, mu, Sigma, A_store, mu_store, Sigma_store, loglik_store] = train_EBW(train_model, A_init, ...
            mu_init, Sigma_init, actions, max_EM, starttype, U); % NOTE, the 5 may be added as a parameter for the EBW model
end

% Decode    
[LL_frame] = frame_lik(train_validation, A_store, mu_store, Sigma_store);
[pred_table precision] = classify(train_validation, actions, LL_frame);
[opt_precision opt_iter] = max(precision); % optimal iteration number

if plot_it
    f3 = figure()
    plot(1:max_EM,precision)
    hold on
    plot([opt_iter opt_iter],[0 1])
    hold off
    xlabel('Number of iterations')
    ylabel('Validation error')
end


% Full model
switch model
    case 'EM'
        [A mu Sigma A_store mu_store  Sigma_store loglik_store]  = train_EM(train, A_init, ...
            mu_init, Sigma_init, actions, opt_iter, starttype);
        
    case 'EBW'
        [A, mu, Sigma, A_store, mu_store, Sigma_store, loglik_store] = train_EBW(train, A_init, ...
            mu_init, Sigma_init, actions, opt_iter, starttype, 5); % NOTE, the 5 may be added as a parameter for the EBW model
end


end


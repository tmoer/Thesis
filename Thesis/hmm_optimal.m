function [precision pred_table opt_iter A mu Sigma ] = hmm_optimal(model, epochs, mydata, K, dim, starttype, max_EM,U)
% Wrapper to assess performance of optimal model

% model can be 'clustered', 'EM
N = size(mydata,1);

A = [];
mu = [];
Sigma = [];
opt_iter = [];
precision= [];
pred_table = [];

for rep=1:epochs
    known = split_index(N,'normal',mydata); % create permuted indices for 3 fold CV
    
    for i=1:3
        tic
        
        [train test] = split_set(mydata, 'normal', i, known);
        [train test] = pca_adjust(train, test, dim, false);
        [A2, mu2, Sigma2, actions] = models_init(train, K, 'clustered');
        
        % Estimate model
        if strcmp(model,'EM')
            [A2 mu2 Sigma2 opt_iter2] = train_optimal('EM', train, K, starttype, max_EM, false);
        end
        if strcmp(model,'EBW')
            [A2 mu2 Sigma2 opt_iter2] = train_optimal('EBW', train, K, starttype, max_EM, false, U);
        end

        % Test error
        [ LL_frame ] = frame_lik(test, A2, mu2, Sigma2);
        [pred_table2 precision2] = classify(test, actions, LL_frame);
        
        % Store values of this run:
        A = cat(4,A,A2);
        mu = cat(3,mu,mu2);
        Sigma= cat(4,Sigma,Sigma2);
        precision = cat(1,precision,precision2);
        pred_table = cat(1,pred_table,pred_table2);
        
        if ~strcmp(model,'clustered')
            opt_iter = cat(1,opt_iter,opt_iter2);
        end
        
        toc
    end
end

end


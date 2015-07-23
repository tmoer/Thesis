function [A mu Sigma A_store mu_store  Sigma_store loglik_store]  = train_EM(train, A, mu, Sigma, actions, n_EM, starttype)
% Function to estimate set of HMM models with shared key postures through
% Expectation-Maximization

% Retrieve some standard parameters:
K = size(A,1);
P = size(mu,1);
Ntrain = size(train,1);
nact = size(actions,1);
video_index = cell2mat(train(:,1));
frame_index=[];
for i=1:Ntrain
    frame_index=[frame_index; repmat(train{i,1},size(train{i,6},2),1)];
end

% Set some threshold values to determine convergence
thresh = 1e-5;
previous_loglik = -Inf;
loglik = 0;
iter = 1;

% Set up storage containers for output
mu_store = [];
A_store = [];
Sigma_store = [];
loglik_store = [];

% For clustered start first iterate transition matrix few times:
if strncmpi(starttype,'c',1)
    for i=1:nact
        [~, ~, A(:,:,i), ~, ~, ~] = mhmm_em(train(logical(video_index==actions(i)),6), repmat(1,K,1)/K, ...
            A(:,:,i), mu, Sigma, repmat(1,K,1), 'max_iter', 3,'cov_type','diag','adj_mu',0,'adj_Sigma',0,'verbose',0);    
    end
end

% Run EM algorithm
while (iter <= n_EM)
    
    % Set up new storage for this iteration
    exp_num_trans = zeros(K,K,nact);
    postmix = zeros(K,1);
    m = zeros(P,K);
    op = zeros(P,P,K);
    ip = 0;
    loglik = 0;
    
    % E-step
    for i=1:nact
        [loglik_2, exp_num_trans_2, ~, postmix_2, m_2, ~, op_2] = ...
            ess_mhmm_EM(repmat(1,K,1)/K, A(:,:,i), repmat(1,K,1), mu, Sigma, train(logical(video_index==actions(i)),6));
        loglik = loglik + loglik_2;
        exp_num_trans(:,:,i) = exp_num_trans_2;
        postmix = postmix + postmix_2;
        m = m + m_2;
        op = op + op_2;
    end
    
    % M-step
    for i=1:nact
        A(:,:,i) = mk_stochastic(exp_num_trans(:,:,i));
    end
    [mu, Sigma] = mixgauss_Mstep(postmix, m, op, ip, 'cov_type', 'diag');
    %Sigma = repmat(mean(Sigma,3),[1,1,K]); % pool variances overall ()
    
    % Store
    loglik_store = cat(1,loglik_store,loglik);
    mu_store = cat(3,mu_store,mu);
    A_store = cat(4,A_store,A);
    Sigma_store = cat(4,Sigma_store,Sigma);
    
    fprintf('EM iteration %d of %d, loglik is %d\n',iter,n_EM,loglik);
    
    % Check convergence
    iter =  iter + 1;
    previous_loglik = loglik;
end

end


%%%%

function [loglik, exp_num_trans, exp_num_visits1, postmix, m, ip, op] = ...
    ess_mhmm_EM(prior, transmat, mixmat, mu, Sigma, data)
% ESS_MHMM Compute the Expected Sufficient Statistics for a MOG Hidden Markov Model.
%
% Outputs:
% exp_num_trans(i,j)   = sum_l sum_{t=2}^T Pr(Q(t-1) = i, Q(t) = j| Obs(l))
% exp_num_visits1(i)   = sum_l Pr(Q(1)=i | Obs(l))
%
% Let w(i,k,t,l) = P(Q(t)=i, M(t)=k | Obs(l))
% where Obs(l) = Obs(:,:,l) = O_1 .. O_T for sequence l
% Then 
% postmix(i,k) = sum_l sum_t w(i,k,t,l) (posterior mixing weights/ responsibilities)
% m(:,i,k)   = sum_l sum_t w(i,k,t,l) * Obs(:,t,l)
% ip(i,k) = sum_l sum_t w(i,k,t,l) * Obs(:,t,l)' * Obs(:,t,l)
% op(:,:,i,k) = sum_l sum_t w(i,k,t,l) * Obs(:,t,l) * Obs(:,t,l)'


verbose = 0;

%[O T numex] = size(data);
numex = length(data);
O = size(data{1},1);
Q = length(prior);
M = size(mixmat,2);
exp_num_trans = zeros(Q,Q);
exp_num_visits1 = zeros(Q,1);
postmix = zeros(Q,M);
m = zeros(O,Q,M);
op = zeros(O,O,Q,M);
ip = zeros(Q,M);

mix = (M>1);

loglik = 0;
if verbose, fprintf(1, 'forwards-backwards example # '); end
for ex=1:numex
  if verbose, fprintf(1, '%d ', ex); end
  %obs = data(:,:,ex);
  obs = data{ex};
  T = size(obs,2);
  if mix
    [B, B2] = mixgauss_prob(obs, mu, Sigma, mixmat);
    [alpha, beta, gamma,  current_loglik, xi_summed, gamma2] = ...
	fwdback(prior, transmat, B, 'obslik2', B2, 'mixmat', mixmat);
  else
    B = mixgauss_prob(obs, mu, Sigma);
    [alpha, beta, gamma,  current_loglik, xi_summed] = fwdback(prior, transmat, B);
  end    
  loglik = loglik +  current_loglik; 
  if verbose, fprintf(1, 'll at ex %d = %f\n', ex, loglik); end

  exp_num_trans = exp_num_trans + xi_summed; % sum(xi,3);
  exp_num_visits1 = exp_num_visits1 + gamma(:,1);
  
  if mix
    postmix = postmix + sum(gamma2,3);
  else
    postmix = postmix + sum(gamma,2); 
    gamma2 = reshape(gamma, [Q 1 T]); % gamma2(i,m,t) = gamma(i,t)
  end
  for i=1:Q
    for k=1:M
      w = reshape(gamma2(i,k,:), [1 T]); % w(t) = w(i,k,t,l)
      wobs = obs .* repmat(w, [O 1]); % wobs(:,t) = w(t) * obs(:,t)
      m(:,i,k) = m(:,i,k) + sum(wobs, 2); % m(:) = sum_t w(t) obs(:,t)
      op(:,:,i,k) = op(:,:,i,k) + wobs * obs'; % op(:,:) = sum_t w(t) * obs(:,t) * obs(:,t)'
      ip(i,k) = ip(i,k) + sum(sum(wobs .* obs, 2)); % ip = sum_t w(t) * obs(:,t)' * obs(:,t)
    end
  end
end
if verbose, fprintf(1, '\n'); end


end


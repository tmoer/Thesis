function [A, mu, Sigma, A_store, mu_store, Sigma_store, loglik_store] = train_EBW(train, A, mu, Sigma, actions, n_EM, starttype,U)
% Function for seperate HMM's with untied states:

% 'U' gives the smoothing parameter in the MMI criterion: it is used as a
% multiplication by (U/T_r) of the logposterior for video r. 

% Retrieve some standard parameters from input: 
K = size(A,1);
P = size(mu,1);
Ntrain = size(train,1);
nact = size(actions,1);
video_index = cell2mat(train(:,1));

% Storage
mu_store = zeros(P,K,n_EM);
A_store = zeros(K,K,nact,n_EM);
Sigma_store = zeros(P,P,K,n_EM);
loglik_store = zeros(n_EM,1);

% Weights = number of frames available for each class
w = zeros(nact,1);
for i=1:nact
    w(i) = sum(cell2mat(train(cell2mat(train(:,1))==actions(i),4)));
end

% Run EBW:
for l=1:n_EM
    fprintf('EBW iteration %d of %d\n',l,n_EM);
    
    % Set up storage
    exp_num_trans = zeros(K,K,nact);
    postmix = zeros(K,1);
    m = zeros(P,K);
    op = zeros(P,P,K);
    denom_gamma = zeros(K,1);
    gamma_true = zeros(K,1);
    loglik = 0;

    % E-step: %%%
    for i=1:nact
        % Sufficient statistics
        [loglik_2, exp_num_trans_2, ~, postmix_2, m_2, ~, op_2,denom_gamma_2,gamma_true_2] = ...
            ess_mhmm_EBW(repmat(1,K,1)/K, A, repmat(1,K,1), mu, Sigma, train(logical(video_index==actions(i)),6),i,U);
        
        % Accumulate
        loglik = loglik + loglik_2;
        exp_num_trans(:,:,i) = exp_num_trans_2 * (1000/w(i));
        postmix = postmix + postmix_2 * (1000/w(i));
        m = m + m_2 * (1000/w(i));
        op = op + op_2 * (1000/w(i));
        denom_gamma = denom_gamma + denom_gamma_2 * (1000/w(i));
        gamma_true = gamma_true + gamma_true_2 * (1000/w(i));
    end
 
    % Determine D(i) values %%%
    pos_var = zeros(K,1);
    for j=1:K % Solve ABC-formula for quadratic equation in D(i)
        a1 = diag(Sigma(:,:,j));
        b1 = diag(op(:,:,j)) + diag(postmix(j)*(Sigma(:,:,j) + mu(:,j)*mu(:,j)')) - diag(2*mu(:,j)*m(:,j)');
        c1 =  diag(postmix(j)*op(:,:,j)) - diag(m(:,j)*m(:,j)');
        d1 = sqrt(b1.^2 - 4.*a1.*c1);
        x=zeros(P,2);
        x(:,1) = ( -b1 + d1 ) ./ (2*a1);
        x(:,2) = ( -b1 - d1 ) ./ (2*a1);
        x(imag(x)~=0) = 0; % remove imaginary roots, they don't matter
        pos_var(j) = max(x(:));
    end
    D = max([2*denom_gamma 2*pos_var]')+1;
    
    % Update mu & Sigma
    mu_new = zeros(P,K);
    for i=1:K
        mu_new(:,i) = (m(:,i) + D(i)*mu(:,i)) / (postmix(i) + D(i));
    end
    Sigma_new = zeros(P,P,K);
    for i=1:K
        SS = ((op(:,:,i) + D(i)*(mu(:,i)*mu(:,i)' + Sigma(:,:,i)))/ (postmix(i) + D(i))) - (mu_new(:,i)*mu_new(:,i)');
        SS = diag(SS);
        SS(SS<=0) = 0.0001;
        Sigma_new(:,:,i) = diag(SS);
    end
    
    % Store and prepare for new iteration
    mu_store(:,:,l) = mu_new;
    Sigma_store(:,:,:,l) = Sigma_new;
    loglik_store(l) = loglik;
    mu=mu_new;
    Sigma=Sigma_new;
    
    %  Throw away unoccupied states:
    %if any(postmix == 0)
    %    mu(:,(postmix == 0)) = [];
    %    Sigma(:,:,postmix==0) = [];
    %    A(postmix==0,:,:) = [];
    %    A(:,postmix==0,:) = [];
    %    K = size(A,1);
    %end 
    
    % Update transition matrix through ordinary ML:
    A_new = zeros(K,K,nact);
    for i=1:nact
        [~, ~, A_new(:,:,i), ~, ~, ~] = mhmm_em(train(logical(video_index==actions(i)),6), repmat(1,K,1)/K, A(:,:,i), mu, Sigma, repmat(1,K,1), 'max_iter', 1,'cov_type','diag','adj_mu',0,'adj_Sigma',0,'verbose',0);    
    end
    A_store(:,:,:,l) = A_new;
    A = A_new;
    
    fprintf('EBW iteration %d of %d: Loglik = %d \n ',l,n_EM,loglik);
    
end
 
fprintf('Finished EBW \n');
end
  

%%% 

function [loglik, exp_num_trans, exp_num_visits1, postmix, m, ip, op,denom_gamma,gamma_true] = ...
    ess_mhmm_EBW(prior, A, mixmat, mu, Sigma, data, true_action,U)
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
P = size(data{1},1);
K = length(prior);
M = size(mixmat,2);
exp_num_trans = zeros(K,K);
exp_num_visits1 = zeros(K,1);
postmix = zeros(K,M);
m = zeros(P,K,M);
op = zeros(P,P,K,M);
ip = zeros(K,M);
mix = (M>1);
denom_gamma = zeros(K,1);
gamma_true = zeros(K,1);
loglik = 0;

nact = size(A,3);

for ex=1:numex
  %obs = data(:,:,ex);
  obs = data{ex};
  T = size(obs,2);
  B = mixgauss_prob(obs, mu, Sigma);
  gamma_store = zeros(K,T,nact);
  xi_summed_store = zeros(K,K,nact);
  LL = zeros(nact,1);
  current_loglik = zeros(nact,1);
  
  % Calculate gamma, xi-summed and postloglik under each model:
  for i=1:nact
  [alpha, beta, gamma_store(:,:,i), current_loglik(i), xi_summed_store(:,:,i)] = fwdback(prior, A(:,:,i), B);   
  LL(i,1) = mhmm_logprob(obs, ones(K,1)/K, A(:,:,i), mu, Sigma, repmat(1,K,1));
  end
  
  % Normalize post prob:
  LL = LL*(U/T);
  
  if all(isinf(LL))
      LL2 = ones(numel(LL),1);
  else
      LL2 = exp(LL - repmat(max(LL),[nact 1]));
  end
  
  LL2 = LL2/sum(LL2); % normalized probability
  postprob = repmat(reshape(LL2,[1,1,nact]),[K,T,1]); % spread out posterior prob
  
  % Add wrong classes:
  gamma3 = gamma_store .* postprob;
  tosum = ~ismember(1:nact,true_action);
  delta_gamma = gamma_store(:,:,true_action) - sum(gamma3,3);
  
  postprob = repmat(reshape(LL2,[1,1,nact]),[K,K,1]); % spread out posterior prob
  xi_summed2 = xi_summed_store .* postprob;
  delta_xi_summed = xi_summed_store(:,:,true_action) - sum(xi_summed2,3);

  % Put back in original variable, and proceed as usual:
  gamma = delta_gamma;
  xi_summed = delta_xi_summed;
  exp_num_trans = exp_num_trans + xi_summed; % sum(xi,3);
  exp_num_visits1 = exp_num_visits1 + gamma(:,1);
  postmix = postmix + sum(gamma,2); 
  denom_gamma = denom_gamma + sum(sum(gamma3(:,:,tosum),3),2);
  gamma_true = gamma_true + sum(gamma3(:,:,true_action),2);
  gamma2 = reshape(gamma, [K 1 T]); % gamma2(i,m,t) = gamma(i,t)
  
  % Sufficient statistics
  for i=1:K
    for k=1:M
      w = reshape(gamma2(i,k,:), [1 T]); % w(t) = w(i,k,t,l)
      wobs = obs .* repmat(w, [P 1]); % wobs(:,t) = w(t) * obs(:,t)
      m(:,i,k) = m(:,i,k) + sum(wobs, 2); % m(:) = sum_t w(t) obs(:,t)
      op(:,:,i,k) = op(:,:,i,k) + wobs * obs'; % op(:,:) = sum_t w(t) * obs(:,t) * obs(:,t)'
      ip(i,k) = ip(i,k) + sum(sum(wobs .* obs, 2)); % ip = sum_t w(t) * obs(:,t)' * obs(:,t)
    end
  end
  
  % Update loglik:
  current_loglik = current_loglik*(U/T);
  if all(isinf(current_loglik))
      curlog_2 = -Inf;
      loglik = loglik +  (-50);
  else
      curlog_2 = exp(current_loglik - repmat(max(current_loglik),[nact 1]));
      curlog_2 = curlog_2/sum(curlog_2); % normalized probability
      loglik = loglik +  log(curlog_2(true_action));
  end
   
end

end





  
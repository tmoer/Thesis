function [ LL_frame ] = frame_lik(test, A_store, mu_store, Sigma_store)
% Find frame-wise posterior likelihood under trained models

K = size(A_store,1);
nact = size(A_store,3);
Ntest = size(test,1);
n_iter = size(A_store,4);
nframe = cell2mat(test(:,4));
LL_frame = [];

for l=1:n_iter
    % Calculate likelihoods for all videos under all models:
    LL_raw = zeros(nact,Ntest);
    for j=1:Ntest
        for i=1:nact    
            LL_raw(i,j) = mhmm_logprob(test{j,6}, ones(K,1)/K, A_store(:,:,i,l), mu_store(:,:,l), Sigma_store(:,:,:,l), repmat(1,K,1));
        end
    end

    LL_raw = LL_raw./repmat(nframe,1,nact)'; % Likelihood per frame
    LL_frame = cat(3,LL_frame,LL_raw);
end

end


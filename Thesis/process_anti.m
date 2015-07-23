function [ LL_back ] = process_anti( test, LL_frame, antitype, A_anti, mu, Sigma, K )
% Function to compute the background corrected likelihoods for LL_frame,
% under each of the antimodels in A_anti

% Initialize parameters:
nact = size(LL_frame,1);
N_anti = numel(antitype);
Ntest = size(test,1);
nframe = cell2mat(test(:,4));

LL_back = [];

for i=1:N_anti
    model = antitype{i};
    antimodel = A_anti{i};
    
    switch model
        case {'none' 'sum' 'entropy'}
            LL_back = cat(3,LL_back,LL_frame);
            
        case 'filler'
            LL_filler = zeros(1,Ntest);
            for j=1:Ntest
                LL_filler(1,j) = mhmm_logprob(test{j,6}, ones(K,1)/K, antimodel, mu, Sigma, repmat(1,K,1));
            end
            LL_filler = LL_filler./(nframe');
            LL_back = cat(3, LL_back, LL_frame - repmat(LL_filler,nact,1));
            
        case 'flat'
            LL_flat = zeros(1,Ntest);
            for j=1:Ntest
                LL_flat(1,j) = mhmm_logprob(test{j,6}, ones(K,1)/K, antimodel, mu, Sigma, repmat(1,K,1));
            end
            LL_flat = LL_flat./(nframe');
            LL_back = cat(3, LL_back, LL_frame - repmat(LL_flat,nact,1));
            
        case 'anti_full'
            LL_full = zeros(nact, Ntest);
            for j=1:Ntest
                for l=1:nact 
                    LL_full(l,j) = mhmm_logprob(test{j,6}, ones(K,1)/K, antimodel(:,:,l), mu, Sigma, repmat(1,K,1));
                end
            end
            LL_full = LL_full./repmat(nframe,1,nact)';
            LL_back = cat(3,LL_back, LL_frame - LL_full);
            
        case 'anti_matej'
             LL_matej = zeros(nact, Ntest);
            for j=1:Ntest
                for l=1:nact 
                    LL_matej(l,j) = mhmm_logprob(test{j,6}, ones(K,1)/K, antimodel(:,:,l), mu, Sigma, repmat(1,K,1));
                end
            end
            LL_matej = LL_matej./repmat(nframe,1,nact)';
            LL_back = cat(3,LL_back, LL_frame - LL_matej);
            
        case 'combination'
            if ~exist('LL_filler','var')
                LL_filler = zeros(1,Ntest);
                for j=1:Ntest
                    LL_filler(1,j) = mhmm_logprob(test{j,6}, ones(K,1)/K, antimodel(:,:,1), mu, Sigma, repmat(1,K,1));
                end
            end
            LL_filler = repmat(LL_filler,nact,1);
            
            if ~exist('LL_full','var')
                LL_full = zeros(nact, Ntest);
                for j=1:Ntest
                    for l=1:nact 
                        LL_full(l,j) = mhmm_logprob(test{j,6}, ones(K,1)/K, antimodel(:,:,(l+1)), mu, Sigma, repmat(1,K,1));
                    end
                end
                LL_full = LL_full./repmat(nframe,1,nact)';
            end
            
            % Make reweighted combination: %% RECHECK THIS PART LATER
            LL_comb = cat(3, LL_full, LL_filler);
            maxLL = max(LL_comb,[],3);
            LL_comb = LL_comb - repmat(maxLL,[1 1 2]);
            LL_back = cat(3, LL_back, LL_frame - (maxLL + log(sum(0.5*exp(LL_comb),3))) );
            
        case 'combination2'
            if ~exist('LL_flat','var')
                 LL_flat = zeros(1,Ntest);
                for j=1:Ntest
                    LL_flat(1,j) = mhmm_logprob(test{j,6}, ones(K,1)/K, antimodel(:,:,1), mu, Sigma, repmat(1,K,1));
                end
            end
            LL_flat = repmat(LL_flat,nact,1);
            
            if ~exist('LL_full','var')
                LL_full = zeros(nact, Ntest);
                for j=1:Ntest
                    for l=1:nact 
                        LL_full(l,j) = mhmm_logprob(test{j,6}, ones(K,1)/K, antimodel(:,:,(l+1)), mu, Sigma, repmat(1,K,1));
                    end
                end
                LL_full = LL_full./repmat(nframe,1,nact)';
            end
            
            % Make reweighted combination: %% RECHECK THIS PART LATER
            LL_comb = cat(3, LL_full, LL_flat);
            maxLL = max(LL_comb,[],3);
            LL_comb = LL_comb - repmat(maxLL,[1 1 2]);
            LL_back = cat(3, LL_back, LL_frame - (maxLL + log(sum(0.5*exp(LL_comb),3))) );
            
    end
    
    
    
end
        
            
            
            
            
   


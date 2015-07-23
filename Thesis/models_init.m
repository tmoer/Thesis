function [A, mu, Sigma, actions] = models_init(train, K, starttype)
% Generic function to initialize HMM's
% K is number of key postures, needs to be specified
% Two starttypes supported: 'flat' and 'clustered'

% Retrieve some data set characteristics:
featuremat = cell2mat(train(:,6)')';
P = size(train{1,6},1);
Ntrain = size(train,1);
actions = unique(cell2mat(train(:,1)));
nact = size(actions,1);
frame_index=[];
for i=1:Ntrain
    frame_index=[frame_index; repmat(train{i,1},size(train{i,6},2),1)];
end

% Initialize output:
A = zeros(K,K,nact);
mu = zeros(P,K);
Sigma = zeros(P,P,K);

% Create start:
switch starttype
    case 'flat'
        mu = repmat(mean(featuremat)',1,K);
        Sigma = repmat(diag(diag(cov(featuremat))),1,1,K);
        A = normalize(rand(K,K,nact),2);
     
    otherwise % clustered
    prevent = true;
    while prevent
        [cluster,C]=kmeans(featuremat,K,'MaxIter',1000);
        QQQ = tabulate(cluster);
        prevent = any(QQQ(:,2)==1);
    end
  
    % mu
    mu = C';
    
    % Sigma
    for i=1:K
        subs = featuremat((cluster==i),:);
        Sigma(:,:,i) = diag(diag(cov(subs)));
    end
    
    % A
    for i=1:nact
        subset = cluster(frame_index==actions(i)); % select transitions in action class i
        counts = full(sparse(subset(1:end-1),subset(2:end),1)); % reconstruct full transition matrix  
        % check whether counts is size K*K
        if  size(counts,1) ~= K
            counts((size(counts,1)+1):K,:)=0;
        end
        if  size(counts,2) ~= K
            counts(:,(size(counts,2)+1):K)=0;
        end
        counts = counts./repmat(sum(counts,2),1,K); % rownormalize
        counts(isnan(counts))=0; % remove NaN's if rowsum was zero
        A(:,:,i) = counts;
    end
    
end

end


function [pred_table precision confusion_novel] = classify(test, actions, LL_frame, antitype,tau)
% Generic function to classify
% A_store(:,:,i,j) holds the model for actions(i) for iteration j
% Output is calculated for each iteration

    % Initialize parameters
    nact = size(actions,1);
    n_iter = size(LL_frame,3);

    % Create storage
    precision = [];
    pred_table = [];

    for l=1:n_iter
        LL = LL_frame(:,:,l);
    
        % P(X|S)
        [value index] = max(LL); 
    
        % P(S|X)
        expsum = exp(LL'- repmat(max(LL)',1,nact))';
        LL2 = expsum./repmat(sum(expsum),nact,1);
        [value2 ~ ] = max(LL2);
        %[value3 ~ ] = max(log(LL2));
    
        % entropy
        value4 = -sum(LL2.*log2(LL2));
    
        % Generate output:
        class = actions(index);
        truth = cell2mat(test(:,1));
    
        % Store:
        precision = cat(1,precision, mean(class == truth));
        pred_table = cat(3,pred_table, [truth class value' value2' value4']);
    end
    
if nargin>3
    N_anti = size(LL_frame,3);
    cat(2,pred_table,zeros(size(pred_table,1),1,size(pred_table,3)));
    
    % Create storage
    precision = [];
    pred_table_novel = [];
    confusion_novel = [];
        
    for i=1:N_anti
        antimodel = antitype{i};
        tau1 = tau(i);
        
        truth = pred_table(:,1,i); % truth with novel
        class = pred_table(:,2,i);
        value = pred_table(:,3,i);
         if strcmp(antimodel,'sum')
            value = pred_table(:,4);
        else if strcmp(antimodel,'entropy')
            value = pred_table(:,5);
            end
         end
        
        class2 = class;
        class(value < tau1) = 100;
        pred_table(:,2,i) = class; % assignment with novel
        pred_table(:,6,i) = class2; % assignment full
        pred_table(:,7,i) = cell2mat(test(:,8)); % truth full
        
    % Novelty confusion table
    confusion = confusionmat(class == 100,truth ==100);
    precision1 = sum(diag(confusionmat(truth(truth~=100),class(truth~=100))));
    confusion = [precision1, confusion(1,2); (confusion(1,1)-precision1) 0; confusion(2,:)];
    confusion_novel1 = mk_stochastic(confusion')';
    
    precision = cat(1,precision, mean([confusion_novel1(1,1),confusion_novel1(3,2)]));
    confusion_novel = cat(3,confusion_novel,confusion_novel1);
    end
end

end






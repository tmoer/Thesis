function [pred_table precision confusion_novel] = classify_novelty(test, actions, LL_frame, antitype,tau)
% novelty classifier

   N_anti = size(LL_frame,3);
    
    % Create storage
    precision = [];
    pred_table_novel = [];
    confusion_novel = [];
        
    for i=1:N_anti
        antimodel = antitype{i};
        tau1 = tau(i);
        
        truth = pred_table(:,1,i);
        class = pred_table(:,2,i);
        value = pred_table(:,3,i);
         if strcmp(antimodel,'sum')
            value = pred_table(:,4);
        else if strcmp(antimodel,'entropy')
            value = pred_table(:,5);
            end
         end
        
        class(value < tau1) = 100;
        pred_table(:,2,i) = class;
        
    % Novelty confusion table
    confusion = confusionmat(class == 100,truth ==100);
    precision1 = sum(diag(confusionmat(truth(truth~=100),class(truth~=100))));
    confusion = [precision1, confusion(1,2); (confusion(1,1)-precision1) 0; confusion(2,:)];
    confusion_novel1 = mk_stochastic(confusion')';
    
    precision = cat(1,precision, mean([confusion_novel1(1,1),confusion_novel1(3,2)]));
    confusion_novel = cat(3,confusion_novel,confusion_novel1);
    end


end


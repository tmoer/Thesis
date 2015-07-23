function [ train test train_new ] = split_set(mydata, type, i, known, new)
% Function to built train and test set

switch type
    case 'normal'
        test = mydata(known(i,:)',:);
        train = mydata(known(1:3~=i,:)',:);
        
    case 'novelty'
        mydata(:,8) = mydata(:,1);
        train = mydata(known(1:3~=i,:)',:);
        train_new = mydata(new(1:3~=i,:)',:);
        train_new(:,1)=num2cell(100); % Novelty indicator
        
        test1 = mydata(known(i,:)',:);
        test2 = mydata(new(i,:)',:);
        test2(:,1)=num2cell(100); % Novelty indicator
        test = [test1 ; test2];

end


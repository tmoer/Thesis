function [known new] = split_index(N,type,mydata,leaveout)
% Function to make a 3-fold split
% type can be 'normal': returns known
% or can be 'novelty': returns known and new

switch type
    case 'normal'
        N = N/3;
        [dummy idx] = sort(rand(3,N));
        known = 3*repmat(0:(N-1),3,1)+idx;
        
    case 'novelty'
            known = find(~ismember(cell2mat(mydata(:,1)),leaveout));
            new = find(ismember(cell2mat(mydata(:,1)),leaveout));
            N1 = numel(known)/3;
            N2 = numel(new)/3;
            known = reshape(known,[3,N1]);
            new = reshape(new,[3,N2]);
            
            % Random permutation
            [dummy idx1] = sort(rand(3,N1));
            [dummy idx2] = sort(rand(3,N2));
            known = known(3*repmat(0:(N1-1),3,1)+idx1);
            new = new(3*repmat(0:(N2-1),3,1)+idx2);
end

end


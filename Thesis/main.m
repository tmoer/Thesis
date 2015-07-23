% Thesis code
ones(10)*ones(10); 
addpath('/home/thomas/Matlab/Action3D');
addpath('/home/thomas/Matlab');
addpath('/home/thomas/Matlab/HMMall/HMM');
addpath('/home/thomas/Matlab/HMMall/KPMtools','-end');
addpath('/home/thomas/Matlab/HMMall/KPMstats');

% Read data & make feature vectors
mydata = read_mydata('/home/thomas/Matlab/Action3D');
center = true; rotated = false; rescale = false; head = true; angles = true; legs = true;
mydata = position_features(mydata,center,rotated,rescale,head,angles,legs);

% Store some global values
N=size(mydata,1)
actions_all = unique(cell2mat(mydata(:,1)));
nact = numel(actions_all)

% Plotting (optional)
% raw=false;
% plot_skeleton(mydata,13,50,raw)




%%%%%%%%%%%%%%%%%%%%%%%%%%% Chapter 4 
% PCA plot on full featurespace
[~] = pca_adjust(mydata, mydata, 2, true);

%%%% CLUSTERED start
% Optimal performance
[precision pred_table opt_iter A mu Sigma ] = hmm_optimal('clustered', 2, mydata, 50, 10);
mean(precision) %% 0.939
err = confusionmatrix(pred_table(:,1),pred_table(:,2))

% Effect of K and P
K_range = 10:10:100;
P_range = [3,5,10,15,30];
[out_store] = hmm_KP(mydata, 5, K_range, P_range, false);

%%%%% EM MODEL 
[train_precision1 test_precision1 pred_table1  train_loglik1] = hmm_sequence('EM', 3, mydata, 50, 10, 'flat', 40, true);

[precision3  pred_table3 opt_iter3 A3 mu3 Sigma3 ] = hmm_optimal('EM', 2, mydata, 50, 10, 'flat',30);
mean(precision3) %% 0.963
err = confusionmatrix(pred_table3(:,1),pred_table3(:,2))

[precision4  pred_table4 opt_iter4 A4 mu4 Sigma4 ] = hmm_optimal('EM', 3, mydata, 50, 10,'clustered',20);
mean(precision4) %% 0.940




%%%%%%%%%%%%%%%%%%%%%%%%%%%% Chapter 5: EBW model
% Sequence from flat
[train_precision5 test_precision5 pred_table5  train_loglik5] = hmm_sequence('EBW',...
    3, mydata, 50, 10, 'flat', 40, true, U);

% Optimal from flat 
U=1.5
[precision7  pred_table7 opt_iter7 A7 mu7 Sigma7 ] = hmm_optimal('EBW',...
    2, mydata, 50, 10, 'flat', 40, U); % 3,5 uur
mean(precision7) % 0.945
err = confusionmatrix(pred_table7(:,1),pred_table7(:,2))

% Effect of U
U_seq = [0.1 1.5 5 20];
[store] = hmm_U(U_seq, 1, mydata, 50, 10, 'flat', 30, true)

%%%%% Effect of training set size
models = {'clustered' 'EM'  'EBW'}
[outout] = hmm_size(models, 6, 1, mydata, 50, 10, 'flat', 40, 1.5)





%%%%%%%%%%%%%%%%%%%%% Chapter 6 - Novelty detection

[pred_table_novel1,tau1,LL_distance1] = hmm_novelty(mydata, 2, 'clustered', 50, 10, ...
    {'none' 'sum' 'filler' 'flat' 'anti_full' 'anti_matej' 'combination' 'combination2'}, 'clustered', 4);
[tau11, confusion_novel11, error11] = plot_novel(pred_table_novel1,...
    {'none' 'sum'  'filler' 'flat' 'anti_full' 'anti_matej' 'combination' 'combination2'}, true)

%[pred_table_novel2,tau2,LL_distance2] = hmm_novelty(mydata, 2, 'clustered', 50, 30,{'none' 'sum' 'filler' 'flat' 'anti_full' 'anti_matej' 'combination' 'combination2'}, 'clustered', 4);
%[tau21, confusion_novel21, error21] = plot_novel(pred_table_novel2, {'none' 'sum'  'filler' 'flat' 'anti_full' 'anti_matej' 'combination' 'combination2'}, true)


[pred_table_novel3,tau3,LL_distance3] = hmm_novelty(mydata, 2, 'EM', 50, 10,...
    {'none' 'sum' 'filler' 'flat' 'anti_full' 'anti_matej' 'combination' 'combination2'}, 'flat', 20);
[tau31, confusion_novel31, error31] = plot_novel(pred_table_novel3,...
    {'none' 'sum' 'filler' 'flat' 'anti_full' 'anti_matej' 'combination' 'combination2'}, true)

%[pred_table_novel4,tau4,LL_distance4] = hmm_novelty(mydata, 2, 'EM', 50, 30, {'none' 'sum' 'filler' 'flat' 'anti_full' 'anti_matej' 'combination' 'combination2'}, 'flat', 20);
%[tau41, confusion_novel41, error41] = plot_novel(pred_table_novel4, {'none' 'sum' 'filler' 'flat' 'anti_full' 'anti_matej' 'combination' 'combination2'}, true)

[pred_table_novel5,tau5,LL_distance5] = hmm_novelty(mydata, 1, 'EBW', 50, 10, ...
    {'none' 'sum' 'filler' 'flat' 'anti_full' 'anti_matej' 'combination' 'combination2'}, 'flat', 35, 5);
[pred_table_novel6,tau6,LL_distance6] = hmm_novelty(mydata, 1, 'EBW', 50, 30, ...
    {'none' 'sum' 'filler' 'flat' 'anti_full' 'anti_matej' 'combination' 'combination2'}, 'flat', 35, 5);

%[tau51, confusion_novel51, error51] = plot_novel(pred_table_novel5, {'none' 'sum' 'filler' 'flat' 'anti_full' 'anti_matej' 'combination' 'combination2'}, true)
%[tau61, confusion_novel61, error61] = plot_novel(pred_table_novel6, {'none' 'sum' 'filler' 'flat' 'anti_full' 'anti_matej' 'combination' 'combination2'}, true)

%%%%%% Multidim scaling
[~] = my_MDS(LL_distance1, pred_table_novel1);

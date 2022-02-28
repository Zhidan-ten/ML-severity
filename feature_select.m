close all;
clear;

addpath('.\FSLib_v7.0.1_2020_2\lib'); % dependencies
addpath('.\FSLib_v7.0.1_2020_2\methods'); % FS methods
addpath(genpath('.\FSLib_v7.0.1_2020_2\lib\drtoolbox'));
addpath('.\FSLib_v7.0.1_2020_2\eval_metrics');



% Select a feature selection method from the list,有监督(ILFS、mrmr、relieff、fsv、infFs、ECFS);无监督()
% listFS = {'ILFS','InfFS','ECFS','mrmr','relieff','mutinffs','fsv','laplacian','mcfs','rfe','L0','fisher','UDFS','llcfs','cfs','fsasl','dgufs','ufsol','lasso'};
% listFS = {'ILFS','mrmr','relieff','mutinffs','fsv','laplacian','mcfs','rfe','fisher','UDFS','llcfs','cfs','fsasl','dgufs','lasso'};
listFS = {'mrmr'};
all_dsc = [];
for r = 1:1
    [ methodID ] = r;
    % [ methodID ] = readInput( listFS );%2和3是二分类
    selection_method = listFS{methodID}; % Selected
    
    X_train = importdata('nwtraindata1.mat');
    X_test = importdata('nwtestdata1.mat');
    Y_train = importdata('nwtrainlabel1.mat');
    Y_test = importdata('nwtestlabel1.mat');
    numF = size(X_train,2);%特征总数
    
    % feature Selection on training data
    switch lower(selection_method)
        case 'inffs'
            % Infinite Feature Selection 2015 updated 2016
            alpha = 0.5;    % default, it should be cross-validated.
            sup = 1;        % Supervised or Not
            [ranking, w] = infFS( X_train , Y_train, alpha , sup , 0 );
            
        case 'ilfs'
            % Infinite Latent Feature Selection - ICCV 2017
            [ranking, weights] = ILFS(X_train, Y_train , 6, 0 );
            
        case 'fsasl'
            options.lambda1 = 1;
            options.LassoType = 'SLEP';
            options.SLEPrFlag = 1;
            options.SLEPreg = 0.01;
            options.LARSk = 5;
            options.LARSratio = 2;
            nClass=4;
            [W, S, A, objHistory] = FSASL(X_train', nClass, options);
            [v,ranking]=sort(abs(W(:,1))+abs(W(:,2)),'descend');
        case 'lasso'
            lambda = 24;
            [B,stats] = lasso(X_train,Y_train);
            %lambda = stats.IndexMinMSE;  % 最小MSE对应lambda
            [v,ranking]=sort(abs(B(:,lambda)),'descend');
            
        case 'ufsol'
            para.p0 = 'sample';
            para.p1 = 1e6;
            para.p2 = 1e2;
            nClass = 4;
            [~,~,ranking,~] = UFSwithOL(X_train',nClass,para) ;
            
        case 'dgufs'
            
            S = dist(X_train');
            S = -S./max(max(S)); % it's a similarity
            nClass = 4;
            alpha = 0.5;
            beta = 0.9;
            nSel = 5206;
            [Y,L,V,Label] = DGUFS(X_train',nClass,S,alpha,beta,nSel);
            [v,ranking]=sort(Y(:,1)+Y(:,2),'descend');
            
            
        case 'mrmr'
            [ranking, w] = mRMR(X_train, Y_train, numF);
            
        case 'relieff'
            [ranking, w] = reliefF( X_train, Y_train, 20);
            
        case 'mutinffs'
            [ ranking , w] = mutInfFS( X_train, Y_train, numF );
            
        case 'fsv'
            [ ranking , w] = fsvFS( X_train, Y_train, numF );
            
        case 'laplacian'
            W = dist(X_train');
            W = -W./max(max(W)); % it's a similarity
            [lscores] = LaplacianScore(X_train, W);
            [junk, ranking] = sort(-lscores);
            
        case 'mcfs'
            % MCFS: Unsupervised Feature Selection for Multi-Cluster Data
            options = [];
            options.k = 5; %For unsupervised feature selection, you should tune
            %this parameter k, the default k is 5.
            options.nUseEigenfunction = 4;  %You should tune this parameter.
            [FeaIndex,~] = MCFS_p(X_train,numF,options);
            ranking = FeaIndex{1};
            
        case 'rfe'
            ranking = spider_wrapper(X_train,Y_train,numF,lower(selection_method));
            
        case 'l0'
            ranking = spider_wrapper(X_train,Y_train,numF,lower(selection_method));
            
        case 'fisher'
            ranking = spider_wrapper(X_train,Y_train,numF,lower(selection_method));
            
            
        case 'ecfs'
            % Features Selection via Eigenvector Centrality 2016
            alpha = 0.5; % default, it should be cross-validated.
            ranking = ECFS( X_train, Y_train, alpha )  ;
            
        case 'udfs'
            % Regularized Discriminative Feature Selection for Unsupervised Learning
            nClass = 4;
            ranking = UDFS(X_train , nClass );
            
        case 'cfs'
            % BASELINE - Sort features according to pairwise correlations
            ranking = cfs(X_train);
            
        case 'llcfs'
            % Feature Selection and Kernel Learning for Local Learning-Based Clustering
            ranking = llcfs( X_train );
            
        otherwise
            disp('Unknown method.')
    end
    
end   

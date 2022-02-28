close all;
clear;
addpath('.\libsvm-3.20-t\matlab\');


traindata = importdata('nwtraindata1.mat');
testdata = importdata('nwtestdata1.mat');
trainlabel = importdata('nwtrainlabel1.mat');
testlabel = importdata('nwtestlabel1.mat');
feats_importance = importdata('feat_import.mat');
[sort_feats,index] = sort(feats_importance,'descend');

q = 1;
for p = 1:q
    m = 10553;
    all_acc =[];        
    for j=1:m
        new_traindata = [];
        new_testdata =[];
        for i = 1:j
            one_train_feats = traindata(:,index(1,10553-i));
            one_test_feats = testdata(:,index(1,10553-i));
            new_traindata = [new_traindata,one_train_feats];
            new_testdata = [new_testdata,one_test_feats];
        end
       
        %classification
        %Random Forest
        B = TreeBagger(222,new_traindata,trainlabel,'OOBPrediction','on','NumPredictorsToSample',2,'MinLeafSize',1,'Method','classification');

%       B  = importdata('RF_RF_model.mat');
        [predict_label,pro,~] = predict(B,new_testdata);
        for k = 1:length(testlabel)
            pre(k,1) = str2num(predict_label{k,1});
        end
%         
        %KNN
%         mdl = ClassificationKNN.fit(new_traindata,trainlabel,'NumNeighbors',1);
%         [pre,pro,~] = predict(mdl, new_testdata);
        
        %Naive Bayes(朴素贝叶斯)
%                 nb = NaiveBayes.fit(new_traindata, trainlabel);
%                 [pre] = predict(nb, new_testdata);
        
        %Ensembels(集成学习)
%                 ens = fitensemble(new_traindata,trainlabel,'AdaBoostM2',100,'tree','type','classification');
%                 [pre,pro] = predict(ens, new_testdata);
%                 score(j) = pro(1);

       %discriminant analysis classifier
%                Factor = ClassificationDiscriminant.fit(new_traindata, trainlabel, 'discrimType', 'pseudoLinear');
%                [pre, Scores] = predict(Factor, new_testdata);
        
%         SVM
%         SVMStruct = svmtrain(trainlabel,new_traindata,'-s 0 -t 1 -c 1.0 -g 1.0 -b 1 ');
% %         SVMStruct  = importdata('SVM_bestmodel.mat');
%         [pre, accuracy, prob_estimates] = svmpredict(testlabel,new_testdata,SVMStruct,'-b 1');
% %         [pre, accuracy, prob_estimates] = svmpredict(trainlabel,new_traindata,SVMStruct,'-b 1');
%          [label,rOrder] = sort(SVMStruct.Label);
%          prob_estimatesR = prob_estimates(:,rOrder);
        
                %LDA和QDA分类器
% %              [pre, err]=classify(new_testdata, new_traindata, trainlabel,'diagLinear');
% %              [pre, err]=classify(new_testdata, new_traindata, trainlabel,'diagQuadratic');
        
        acc = sum((pre-testlabel) == 0)/length(testlabel);
%         acc = sum((pre-trainlabel) == 0)/length(trainlabel);
        all_acc = [all_acc;acc];
        accuracy1 = [];
        sen1 = [];
        spe1 = [];
        AA1 = [];
        PLR1 = [];
        NPR1 = [];
        AUC1 = [];
        AUC_CI1 = [];

            disp(['Accuracy is: ',num2str(acc)]);
            disp(j);
            for n = 1:max(testlabel)
                testlabeln = testlabel; pren = pre;
                testlabeln(testlabeln ~= n) = 0;
                testlabeln(testlabeln == n) = 1;
                pren(pren ~= n) = 0;
                pren(pren == n) = 1;
                [accuracy,sen,spe,ppv,npv,AA,PLR,NPR,f1_score,ci_all] = cul_acc(pren,testlabeln);
                [auc,auc_ci] = Delong_CI(pro(:,n)',testlabeln');
                accuracy1 = [accuracy1;accuracy];
                sen1 = [sen1;sen];
                spe1 = [spe1;spe];
                AA1 = [AA1;AA];
                PLR1 = [PLR1;PLR];
                NPR1 = [NPR1;NPR];
                AUC1 = [AUC1;auc];
                AUC_CI1 = [AUC_CI1;auc_ci];
                disp(['第',num2str(n),'类准敏特为: ',num2str(accuracy),' ',num2str(sen),' ',num2str(spe)]);
                disp(ci_all);
                disp(auc_ci);
            end
    end
    all_all_acc(:,p) = all_acc;
end
all_all_acc1 = sum(all_all_acc,2) / q;

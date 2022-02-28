function feats = Wfeats2D(image)%二维小波变换后特征(146)
% clear
addpath('F:\Program Files\MATLAB\workspace\GLRL\');
warning('off','all');
% image = importdata('27.png');
roi = image;



wt = dwt2(roi,'haar');%二维小波变换，只单层
component = wt;

feats = [];
D = component;
Vals = D(:);
    
% First-order statistical properties
energy = Vals'*Vals;
range = max(Vals)-min(Vals);
mad = mean(abs(Vals-mean(Vals)));
rms = (energy/length(Vals))^0.5;
[h,~] = hist(Vals);
uniformity = h*h';

    
% 1~14
feats = [feats;energy;entropy(Vals);kurtosis(Vals);max(Vals);mean(Vals);...
        mad;median(Vals);min(Vals);range;rms;skewness(Vals);...
        std(Vals);uniformity;var(Vals)];
    

    
n = length(D);
glcms = zeros(5,5,4); glrlms = zeros(5,n,4);%四个方向的矩阵
offset = [0 1;-1 1;-1 0;-1 -1];
    for i = 1:4
        a1 = squeeze(D(:,:));
        glcms(:,:,i) = squeeze(glcms(:,:,i)) + graycomatrix(a1,'offset',offset(i,:),'Numlevels',5);
        [glrlm,~] = grayrlmatrix(a1,'offset',i,'NumLevels',5);
        glrlms(:,:,i) = squeeze(glrlms(:,:,i))+glrlm{1,1};
    end
    for s = 1:4
        glcm = squeeze(glcms(:,:,s));
        glcm = glcm/sum(glcm(:));
        px = sum(glcm); py = sum(glcm');
        mux = mean(glcm); muy = mean(glcm');
        sigx = std(glcm); sigy = std(glcm');
        hx = entropy(px); hy = entropy(py); hxy = entropy(glcm(:));
        hxpy = zeros(1,10); hxsy = zeros(1,5);
        au = 0; cp = 0; cs = 0; ct = 0; di = 0; cor = 0; co = 0;
        hxy1 = 0; hxy2 = 0; hm1 = 0; hm2 = 0;
        idmn = 0; idn = 0; iv = 0; smooth = 0.001;
        for i = 1:5
            for j = 1:5
                hxpy(i+j) = hxpy(i+j)+glcm(i,j);
                hxsy(abs(i-j)+1) = hxsy(abs(i-j)+1)+glcm(i,j); %0,n-1移到1,n
                hxy1 = hxy1-glcm(i,j)*log(max(px(i)*py(j),smooth));
                hxy2 = hxy2-px(i)*py(j)*log(max(px(i)*py(j),smooth));
                
                au = au+i*j*glcm(i,j);
                cp = cp+(i+j-mux(i)-muy(j))^4*glcm(i,j);
                cs = cs+(i+j-mux(i)-muy(j))^3*glcm(i,j);
                ct = ct+(i+j-mux(i)-muy(j))^2*glcm(i,j);
                co = co+(i-j)^2*glcm(i,j);
                cor = cor+(i*j*glcm(i,j)-mux(i)*muy(j))/max(sigx(i)*sigy(j),smooth);
                di = di+abs(i-j)*glcm(i,j);
                hm1 = hm1+glcm(i,j)/(1+abs(i-j));
                hm2 = hm2+glcm(i,j)/(1+(i-j)^2);
                idmn = idmn+glcm(i,j)/(1+(i-j)^2/5^2);
                idn = idn+glcm(i,j)/(1+abs(i-j)/5);
                if i ~= j
                    iv = iv+glcm(i,j)/(i-j)^2;
                end
            end
        end
        de = hxpy(:)'*hxpy(:);
        en = glcm(:)'*glcm(:);
        ent = entropy(glcm(:));
        
        imc1 = (hxy-hxy1)/max(max(hx,hy),smooth);
        imc2 = abs((1-exp(-2*(hxy2-hxy))))^0.5;
        mp = max(glcm(:));
        sa = mean(hxpy);
        se = entropy(hxpy);
        sv = var(hxpy);
        va = var(glcm(:));
        
        % 1-22
        feats = [feats;au];
        feats = [feats;cp];
        feats = [feats;cs];
        feats = [feats;ct];
        feats = [feats;co];
        feats = [feats;cor];
        feats = [feats;de];
        feats = [feats;di];
        feats = [feats;en];
        feats = [feats;ent];
        feats = [feats;hm1];
        feats = [feats;hm2];
        feats = [feats;imc1];
        feats = [feats;imc2];
        feats = [feats;idmn];
        feats = [feats;idn];
        feats = [feats;iv];
        feats = [feats;mp];
        feats = [feats;sa];
        feats = [feats;se];
        feats = [feats;sv];
        feats = [feats;va];
        
        % 23-33
        stats = grayrlprops({glrlms(:,:,s)});
        feats = [feats;stats'];
    end
    

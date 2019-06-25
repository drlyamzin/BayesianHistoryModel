function [prob_out,init_time,bslprep_time,bsloop_time,half_time,full_time,int_time]=calcBayesProb_nh(mu,params,mode,bessel_table,bessel_coords)
% model modification with no history effects

if nargin<5,
    mode = 'CPU';
end

%unpack
sgmax = params(1);% sigma for x
sgmay = params(2);% sigma for y
x0 = params(3);% perceptual bias for x
y0 = params(4);% perceptual bias for y


% if computing on CPU, use anonymous functions (1.convenience, 2.precision of besseli)
% also, Maths in "CPU" section is more readable, and mirrors exactly the
% operations in "GPU" section
if strcmp(mode,'CPU'),
    disp('THIS MODE IS CURRENTLY OUT OF USE');
    return
    
    h = a*s+b*r;% the only way history terms appear in calculations
    
    % unique conditions:
    all_contions = [mu h];
    [unq_cond,~,ind_orig_in_unique] = unique(all_contions,'rows');
    
    mu1 = unq_cond(:,1);
    mu2 = unq_cond(:,2);
    hist = unq_cond(:,3);
    
    nCond = size(unq_cond,1); 
    prob = zeros(length(unq_cond),1);%choose-1 probability, allocate array

    % function definitions (these don't change with loop iteration):
    % von Mises distribution
    Cx = 1/(2*pi*besseli(0,sgmax));
    Cy = 1/(2*pi*besseli(0,sgmay));
    px = @(x,mu) Cx * exp((sgmax)*cos(x-mu));
    py = @(x,mu) Cy * exp((sgmay)*cos(x-mu));
    
    parfor iCond = 1:nCond
        
        this_mu1 = mu1(iCond);
        this_mu2 = mu2(iCond);
        this_hist = hist(iCond);
        
        
        Cprior = 1/(2*pi*besseli(0,sgmapr*this_hist));
        prior = @(x,y) Cprior^2*exp(sgmapr*this_hist*(cos(x)-cos(y)));
        
        % numerator function -- note the order of (y,x); y has to come first
        % for integral2
        pnumf = @(y,x) px(x-x0,this_mu1).*py(y-y0,this_mu2).*prior(x,y);
        
        % integrals
        xmax = @(y) y;
        xmin = @(y) -y;
        
        abstol = 1e-4;
        reltol = 1e-3;
        pnum(iCond) = integral2(pnumf,0,pi,xmin,xmax,'AbsTol',abstol,'RelTol',reltol)+integral2(pnumf,-pi,0,xmax,xmin,'AbsTol',abstol,'RelTol',reltol);%a,b,c,d,u,v
        normc(iCond) =  integral2(pnumf,-pi,pi,-pi,pi,'AbsTol',abstol,'RelTol',reltol);%a,b,c,d,u,v
        
        prob(iCond) = pnum(iCond)/normc(iCond);
        
    end
    
    
else% GPU: 
    % prepare arrays for integration, use precalculated bessel function values
    
    % unique conditions:
    all_contions = mu;
    [unq_cond,~,ind_orig_in_unique] = unique(all_contions,'rows');
    mu1 = unq_cond(:,1);
    mu2 = unq_cond(:,2);
    nCond = size(unq_cond,1);
    
    prob = zeros(length(unq_cond),1,'gpuArray');%choose-1 probability, allocate array

    % -I. Bessel function values
    bessel_table = gpuArray(bessel_table);
    bessel_coords = gpuArray(bessel_coords);
    [~,idx_x] = min(abs(bessel_coords-sgmax));
    [~,idx_y] = min(abs(bessel_coords-sgmay));
    besx = bessel_table(idx_x);
    besy = bessel_table(idx_y);
    
    % 
    % sequence of operations is such that it takes less GPU memory (which
    % is unexpectedly an issue)
    
    % 0. create X3 and Y3 arrays, used in all the calculations below
    N=300;
    xvals = gpuArray.linspace(-pi, pi, N);
    yvals = gpuArray.linspace(-pi, pi, N);
    [X, Y] = meshgrid(xvals, yvals);
    xspacing = 2*pi/N;
    yspacing = 2*pi/N;
    
    % I. compute the mu1-containing term
    this_mu1(1,1,:) = gpuArray(mu1+x0); 
    
    % II. compute the mu2-containing term
    this_mu2(1,1,:) = gpuArray(mu2+y0); 
    
    % III. keep only their sum
    tmp12 = sgmax*cos(bsxfun(@minus,X,this_mu1))+sgmay*cos(bsxfun(@minus,Y,this_mu2)); 
    clear this_mu1 this_mu2

    % IV. calculate a coefficient for the full expression
    C1C2C3 = 1./(2*pi*besx * 2*pi*besy * ones(1,1,nCond,'gpuArray') );
    
    % V. calculate the full expression
    tmp3 = bsxfun(@times,exp(tmp12),C1C2C3);
    
    
    % compute mask for integration
    mask = gpuArray(abs(X)<=abs(Y));% half-space over which to integrate

    % integral over the full space
    Z1 = trapz(tmp3)*yspacing;
    int_full = trapz(Z1)*xspacing; clear Z1
    
    % integral over the half space (|X|<|Y|)
    func_half = bsxfun(@mtimes,tmp3,mask); clear tmp3 mask
    Z1_half = trapz(func_half)*yspacing; clear func_half
    int_half = trapz(Z1_half)*xspacing; clear Z1_half
    
    
    prob = squeeze(int_half./int_full);
    
end


prob_out = gather(prob(ind_orig_in_unique));
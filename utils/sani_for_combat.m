% this_dat = sani_for_combat(this_dat, sid)
%
% "Sanitizes" data for combat harmonization. Drops zero or low variance
% voxels that may interfere with combat harmonization, and converts data to
% type double. Without these steps combat harmonization can be corrupted in
% various ways.
%
% Uses a Gaussian approximation for estimating 'low variance' outliers. An
% inverse gamma would be better, but hasn't been implemented yet.
% 
function this_dat = sani_for_combat(this_dat, sid)
    % drop zero and approximately zero (low variance) voxels. These can
    % break combat normalization. See comment below.    
    [uniq_sid, exp_sid] = unique(sid,'stable');
    cmat = []; % within subject centering matrix
    mumat = []; % within subject averaging matrix;
    n = [];
    for j = 1:length(uniq_sid)
        this_blk = uniq_sid(j);
        this_n = sum(this_blk == sid);
        n = [n(:); this_n*ones(this_n,1)];
        cmat = blkdiag(cmat, eye(this_n) - 1/this_n);
        mumat = blkdiag(mumat, ones(this_n)*1/this_n);
    end
    
    vdat = ((this_dat.dat*cmat).^2)*mumat;
    vdat = vdat(:,exp_sid); % this is the voxel-wise variance for each subject
    
    % this thresholding is important. We don't want zeros, but we also
    % don't want approximatley zero variance estimates, which will explode
    % our data when combat standardizes data. Theoretically these violate 
    % the parametric assumptions combat makes regarding the variance 
    % distribution following an inverse gamma pdf. The precise threshold we 
    % want isn't obvious, so we take a standardized approximation and 
    % remove anything <0.1% likely according to a Gaussian distribition.
    % The inverse gamma distribution isn't defined in matlab, but future
    % versions of this script could manually implement an inverse gamma and
    % use that instead.
    lvdat = log(vdat(vdat(:) > 0)); % normalized variance estimates
    s = std(lvdat(:));
    m = mean(lvdat(:));
    thresh = m - icdf('norm',0.999,0,1)*s; % flag anything less than 0.1% likely
    thresh = exp(thresh); % transform back to vdat space
    this_dat.dat(any(vdat<thresh,2),:) = 0; 
    
    this_dat = this_dat.remove_empty();
    
    % convert to double to avoid convergence issues with combat
    this_dat.dat = double(this_dat.dat);
end
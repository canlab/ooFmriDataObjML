function err = get_hinge_loss(yFitObj)
    assert(isa(yFitObj,'yFit'),'get_hinge_loss() takes only yFit objects as input');
    assert(~isempty(yFitObj.yfit_raw), 'no raw scores available, cannot compute hinge loss');
    assert(~isempty(yFitObj.classLabels),'No classLabel vector provided, cannot disambiguiate columns of yfit_raw.');
    
    uniq_classes = categorical(unique(yFitObj.classLabels(:),'stable'))'; %1 x n vector of class labels, assumes order matches columns of yfit_raw
    
    yfit_raw = yFitObj.yfit_raw;
    
    if length(uniq_classes) == 2 % binary hinge
        Y = zeros(size(yFitObj.Y));
        
        assert(1 == size(yfit_raw,2), ...
            sprintf('Expected on score per obseation, but recieved %d scores for each', size(yfit_raw,2)));
        
        Y(categorical(yFitObj.Y) == uniq_classes(1)) = -1;
        Y(categorical(yFitObj.Y) == uniq_classes(2)) = 1;
        
        margin = Y.*yfit_raw;
    else
        assert(size(uniq_classes,2) == size(yfit_raw,2), ...
            sprintf('Expected on score per class, but recieved %d scores for %d classes', size(yfit_raw,2),size(uniq_classes,2)));
        % crammer and singler multiclass loss generalization
        correctClass = categorical(yFitObj.Y)  == repmat(uniq_classes, size(yFitObj.Y,1), 1);

        assert(all(sum(correctClass,2) == 1), ...
            sprintf('Unexpected number of correct class labels for entry %d.\n', find(sum(correctClass,2)~=1)));
        
        % we will use linear indexing which operates column wise, but we
        % want operations to be row wise, so let's rotate a bunch of
        % matrices.
        yfit_raw = yfit_raw';
        correctClass = correctClass';
        
        margin = yfit_raw(correctClass);   
        yfit_raw(correctClass) = [];
        
        % rotate back to n x 1
        margin = margin(:);
        yfit_raw = reshape(yfit_raw, length(uniq_classes)-1, size(yFitObj.yfit,1))';
        
        margin = margin - max(yfit_raw, [], 2);
    end
    err = max(0, 1 - margin(:));
    
    err = sum(err)./length(yFitObj.classLabels);
end
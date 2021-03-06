function err = get_hinge_loss(yFitObj)
    assert(isa(yFitObj,'yFit'),'get_hinge_loss() takes only yFit objects as input');
    assert(~isempty(yFitObj.yfit_raw), 'no raw scores available, cannot compute hinge loss');
    if size(yFitObj.yfit_raw,2) ~= 1 % this will happen for multinomial predictions
        error('Multiple raw scores found. Cannot compute hinge loss. Try metrics that operate on predicted labels like @get_f1_macro');
    end

    if all(ismember(yFitObj.Y,[-1,1]))
        Y = yFitObj.Y;
    else
        error('Y values (targets) should be [-1,1] for use with @get_hinge_loss computation. ');
    end
    
    yfit_raw = yFitObj.yfit_raw;
    
    err = max(0, 1 - Y .* yfit_raw)./2; % division by two follows matlab's convention, although I'm not sure it's standard
    err = sum(err(:));    
end
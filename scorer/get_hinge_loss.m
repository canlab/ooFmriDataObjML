function err = get_hinge_loss(yFitObj)
    assert(isa(yFitObj,'yFit'),'get_hinge_loss() takes only yFit objects as input');
    assert(~isempty(yFitObj.yfit_raw), 'no raw scores available, cannot compute hinge loss');
    if size(yFitObj.yfit_raw,2) == 1
        warning('Multiple raw scores found. Are you sure you want to be using hinge loss?');
    end

    if all(ismember(yFitObj.Y,[-1,1]))
        Y = yFitObj.Y;
    else
        warning('Y values (targets) should be [-1,1] for hinge_loss computation. Autoconverting {x <= 0 -> 1, x > 0 -> 1}, make sure this is appropriate.');
        Y = double(yFitObj.Y > 0);
        Y(Y==0) = -1;
    end
    
    yfit_raw = yFitObj.yfit_raw;
    
    err = max(0, 1 - Y .* yfit_raw)./2; % division by two follows matlab's convention, although I'm not sure it's standard
    err = sum(err(:));    
end
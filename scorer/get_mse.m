function err = get_mse(yFitObj)
    assert(isa(yFitObj,'yFit'),'get_mse() takes only yFit objects as input');

    if ~isempty(yFitObj.yfit)
        err = mean((yFitObj.yfit - yFitObj.Y).^2);
    elseif ~isempty(yfitObj.yfit_raw)
        warning('yfit not found, using yfit_raw');
        err = mean((yFitObj.yfit_raw - yFitObj.Y).^2);
    else
        error('Neither yfit nor yfit_raw found.');
    end
end
        
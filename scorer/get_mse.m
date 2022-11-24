function err = get_mse(yFitObj)
    assert(isa(yFitObj,'yFit'),'get_mse() takes only yFit objects as input');

    %if isvector(yFitObj.Y)
        meanFun = @(x1,x2)(mean((x1 - x2).^2));
    %else
    %    meanFun = @(x1,x2)(mean(arrayfun(@(ind)(mean((x1(:,ind) - x2(:,ind)).^2,1)),1:size(x2,2)),2));
    %end
    
    if ~isempty(yFitObj.yfit)
        err = meanFun(yFitObj.yfit,yFitObj.Y);
    elseif ~isempty(yFitObj.yfit_raw)
        warning('yfit not found, using yfit_raw');
        err = meanFun(yFitObj.yfit_raw,yFitObj.Y);
    else
        error('Neither yfit nor yfit_raw found.');
    end
end
        
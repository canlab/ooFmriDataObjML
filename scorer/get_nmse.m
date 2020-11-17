function err = get_nmse(yFitObj)
    assert(isa(yFitObj,'yFit'),'get_nmse() takes only yFit objects as input');

    if isempty(yFitObj.yfit_null)
        yFitObj = yFitObj.do_null();
    end

    err = mean((yFitObj.yfit - yFitObj.Y).^2)/mean((yFitObj.yfit_null - yFitObj.Y).^2);
end
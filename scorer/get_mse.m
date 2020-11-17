function err = get_mse(yFitObj)
    assert(isa(yFitObj,'yFit'),'get_mse() takes only yFit objects as input');

    err = mean((yFitObj.yfit - yFitObj.Y).^2);
end
        
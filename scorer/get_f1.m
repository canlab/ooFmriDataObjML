% F1 score for binary targets
function err = get_f1(yFitObj)
    assert(isa(yFitObj,'yFit'),'get_f1() takes only yFit objects as input');

    if sum(yFitObj.yfit == 1) > 0
        true_positive = (yFitObj.Y == 1).*(yFitObj.yfit == 1);
        precision = sum(true_positive)/sum(yFitObj.yfit == 1);
        
        recall = sum(true_positive)/sum(yFitObj.Y == 1); % also known as the true positive rate
        
        f1 = 2 * precision * recall / (precision + recall);
    else
        warning('All labels assigned to a single class. Cannot compute precision, F1=0.');
        f1 = 0;
    end

    err = f1;
end
% F1 score for binary targets
function err = get_f1_macro(yFitObj)
    assert(isa(yFitObj,'yFit'),'get_f1() takes only yFit objects as input');

    if ~iscategorical(yFitObj.Y) 
        warning('Observed class labels are not categorical, but f1 scores only make sense for evaluating classification performance. Attempting naive conversion.');
    end
    if ~iscategorical(yFitObj.yfit)
        warning('Predicted class labels are not categorical, but f1 scores only make sense for ealuating classification performance. Attempting naive conversion.');
    end
    
    % convert to numeric if not already
    Y = categorical(yFitObj.Y);
    yfit = categorical(yFitObj.yfit);
    
    labels = categorical(yFitObj.classLabels);
    
    f1 = 0;
    for i = 1:length(labels)
        true_positive = (Y == labels(i)).*(yfit == labels(i));
        precision = sum(true_positive)/sum(yfit == labels(i));
    
        recall = sum(true_positive)/sum(Y == labels(i)); % also known as the true positive rate
    
        f1 = f1 + 2 * precision * recall / (precision + recall);
    end
    
    err = f1/length(labels); % average f1
end

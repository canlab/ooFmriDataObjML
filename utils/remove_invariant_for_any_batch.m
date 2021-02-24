% removes voxels that are empty iny any batch indicated by batch_id

function [obj, isinvariant] = remove_invariant_for_any_batch(obj, batch_id)
    [~,~,batch_id] = unique(batch_id,'stable');
    uniq_batch_id = unique(batch_id);
    n_batches = length(uniq_batch_id);
    isinvariant = zeros(size(obj.dat,1),1);
    
    for i = 1:n_batches
        this_batch = uniq_batch_id(i);
        wh = std(obj.dat(:,this_batch == batch_id),[],2);
        isinvariant(wh == 0) = 1;
    end
    
    obj.dat(isinvariant == 1,:) = 0;
    obj = obj.remove_empty();
end
% obj = features(dat, metadata)
% this function extends double's to attach some metadata (e.g. subject
% block identifiers) to vectors. size(metadata,1) must equal size(data,1)
% size(data,1). In other words, you need an identifier for each row of your
% data.
%
% The object will persist across subscripting, subreferencing, and 
% concatenation but other operations will promote the object back to it's
% parent type double, erasing the metadata property. This should be fine
% for managing block identifiers for cross validation though.
%
% see here for details on the coding methods used below
% https://www.mathworks.com/help/matlab/matlab_oop/built-in-subclasses-with-properties.html

classdef features < double
    properties
        metadata;
    end
    
    methods
        function obj = features(dat, metadata)
            if nargin == 0
                dat = 0;
                metadata = [];
            elseif nargin == 1
                metadata = [];
            elseif ~isempty(metadata)
                [n1,~] = size(dat);
                [n2,~] = size(metadata);
                assert(n1 == n2, 'dat and metadata dimension mismatch');
            end
            
            obj = obj@double(dat);
            obj.metadata = metadata;
        end
        
        function sref = subsref(sref, s)
            switch s(1).type
                case '.'
                   switch s(1).subs
                      case 'metadata'
                          if length(s)<2
                             sref = sref.metadata;
                          else
                             sref = subsref(sref.(s(1).subs), s(2:end));
                          end
                      case 'dat'
                         d = double(sref);
                         if length(s)<2
                            sref = d;
                         elseif length(s)>1 && strcmp(s(2).type,'()')
                            sref = subsref(d,s(2:end));
                         end
                      otherwise
                         error('Not a supported indexing expression')
                   end
                case '()'
                    d = double(sref);
                    newd = subsref(d, s);

                    if size(sref.metadata,1) == size(sref,1)
                        % assume we only take corresponding rows of
                        % metadata table, but all column entries, even if
                        % specific column entries were specified for dat.
                        md_s = s;                        
                        md_s.subs(2) = {':'};
                        md_s.subs = md_s.subs(1:2);
                        
                        if size(sref.metadata,2) == size(sref,2)
                            warning('sref.metadata appears to have as many columns as you have features. This is unexpected and may result in unexpected feature metadata. Please check results of features subscripting.');
                        end
                        
                        newmd = subsref(sref.metadata, md_s);
                    else
                        newmd = sref.metadata;
                    end

                    sref = features(newd, newmd);
                case '{}'
                    error('Not a supported indexing expression');
            end
        end
        
        function obj = subsasgn(obj, s, b)
            switch s(1).type
                case '.'
                   switch s(1).subs
                      case 'metadata'
                         obj.metadata = b;
                      case 'data'
                         if length(s)<2
                            obj = features(b,obj.DataString);
                         elseif length(s)>1 && strcmp(s(2).type,'()')
                            d = double(obj);
                            newd = subsasgn(d,s(2:end),b);
                            obj = features(newd,obj.DataString);
                         end
                      otherwise
                         error('Not a supported indexing expression')
                   end
                case '()'
                    d = double(obj);
                    newd = subsasgn(d, s, double(b));

                    if ~isa(obj,'features')
                        obj = features(d,[]);
                    end
                    
                    if size(obj.metadata,1) == size(obj,1)
                        newmd = subsasgn(obj.metadata, s, b.metadata);
                    else
                        newmd = obj.metadata;
                    end

                    obj = features(newd, newmd);
                case '{}'
                    error('Not a supported indexing expression')
            end
        end
        
        function obj = horzcat(varargin)
            d1 = cellfun(@double,varargin,'UniformOutput',false);
            data = horzcat(d1{:});
            
            for i = 1:length(varargin)
                if ~isa(varargin{i},'features')
                    varargin{i} = features(varargin{i});
                end
            end
            
            newmd = cellfun(@(x1)(x1.metadata),varargin,'UniformOutput',false);
            newmd = horzcat(newmd{:});
            obj = features(data, newmd);
        end
      
        function obj = vertcat(varargin)
            d1 = cellfun(@double,varargin,'UniformOutput',false);
            data = vertcat(d1{:});
            newmd = eval(['cellfun(@(x1)(' class(varargin{1}.metadata) '(x1.metadata)), varargin,''UniformOutput'',false);']);
            newmd = vertcat(newmd{:});
            obj = features(data, newmd);
        end
    end
end
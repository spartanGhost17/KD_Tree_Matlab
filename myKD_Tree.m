classdef myKD_Tree   
       methods(Static)
            function m = fit(train_examples, train_labels, k)

            % start of standardisation process (increase model accuracy)
            %% calculate mean of all features in table
			m.mean = mean(train_examples{:,:});
            %% calculate standard deviation (how much feature differ from mean) of all features in table
			m.std = std(train_examples{:,:});
            %% for however many rows we have in our train example matrix apply z-score standardization to avoid individual large features drowning out those with smaller footprint in euclidean calculation
            for i=1:size(train_examples,1)
                %% substract mean from current training example to have feature centered at zero
				train_examples{i,:} = train_examples{i,:} - m.mean;
                %% scales down features with big spreads and scales up features with small spreads, all feature std will be 1
                train_examples{i,:} = train_examples{i,:} ./ m.std;
            end
            % end of standardisation process
            
            %% copies train examples as a new train_examples field for the returned m structure
            m.train_examples = train_examples
            %% copies train examples as a new train_labels field for the returned m structure
            m.train_labels = train_labels;
            %% copies train k (number of nearest neighbours) as a new k field for the returned m structure
            m.k = k;
            %% number of training examples used for tree
            m.num_of_Tr_Ex = size(train_examples,1);
            % determine total number of dimenssions (columns) in set
            m.number_Of_Dimenssions = size(train_examples,2); 
            % initialize dimenssion pointer  
            m.current_dimension = 1; 
            
            
  
            
            
            %% initialize root
            my_r.train_examples = m.train_examples;
            my_r.class = m.train_labels ;
            my_r.Children = {};
            
            % later will be change to leaf when no more split available
            my_r.type = 'node';
            %% memorize original rows and colums
            my_r.original_rows = [];
            my_r.original_cols = [];
            %vmemorize median to know how to descend in classification
            %phase
            my_r.median_of_dimension = 0;
            % initialize dimenssion pointer              
            my_r.current_dimension = 1;
            my_r.stopDecent = 0;
            
            %record parent
            my_r.myParent = {};
            
            %record side
            my_r.side = 'root';
            
            % record best distance, usefull in classification phase
            my_r.best_dist = 0;
            my_r.closest_point_to_test_value = [];
            my_r.index_closest_point_to_test_value = [];
            %% number of nodes in tree
            my_r.num_of_nodes = 1;
            m.myKD_Tree = myKD_Tree.BuildTree(m, my_r);
            m.number_of_nodes = myKD_Tree.number_of_nodes(m.myKD_Tree,0); 
            m.tree_height = myKD_Tree.find_height(m.myKD_Tree,0)
            
            end
            
            %% the build tree function builds a KD-Tree by using the median of the data in node as spliting point
            %% median value will be used for traversal
            %% returns a node struct with links to all decendents            
            function node = BuildTree(m, node)
                
                   % if we have reached the end of our dimension count reset the counter to initial dimension 1 
                   % (dimension means, column)
                   if node.current_dimension > m.number_Of_Dimenssions
                       node.current_dimension = 1;
                   end
                   % copy current dimensional data in node for median calculation 
                   local_data = node.train_examples{:,node.current_dimension};
                   node.median_of_dimension = median(local_data);
                   % split data into less than median to left and greater
                   % than or equal to median on right (i.e tree property small on left, big on right)
                   [rows_left, cols_left] = find(node.train_examples{:,node.current_dimension}<node.median_of_dimension);                   

                   %% if can't find data less than median at current dimension (column) don't add new node to tree,
                   %% else add node to tree
                   if isempty(rows_left) | size(node.train_examples,1)==1
                       % leaf node was reached erase children copied from
                       % branch
                       node.Children = {};
                       % set type to leaf
                       node.type = 'leaf';
                       return
                   else
                      % 1 is left child 
                      node.Children{1} = node
                      %% move to next dimension of data set (next column)
                      node.Children{1}.current_dimension = node.current_dimension+1;
                      % copy all training points from split UNTILL point
                      node.Children{1}.train_examples = node.train_examples(rows_left,:);
                      node.Children{1}.class = node.class(rows_left);
                      node.Children{1}.median_of_dimension = 0;
                      node.Children{1}.type = 'node';
                      % keep track of rows of data for current index
                      node.Children{1}.original_rows = rows_left;
                      % memorize parent
                      node.Children{1}.myParent = node;
                      
                      %m.num_of_nodes = m.num_of_nodes+1;
                      node.Children{1}.stopDecent = 0;
                      %branche
                      node.Children{1}.side = 'L';
                      

                   end

                   
                   
                   [rows_right, cols_right] = find(node.train_examples{:,node.current_dimension}>=node.median_of_dimension);
                   %% if can't find data greater than or equal to median at current dimension (column) don't add new node to tree,
                   %% else add node to tree                   
                   if isempty(rows_right) | size(node.train_examples,1)==1
                       % leaf node was reached erase children copied from
                       % branch
                       node.Children = {};
                       % set type to leaf
                       node.type = 'leaf';
                       return
                   else
                       node.Children{2} = node;
                       %% move to next dimension of data set (next column)
                       node.Children{2}.current_dimension = node.current_dimension+1;
                       % copy all training points from split point down
                       node.Children{2}.train_examples = node.train_examples(rows_right,:);
                       node.Children{2}.class = node.class(rows_right);
                       node.Children{2}.median_of_dimension = 0;
                       node.Children{2}.type = 'node';
                       % keep track of rows of data for current index
                       node.Children{2}.original_rows = rows_right; 
                       % memorize parent
                       node.Children{2}.myParent = node;
                      
                       %m.num_of_nodes = m.num_of_nodes+1;
                       node.Children{2}.stopDecent = 0;
                       % branch side
                       node.Children{2}.side = 'R';
                       

                   end                   
                   % go left subtree (recursive call)                 
                   node.Children{1} = myKD_Tree.BuildTree(m, node.Children{1});
                   % go right subtree (recursive call)
                   node.Children{2} = myKD_Tree.BuildTree(m, node.Children{2});
                
            end
            
        %% predict a label for our testing point 
        %% and performs Z score standardization on test point (avoid drowing feature if one's scale exceeds the other
        %% allows to not lose important details )
        %% return prediction label
            
        function predictions = predict(m, test_examples)
            %% initialise categorical array predictions
            predictions = categorical;
            %% loop through all rows in test_examples table (testing examples)
            for i=1:size(test_examples,1)
                
                fprintf('classifying example example %i/%i\n', i, size(test_examples,1));
                
                %% copy current row data from test_examples table in this_test_example 
                this_test_example = test_examples{i,:};
                
                % start standardisation process (increase model accuracy)
                %% substract mean from current training example to have feature centered at zero
                this_test_example = this_test_example - m.mean;
                %% scales down features with big spreads and scales up features with small spreads, all std will be 1                
                this_test_example = this_test_example ./ m.std;
                % end of standardisation process
                %% copy prediction returned from predict_one function in this_prediction by passing the model m and the current standardized test data this_test_example
                this_prediction = myKD_Tree.predict_one(m, this_test_example);
                %% add prediction to prediction array
                predictions(end+1,1) = this_prediction;
            
            end           
        end
        
        %% predict one label for a single testing example
        function prediction = predict_one(m, this_test_example)
            
            %% descend tree, copy leaf node and take it's prediction 
            [neighbors, best, ~] = myKD_Tree.descend_tree(m.myKD_Tree, this_test_example, 1, m, m.myKD_Tree([]),0);
            
            %% get last best guess in cell array neigbors
            node_best_guess_neighbor = neighbors(1,1);
            
            %% copy value of most common categorical class associated with current test example in prediction
            prediction = node_best_guess_neighbor.class(node_best_guess_neighbor.index_closest_point_to_test_value);
        
        end 
        
        %% apply pithagoras theorem and store final result euclidian dist in distance
        function distance = calculate_distance(p, q)
            
            %% difference between elements in array p and q store result in differences
			differences = q - p;
            %% square root of differences store resulting array in squares
            squares = differences .^ 2;
            %% sum squares array
            total = sum(squares);
            %% square root of distance
            distance = sqrt(total);
        
        end 
        
        %% helper function compare best distance vs distance to parent dimensional split plane from test point
        %% return a flag
        function visit_sibling = find_Best_distance(query_p_dim_coord, node_median_of_dim, best_distance)
            visit_sibling = 'skip';
            % query point dimesional coordinate minus mean of parent
            if  abs(query_p_dim_coord - node_median_of_dim) < best_distance
                visit_sibling = 'visit';
            end
        end
        
        %% descend the tree and return a list of leaf nodes
        function [node_neighbors, best_distance, N_Visited] = descend_tree(node, this_test_example, best_distance, m, node_neighbors, N_Visited)            
            
            if isempty(m.myKD_Tree)
                fprintf("Sorry can't process empty tree \n");
                return;
            end
            %% if leaf node was reached return node
            if isempty(node.Children)
                        test_example_dimension_value = this_test_example(:,node.current_dimension );
                        temp_best_dists = [];
                        % keep track of best distance in current leaf data
                        % set
                        for data_point_in_leaf = 1:size(node.train_examples,1)
                            best_distance = myKD_Tree.calculate_distance(test_example_dimension_value, node.train_examples{data_point_in_leaf,:});
                            temp_best_dists(end+1) = best_distance;
                        end
                        % find smallest distance to test point in dists from node dataset to test point and its index
                        [min_best_dist, index] = min(temp_best_dists);
                        % record best distance in current dimensional section
                        node.best_dist = min_best_dist;
                        % record closest point in current section to test point
                        node.closest_point_to_test_value = node.train_examples{index,:};
                        node.index_closest_point_to_test_value(:,1) = index;
                        % memorize leaf during recursion
                        node_neighbors = myKD_Tree.memorize_leaves(node_neighbors, node,m.k);
                        % change best distance to smallest distance found
                        % so far
                        best_distance = node_neighbors(1,1).best_dist;
                        
                        % leave leaf node, (unwind recursion)        
                        return;
                        
                    %% since leaf node has been reached check if distance from current node dimension value minus value of parent plane split value is less than 
                    %% current best distance
            
            else
                test_example_dimension_value = this_test_example(:,node.current_dimension );
                %% if data in current_dimension is less than median_of_dimension value in node, descend left (left side holds small values right side holds higher values)
                if test_example_dimension_value < node.median_of_dimension
                                       
                    %% recursively call descend_tree to move down the tree
                    [node_neighbors, best_distance, N_Visited] = myKD_Tree.descend_tree(node.Children{1}, this_test_example, best_distance, m, node_neighbors, N_Visited);
                    
                    %% on unwinding recursion check for a closer point to
                    %% test point
                    [node_neighbors2,best_distance2,N_Visited2] = myKD_Tree.find_better_match(m, this_test_example, node_neighbors, best_distance, node, node.Children{1}, N_Visited);
                    node_neighbors = node_neighbors2; best_distance = best_distance2; N_Visited = N_Visited2;
                    
                %% else descend right child (bigger values on right side nodes)    
                else
              
                    %% recursively call descend_tree to move down the tree
                    [node_neighbors, best_distance, N_Visited] = myKD_Tree.descend_tree(node.Children{2}, this_test_example, best_distance, m, node_neighbors, N_Visited);
                    
                    %% on unwinding recursion check for a closer point to
                    %% test point                    
                    [node_neighbors2,best_distance2,N_Visited2] = myKD_Tree.find_better_match(m,this_test_example,node_neighbors,best_distance,node, node.Children{2},N_Visited);
                    node_neighbors = node_neighbors2; best_distance = best_distance2; N_Visited = N_Visited2; 
                end
            end
            
        end
        
        %% used when unwinding the the recursion of the tree to find better
        %% match after finding first leaf node
        %% return array of leaves, current best distance and number of visited nodes in closer point (better match) search 
        function [array_of_leaves,best_distance2,N_Visited2] = find_better_match(m ,this_test_example,node_neighbors, best_distance ,node, visited_child, N_Visited)
                    
                    array_of_leaves = node_neighbors;
                    best_distance2 = best_distance;
                    N_Visited2 = N_Visited;
                    %%  check if distance (absolute value) from current dimension value of current parent is less than 
                    %%  minus value of parent plane split value is less than 
                    %% current best distance
                     test_example_dimension_value = this_test_example(:,node.current_dimension);
                     if strcmp(myKD_Tree.find_Best_distance(test_example_dimension_value, node.median_of_dimension, best_distance), 'visit')
                         %% only investigate other side if there are more than one child
                         if size(node.Children,2)>1 
                                % recently visited child was on the left, visit child on opposite side 
                                 if strcmp( visited_child.side, 'L')
                                     N_Visited2 = N_Visited2 + 1; % add one to visited node count
                                     [node_neighbors, best_distance, N_Visited] = myKD_Tree.descend_tree(node.Children{2}, this_test_example, best_distance2, m, node_neighbors, N_Visited2);
                                     array_of_leaves = node_neighbors; best_distance2 = best_distance; N_Visited2 = N_Visited;
                                 % recently visited child was on the right, visit child on opposite side    
                                 elseif strcmp( visited_child.side, 'R')
                                     N_Visited2 = N_Visited2 + 1;% add one to visited node count
                                     [node_neighbors, best_distance, N_Visited] = myKD_Tree.descend_tree(node.Children{1}, this_test_example, best_distance, m, node_neighbors, N_Visited2);
                                     array_of_leaves = node_neighbors; best_distance2 = best_distance; N_Visited2 = N_Visited;
                                 end
                                 
                         else
                            % if node to investigate has less than 1 child, return  
                            return;
                         end
                     else
                         % condition not met to visit other side of parent
                         return;
                     end            
        end
        
        %% find number of nodes in the tree
        function number_of_nodes = number_of_nodes(root_node, number_of_nodes)
            % if no children node exist, leave
            if isempty(root_node.Children)
                return;
            % explore tree    
            else
                % go left if median is small 
                number_of_nodes = 1 + myKD_Tree.number_of_nodes(root_node.Children{1}, number_of_nodes);
                % right if median is big
                number_of_nodes = 1 + myKD_Tree.number_of_nodes(root_node.Children{2}, number_of_nodes);   
            end
            
        end
        
        %% find tree height
        function height = find_height(root_node, height)
            if isempty(root_node.Children)
                return;
            
            else
                height = 1 + max([myKD_Tree.find_height(root_node.Children{1}, height), myKD_Tree.find_height(root_node.Children{2}, height)]);
            end
            
        end
        
        %% retain visited leaves
        %% return sorted array of visited leaf node sorted based on best distance to test point
        function array_of_leaves = memorize_leaves(struct_Array, leaf, k)
            %% record best node
            array_of_leaves = struct_Array;
            %% if sturct array of sorted leaves is less than k (number of neighbors)
            %% add leaf to array and sort it based on best distance
            if size(struct_Array,1)< k
                % add leaf to struct array of leaves
                struct_Array(end+1,1) = leaf;
                array_of_leaves = struct_Array;
                
                T = struct2table(array_of_leaves,'AsArray',true); % convert the struct array to a table
                sortedT = sortrows(T, 'best_dist'); % sort the table by 'best_dist'
                sortedS = table2struct(sortedT); % change it back to struct array 
                array_of_leaves = sortedS;
                
            %% else if we already filled the queue  
            else
                % find all nodes with distance greater than (worst then) new
                % proposed distance
             
                [~, col] = find([struct_Array(:,1).best_dist] > leaf.best_dist);
                if ~isempty(col)
                    % find max value returned 
                    % replace max value with new best node                
                    [~, index] = max([struct_Array(col,1).best_dist]);
                    % get original max value corresponding index  
                    original_index = col(:,index);
                    % replace worst result with new leaf
                    struct_Array(original_index,1) = leaf;
                    %%______________________________________
                    % sort back array based on best distance 
                    T = struct2table(struct_Array,'AsArray',true); % convert the struct array to a table
                    sortedT = sortrows(T, 'best_dist'); % sort the table by 'best_dist'
                    sortedS = table2struct(sortedT); % change it back to struct array 
                    % copy back sorted list
                    array_of_leaves = sortedS;  

                end
                
            end

        end
                
   end       
 
end


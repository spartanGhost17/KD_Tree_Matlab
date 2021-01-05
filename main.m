% read the iris dataset:
data = readtable('iris.csv');

% examples for 25%:
nTest = round(0.25 * size(data,1));
% re-seed Matlab's random number generator:
rng(1);
% shuffle the data and create a testing dataset and a training dataset:
data_shuffled = data(randperm(size(data,1)), :);
data_test = data_shuffled(1:1:nTest, :);
size(data_test);
data_train = data_shuffled(nTest+1:1:end, :);
size(data_train);
% separate the examples and the labels for the testing dataset:
test_labels = categorical(data_test{:,'species'});
test_examples = data_test;
test_examples(:,'species') = [];
% separate the examples and the labels for the training dataset:
train_labels = categorical(data_train{:,'species'});
train_examples = data_train;
train_examples(:,'species') = [];

% fit our model with training data
model = myKD_Tree.fit(train_examples, train_labels, 10)

%predict label with resulting kd_tree
make_prediction = myKD_Tree.predict(model, test_examples);
% compute confusion matrix
[confusion_mat2,order] = confusionmat(test_labels, make_prediction)

% calculate the overall classification accuracy
% by diving the sum of correctly predicted values(along the diagonal of confusion_mat)
% and sum of all other values in confusion mat:
p_accuracy2 = sum(diag(confusion_mat2)) / sum(confusion_mat2(1:1:end))


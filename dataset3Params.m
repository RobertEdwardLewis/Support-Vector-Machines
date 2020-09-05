function [C, sigma,u] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C=1;
sigma=0.3;
Cc = [0.01 0.03 0.1 0.3 1 3 10 30];
sigmaa = [0.01 0.03 0.1 0.3 1 3 10 30];
m=length(Cc);
% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(doublexe(predictions ~= yval))
%
results=zeros(length(Cc) * length(sigmaa), 3);
row=1;
for i = 1:m;
	for j =1:m;
		model= svmTrain(X, y, Cc(i),@(x1, x2) gaussianKernel(x1, x2, sigmaa(j))); 
		predictions=svmPredict(model,Xval);
		error=mean(double(predictions ~= yval));
		
		results(row,:)=[Cc(i) sigmaa(j) error];
		row=row+1;
end



[dummy p] = min(results(:,3));
C=results(p,1);
sigma=results(p,2);
u=results
end
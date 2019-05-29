%% Number of repeated times on experiments
nFold = 2;

%% Classificaiton task or Regression task
if flagRC
    nMet = 5;
else
    nMet = 3;
end

%% Initialize AGILE
opts.flagRC  = flagRC;   % classification (true) or regression (false)
opts.alpha   = 10^0;     % regularization parameter on row-sparsity
opts.beta    = 10^0;     % regularization parameter on view consistency
opts.gamma   = 10^0;     % regularization parameter on group-sparsity
opts.nIter   = 500;      % maximum number of main iterations
opts.absTol  = 10^-5;    % tolerance for teminating the iterative algorithm
opts.debug   = false;    % show debug information (true) or not (false)
opts.flagEta = 'line';   % line search ('line') or fixed ('fix') for opts.eta
opts.eta     = 10^-5;    % Learning rate for gradient descent

%% For data splition
para.trRt = 0.6;         % Ratio of training data
para.teRt = 0.2;         % Ratio of testing data
para.vaRt = 0.2;         % Ratio of validation data
para.pctL = 0.5;         % Ratio of labeled data in training set

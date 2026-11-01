import math

def gaussian_naive_bayes(X_train: list, y_train: list, X_test: list) -> list:
    """Returns a predicted class label for every test sample."""
    n_samples = len(X_train)
    n_features = len(X_train[0])
    
    classes = list(set(y_train))
    n_classes = len(classes)
    
    class_data = {c: [] for c in classes}
    for x, y in zip(X_train, y_train):
        class_data[y].append(x)
        
    priors = {}
    means = {}
    variances = {}
    
    for c in classes:
        X_c = class_data[c]
        n_c = len(X_c)
        priors[c] = n_c / n_samples
        
        means[c] = [sum(col) / n_c for col in zip(*X_c)]
        
        vars_c = []
        for j in range(n_features):
            col = [row[j] for row in X_c]
            mu = means[c][j]
            var = sum((val - mu) ** 2 for val in col) / n_c
            vars_c.append(var + 1e-9)
        variances[c] = vars_c
        
    predictions = []
    for x in X_test:
        best_class = None
        max_log_posterior = float('-inf')
        
        for c in classes:
            log_prior = math.log(priors[c])
            log_likelihood = 0.0
            
            for j in range(n_features):
                var = variances[c][j]
                mu = means[c][j]
                xj = x[j]
                
                term1 = -0.5 * math.log(2 * math.pi * var)
                term2 = -((xj - mu) ** 2) / (2 * var)
                log_likelihood += term1 + term2
                
            log_posterior = log_prior + log_likelihood
            
            if log_posterior > max_log_posterior:
                max_log_posterior = log_posterior
                best_class = c
                
        predictions.append(best_class)
        
    return predictions
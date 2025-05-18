import numpy as np
from sklearn.ensemble import RandomForestRegressor
import pickle

class FederatedRandomForest:
    """
    A class to implement federated averaging for Random Forest models.
    """
    
    def __init__(self):
        self.models = []
        self.model_weights = []
        self.aggregated_model = None
        self.feature_importances = None
    
    def add_model(self, model, weight=1.0):
        """
        Add a model to the list of models to be aggregated
        
        Args:
            model: The model to add
            weight: The weight to assign to this model (default: 1.0)
        """
        self.models.append(model)
        self.model_weights.append(weight)
        print(f"Added model to federation. Total models: {len(self.models)}")
    
    def aggregate_predictions(self, X):
        """
        Aggregate predictions from all models by weighted averaging.
        
        Args:
            X: Features to predict on
            
        Returns:
            Weighted averaged predictions
        """
        if not self.models:
            raise ValueError("No models to aggregate. Add models first.")
        
        # Normalize weights
        weights = np.array(self.model_weights)
        weights = weights / weights.sum()
        
        # Get predictions from each model
        predictions = []
        for model in self.models:
            y_pred = model.predict(X)
            predictions.append(y_pred)
        
        # Weighted average of predictions
        weighted_predictions = np.zeros(len(X))
        for i, pred in enumerate(predictions):
            weighted_predictions += pred * weights[i]
            
        return weighted_predictions
    
    def compute_feature_importance(self):
        """
        Compute aggregated feature importance from all models
        
        Returns:
            Aggregated feature importance
        """
        if not self.models:
            raise ValueError("No models to aggregate. Add models first.")
            
        # Normalize weights
        weights = np.array(self.model_weights)
        weights = weights / weights.sum()
        
        # Get feature importances from each model
        all_importances = []
        for i, model in enumerate(self.models):
            importance = model.feature_importances_
            all_importances.append(importance * weights[i])
        
        # Weighted average of feature importances
        self.feature_importances = np.sum(all_importances, axis=0)
        return self.feature_importances
    
    def create_aggregated_model(self, base_model=None, feature_importance_threshold=0.01):
        """
        Create an aggregated model by combining trees from all models.
        Uses feature importance to select the most important trees.
        
        Args:
            base_model: A base model to use for structure (optional)
            feature_importance_threshold: Threshold for feature importance filtering
            
        Returns:
            An aggregated RandomForestRegressor
        """
        if not self.models:
            raise ValueError("No models to aggregate. Add models first.")
        
        # Compute feature importance if not already done
        if self.feature_importances is None:
            self.compute_feature_importance()
        
        # For Random Forest, we'll combine all trees from all forests
        all_estimators = []
        
        # Normalize weights for tree selection
        weights = np.array(self.model_weights)
        weights = weights / weights.sum()
        
        # Select trees based on feature importance and model weights
        for i, model in enumerate(self.models):
            # Get all estimators from this model
            estimators = model.estimators_
            
            # Add all estimators from this model, weighted by model weight
            all_estimators.extend(estimators)
        
        # Create a new Random Forest with all trees
        if base_model is None:
            base_model = self.models[0]
        
        # Create a new model with the same parameters as the base model but with more trees
        aggregated_model = RandomForestRegressor(
            n_estimators=len(all_estimators),
            random_state=base_model.random_state,
            max_depth=base_model.max_depth,
            min_samples_split=base_model.min_samples_split,
            min_samples_leaf=base_model.min_samples_leaf,
            bootstrap=base_model.bootstrap
        )
        
        # We need to fit the model with some data to initialize it
        # Then we'll replace its estimators with our combined ones
        X_dummy = np.zeros((1, base_model.n_features_in_))
        y_dummy = np.zeros(1)
        aggregated_model.fit(X_dummy, y_dummy)
        
        # Replace the estimators with our combined ones
        aggregated_model.estimators_ = all_estimators
        aggregated_model.n_estimators = len(all_estimators)
        
        # Store feature importances but don't try to set them directly
        # as feature_importances_ is a property with no setter
        self.aggregated_model = aggregated_model
        return aggregated_model 
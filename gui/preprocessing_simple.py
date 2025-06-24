"""
Simplified Preprocessing Techniques for GUI
==========================================

This module contains simplified versions of bias mitigation preprocessing
techniques that work independently of the complex research framework.
"""

import numpy as np
import pandas as pd
from collections import Counter
from sklearn.utils import resample

class SimplePreprocessor:
    """Base class for simple preprocessing techniques"""
    
    def __init__(self, method_name, **kwargs):
        self.method_name = method_name
        self.params = kwargs
        
    def get_demographics_key(self, demo_dict):
        """Extract a simple demographic key for grouping"""
        # Use gender as the primary demographic attribute
        return demo_dict.get('gender', 0)
    
    def fit_transform(self, X, y, demographics):
        """Apply preprocessing and return transformed data"""
        raise NotImplementedError

class NoPreprocessor(SimplePreprocessor):
    """No preprocessing - returns data as-is"""
    
    def __init__(self):
        super().__init__("None (Baseline)")
    
    def fit_transform(self, X, y, demographics):
        return X, y, demographics

class SimpleRebalancing(SimplePreprocessor):
    """Simple rebalancing through oversampling minority groups"""
    
    def __init__(self, target_size_multiplier=2):
        super().__init__("Rebalancing")
        self.target_size_multiplier = target_size_multiplier
    
    def fit_transform(self, X, y, demographics):
        # Convert to numpy for easier manipulation
        X = np.array(X)
        y = np.array(y)
        
        # Group by demographics
        demo_keys = [self.get_demographics_key(demo) for demo in demographics]
        unique_groups = np.unique(demo_keys)
        
        # Find the maximum group size
        group_sizes = [np.sum(np.array(demo_keys) == group) for group in unique_groups]
        target_size = max(group_sizes) * self.target_size_multiplier
        
        # Resample each group to target size
        X_resampled, y_resampled, demo_resampled = [], [], []
        
        for group in unique_groups:
            group_mask = np.array(demo_keys) == group
            group_X = X[group_mask]
            group_y = y[group_mask]
            group_demos = [demographics[i] for i in range(len(demographics)) if group_mask[i]]
            
            # Resample with replacement to reach target size
            if len(group_X) > 0:
                indices = np.random.choice(len(group_X), size=target_size, replace=True)
                X_resampled.extend(group_X[indices])
                y_resampled.extend(group_y[indices])
                demo_resampled.extend([group_demos[i] for i in indices])
        
        return X_resampled, y_resampled, demo_resampled

class SimpleSMOTE(SimplePreprocessor):
    """Simplified SMOTE-like oversampling"""
    
    def __init__(self, k_neighbors=5):
        super().__init__("SMOTE (Oversampling)")
        self.k_neighbors = k_neighbors
    
    def fit_transform(self, X, y, demographics):
        try:
            from sklearn.neighbors import NearestNeighbors
            
            X = np.array(X)
            y = np.array(y)
            
            # Find minority class
            class_counts = Counter(y)
            minority_class = min(class_counts, key=class_counts.get)
            majority_count = max(class_counts.values())
            minority_count = class_counts[minority_class]
            
            # Calculate how many synthetic samples we need
            n_synthetic = majority_count - minority_count
            
            if n_synthetic <= 0:
                # No oversampling needed
                return X.tolist(), y.tolist(), demographics
            
            # Get minority class samples
            minority_mask = y == minority_class
            minority_X = X[minority_mask]
            minority_demos = [demographics[i] for i in range(len(demographics)) if minority_mask[i]]
            
            if len(minority_X) < 2:
                # Not enough samples for SMOTE, fall back to simple duplication
                indices = np.random.choice(len(minority_X), size=n_synthetic, replace=True)
                synthetic_X = minority_X[indices]
                synthetic_y = np.full(n_synthetic, minority_class)
                synthetic_demos = [minority_demos[i] for i in indices]
            else:
                # Apply simplified SMOTE
                k = min(self.k_neighbors, len(minority_X) - 1)
                nbrs = NearestNeighbors(n_neighbors=k + 1).fit(minority_X)
                
                synthetic_X = []
                synthetic_y = []
                synthetic_demos = []
                
                for _ in range(n_synthetic):
                    # Random minority sample
                    idx = np.random.choice(len(minority_X))
                    sample = minority_X[idx]
                    
                    # Find neighbors
                    distances, indices = nbrs.kneighbors([sample])
                    neighbor_idx = np.random.choice(indices[0][1:])  # Exclude the sample itself
                    neighbor = minority_X[neighbor_idx]
                    
                    # Generate synthetic sample
                    alpha = np.random.random()
                    synthetic_sample = sample + alpha * (neighbor - sample)
                    
                    synthetic_X.append(synthetic_sample)
                    synthetic_y.append(minority_class)
                    synthetic_demos.append(minority_demos[idx])  # Copy demographics
            
            # Combine original and synthetic data
            X_combined = np.vstack([X, np.array(synthetic_X)])
            y_combined = np.hstack([y, synthetic_y])
            demo_combined = demographics + synthetic_demos
            
            return X_combined.tolist(), y_combined.tolist(), demo_combined
            
        except ImportError:
            # Fallback to simple duplication if sklearn not available
            return self._simple_duplication_fallback(X, y, demographics)
    
    def _simple_duplication_fallback(self, X, y, demographics):
        """Simple fallback if advanced libraries not available"""
        X = np.array(X)
        y = np.array(y)
        
        class_counts = Counter(y)
        minority_class = min(class_counts, key=class_counts.get)
        majority_count = max(class_counts.values())
        minority_count = class_counts[minority_class]
        
        n_duplicates = majority_count - minority_count
        
        if n_duplicates <= 0:
            return X.tolist(), y.tolist(), demographics
        
        # Simple duplication with noise
        minority_mask = y == minority_class
        minority_X = X[minority_mask]
        minority_demos = [demographics[i] for i in range(len(demographics)) if minority_mask[i]]
        
        # Add duplicates with small noise
        indices = np.random.choice(len(minority_X), size=n_duplicates, replace=True)
        duplicated_X = minority_X[indices]
        
        # Add small noise to avoid exact duplicates
        noise = np.random.normal(0, 0.01, duplicated_X.shape)
        duplicated_X = duplicated_X + noise
        
        duplicated_y = np.full(n_duplicates, minority_class)
        duplicated_demos = [minority_demos[i] for i in indices]
        
        # Combine
        X_combined = np.vstack([X, duplicated_X])
        y_combined = np.hstack([y, duplicated_y])
        demo_combined = demographics + duplicated_demos
        
        return X_combined.tolist(), y_combined.tolist(), demo_combined

class SimpleCalders(SimplePreprocessor):
    """Simplified Calders reweighting through resampling"""
    
    def __init__(self, resampling_factor=1.5):
        super().__init__("Calders (Reweighting)")
        self.resampling_factor = resampling_factor
    
    def fit_transform(self, X, y, demographics):
        X = np.array(X)
        y = np.array(y)
        
        # Calculate demographic and label proportions
        demo_keys = [self.get_demographics_key(demo) for demo in demographics]
        
        # Calculate desired proportions for fairness
        unique_demos = np.unique(demo_keys)
        unique_labels = np.unique(y)
        
        # Calculate weights for each (demographic, label) combination
        total_samples = len(X)
        weights = {}
        
        for demo in unique_demos:
            for label in unique_labels:
                # Count occurrences of this (demo, label) combination
                demo_label_count = np.sum((np.array(demo_keys) == demo) & (y == label))
                
                if demo_label_count > 0:
                    # Calculate ideal proportion (demographic proportion * label proportion)
                    demo_prop = np.sum(np.array(demo_keys) == demo) / total_samples
                    label_prop = np.sum(y == label) / total_samples
                    ideal_prop = demo_prop * label_prop
                    actual_prop = demo_label_count / total_samples
                    
                    # Weight inversely proportional to over-representation
                    weight = ideal_prop / actual_prop if actual_prop > 0 else 1.0
                    weights[(demo, label)] = weight
                else:
                    weights[(demo, label)] = 0.0
        
        # Resample based on weights
        sample_weights = []
        for i in range(len(X)):
            demo = demo_keys[i]
            label = y[i]
            sample_weights.append(weights.get((demo, label), 1.0))
        
        # Normalize weights
        sample_weights = np.array(sample_weights)
        sample_weights = sample_weights / np.sum(sample_weights)
        
        # Resample according to weights
        new_size = int(len(X) * self.resampling_factor)
        resampled_indices = np.random.choice(
            len(X), 
            size=new_size, 
            replace=True, 
            p=sample_weights
        )
        
        X_resampled = X[resampled_indices]
        y_resampled = y[resampled_indices]
        demo_resampled = [demographics[i] for i in resampled_indices]
        
        return X_resampled.tolist(), y_resampled.tolist(), demo_resampled

def create_preprocessor(method_name):
    """Factory function to create preprocessors"""
    if method_name == 'None (Baseline)':
        return NoPreprocessor()
    elif method_name == 'SMOTE (Oversampling)':
        return SimpleSMOTE()
    elif method_name == 'Rebalancing':
        return SimpleRebalancing()
    elif method_name == 'Calders (Reweighting)':
        return SimpleCalders()
    else:
        return NoPreprocessor() 
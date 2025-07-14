#!/usr/bin/env python3
"""
Dynamic preprocessing system with LAZY LOADING
==============================================

This module dynamically discovers and loads preprocessing methods from the research framework.
Methods are loaded on-demand to improve startup time.
"""

import importlib
import sys
import os
import warnings
import traceback
import numpy as np
import yaml
from typing import Dict, Tuple, Optional

# Add src to path for importing research framework methods
project_root = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Preprocessing pipeline will be imported when needed to avoid startup issues

class DynamicPreprocessingWrapper:
    """
    Wrapper class for dynamically loading preprocessing methods with LAZY LOADING
    """
    
    def __init__(self):
        # Store method configurations (not actual classes)
        self.method_configs = {}
        # Cache for loaded methods
        self.loaded_methods = {}
        # Track failed import attempts to avoid retrying
        self.failed_imports = {}
        self.loading_attempts = {}
        
        # Load method configurations (fast - no imports)
        self._load_method_configurations()
    
    def _load_method_configurations(self):
        """
        Load method configurations without importing - FAST startup
        """
        print("Loading method configurations (lazy loading enabled)...")
        
        # Comprehensive list of ALL available preprocessing methods from research framework
        self.method_configs = {
            # Basic sampling and reweighting methods
            'rebalance': {
                'module': 'mitigation.preprocessing.rebalance',
                'class': 'RebalancePreProcessor',
                'name': 'Rebalancing',
                'description': 'Rebalances demographic groups through oversampling with replacement'
            },
            'calders': {
                'module': 'mitigation.preprocessing.calders',
                'class': 'CaldersPreProcessor',
                'name': 'Calders Reweighting',
                'description': 'Reweights instances to achieve demographic independence (Calders et al. 2009)'
            },
            'smote': {
                'module': 'mitigation.preprocessing.smote',
                'class': 'SmotePreProcessor',
                'name': 'SMOTE (Chawla)',
                'description': 'Synthetic Minority Oversampling Technique for class imbalance (Chawla et al. 2002)'
            },
            
            # Kamiran family methods
            'kamiran': {
                'module': 'mitigation.preprocessing.kamiran',
                'class': 'KamiranPreProcessor',
                'name': 'Kamiran Massaging',
                'description': 'Massage data to remove disparate impact through promotion/demotion (Kamiran & Calders 2009)'
            },
            'kamiran2': {
                'module': 'mitigation.preprocessing.kamiran2',
                'class': 'Kamiran2PreProcessor',
                'name': 'Kamiran Alternative',
                'description': 'Alternative Kamiran method implementation for bias mitigation'
            },
            
            # Zelaya family methods (multiple variants)
            'zelaya_over': {
                'module': 'mitigation.preprocessing.zelaya_over',
                'class': 'ZelayaOverPreProcessor',
                'name': 'Zelaya Oversampling',
                'description': 'Zelaya oversampling approach for fairness'
            },
            'zelaya_under': {
                'module': 'mitigation.preprocessing.zelaya_under',
                'class': 'ZelayaUnderPreProcessor',
                'name': 'Zelaya Undersampling',
                'description': 'Zelaya undersampling approach for fairness'
            },
            'zelaya_smote': {
                'module': 'mitigation.preprocessing.zelaya_smote',
                'class': 'ZelayaSMOTEPreProcessor',
                'name': 'Zelaya SMOTE',
                'description': 'Zelaya SMOTE-based fairness preprocessing'
            },
            'zelaya_psp': {
                'module': 'mitigation.preprocessing.zelaya_psp',
                'class': 'ZelayaPSPPreProcessor',
                'name': 'Zelaya PSP',
                'description': 'Zelaya Preferential Sampling with Parity approach'
            },
            
            # Iosifidis family methods (4 variants)
            'iosifidis_smote_attr': {
                'module': 'mitigation.preprocessing.iosifidis_smoteattribute',
                'class': 'IosifidisSmoteAttributePreProcessor',
                'name': 'Iosifidis SMOTE Attribute',
                'description': 'SMOTE targeting sensitive attributes (Iosifidis & Ntoutsi 2019)'
            },
            'iosifidis_smote_target': {
                'module': 'mitigation.preprocessing.iosifidis_smotetarget',
                'class': 'IosifidisSmoteTargetPreProcessor',
                'name': 'Iosifidis SMOTE Target',
                'description': 'SMOTE targeting outcome variables (Iosifidis & Ntoutsi 2019)'
            },
            'iosifidis_resample_attr': {
                'module': 'mitigation.preprocessing.iosifidis_resampledattribute',
                'class': 'IosifidisResamplingAttributePreProcessor',
                'name': 'Iosifidis Resample Attribute',
                'description': 'Resampling based on sensitive attributes (Iosifidis & Ntoutsi 2019)'
            },
            'iosifidis_resample_target': {
                'module': 'mitigation.preprocessing.iosifidis_resampletarget',
                'class': 'IosifidisResamplingTargetPreProcessor',
                'name': 'Iosifidis Resample Target',
                'description': 'Resampling based on target variables (Iosifidis & Ntoutsi 2019)'
            },
            
            # Statistical and advanced methods
            'singh': {
                'module': 'mitigation.preprocessing.singh',
                'class': 'SinghSamplePreProcessor',
                'name': 'Singh Sampling',
                'description': 'Median-based multi-sensitive attribute debiasing (Singh et al. 2022)'
            },
            'alabdulmohsin': {
                'module': 'mitigation.preprocessing.alabdulmohsin',
                'class': 'AlabdulmohsinPreProcessor',
                'name': 'Alabdulmohsin Method',
                'description': 'Alabdulmohsin fairness preprocessing technique'
            },
            'salazar': {
                'module': 'mitigation.preprocessing.salazar',
                'class': 'SalazarPreProcessor', 
                'name': 'Salazar FAWOS',
                'description': 'Fair Auto-Weighted Oversampling (Salazar et al.)'
            },
            'zemel': {
                'module': 'mitigation.preprocessing.zemel',
                'class': 'ZemelPreProcessor',
                'name': 'Zemel Fair Representations',
                'description': 'Learning fair representations (Zemel et al. 2013)'
            },
            'luong': {
                'module': 'mitigation.preprocessing.luong',
                'class': 'LuongPreProcessor',
                'name': 'Luong Method',
                'description': 'Luong fairness preprocessing approach'
            },

            'chakraborty': {
                'module': 'mitigation.preprocessing.chakraborty',
                'class': 'ChakrabortyPreProcessor',
                'name': 'Chakraborty Method',
                'description': 'Chakraborty neighbor-based synthetic generation'
            },
            'cohausz': {
                'module': 'mitigation.preprocessing.cohausz',
                'class': 'CohauszPreProcessor',
                'name': 'Cohausz Method',
                'description': 'Cohausz fairness preprocessing technique'
            },
            'dablain': {
                'module': 'mitigation.preprocessing.dablain',
                'class': 'DablainPreProcessor',
                'name': 'Dablain Method',
                'description': 'Dablain fair oversampling variants'
            },
            'cock': {
                'module': 'mitigation.preprocessing.cock',
                'class': 'CockPreProcessor',
                'name': 'Cock Method',
                'description': 'Cock cluster-optimized resampling technique'
            },
            'li': {
                'module': 'mitigation.preprocessing.li',
                'class': 'LiPreProcessor',
                'name': 'Li Method',
                'description': 'Li fairness preprocessing technique'
            },
            'lahoti': {
                'module': 'mitigation.preprocessing.lahoti',
                'class': 'LahotiPreProcessor',
                'name': 'Lahoti Method',
                'description': 'Lahoti fairness preprocessing approach'
            },

        }
        
        print(f"Loaded {len(self.method_configs)} method configurations")
    
    def _load_method(self, method_key: str) -> Optional[type]:
        """
        Lazy load a specific method when it's actually needed
        """
        # Return from cache if already loaded
        if method_key in self.loaded_methods:
            return self.loaded_methods[method_key]
        
        # Don't retry methods that have already failed
        if method_key in self.failed_imports:
            return None
        
        if method_key not in self.method_configs:
            self.failed_imports[method_key] = {
                'name': f'Unknown method: {method_key}',
                'error': 'Method not configured',
                'module': 'unknown'
            }
            return None
        
        config = self.method_configs[method_key]
        
        try:
            print(f"Loading {config['name']}...")
            module = importlib.import_module(config['module'])
            cls = getattr(module, config['class'])
            
            # Cache the loaded class
            self.loaded_methods[method_key] = cls
            print(f"✓ Successfully loaded: {config['name']}")
            return cls
            
        except Exception as e:
            self.failed_imports[method_key] = {
                'name': config['name'],
                'error': str(e),
                'module': config['module']
            }
            print(f"✗ Failed to load {config['name']}: {str(e)}")
            return None
    
    def get_available_methods(self) -> Dict[str, Dict]:
        """
        Return dictionary of all available preprocessing methods (from configs)
        """
        methods = {
            'none': {
                'name': 'None (Baseline)',
                'description': 'No preprocessing - baseline comparison'
            }
        }
        
        for key, config in self.method_configs.items():
            methods[key] = {
                'name': config['name'],
                'description': config['description']
            }
        
        return methods
    
    def get_failed_imports(self) -> Dict[str, Dict]:
        """
        Return information about methods that failed to import
        """
        return self.failed_imports
    
    def get_loading_stats(self) -> Dict[str, int]:
        """
        Get statistics about loaded vs available methods
        """
        return {
            'configured': len(self.method_configs),
            'loaded': len(self.loaded_methods),
            'failed': len(self.failed_imports)
        }
    
    def create_comprehensive_settings(self, method_key: str) -> Dict:
        """
        Create comprehensive settings dictionary based on default config and research pipeline requirements
        """
        # Load default configuration
        config_path = os.path.join(src_path, 'configs', 'default_config.yaml')
        with open(config_path, 'r') as f:
            settings = yaml.load(f, Loader=yaml.FullLoader)
        
        # Load experiment configuration for preprocessor and predictor settings
        exp_config_path = os.path.join(src_path, 'configs', 'experiment_config.yaml')
        with open(exp_config_path, 'r') as f:
            exp_config = yaml.load(f, Loader=yaml.FullLoader)
        
        # Override with GUI-specific values
        settings['experiment']['name'] = 'gui_experiment'
        settings['experiment']['root_name'] = 'gui_root'
        settings['experiment']['nclasses'] = 2
        
        # Add missing top-level fields that methods expect
        settings['baseline'] = False
        settings['preprocessing'] = True
        
        # Set pipeline configurations  
        settings['pipeline']['labels'] = 'binary_label'
        settings['pipeline']['crossvalidator'] = 'nonnested'
        settings['pipeline']['gridsearch'] = 'nogs'
        settings['pipeline']['predictor'] = 'decision_tree'
        settings['pipeline']['dataset'] = 'gui_dataset'
        settings['pipeline']['nclasses'] = 2  # For alabdulmohsin
        
        # Add features configuration for methods that need it
        settings['pipeline']['features'] = ['numerical'] * 10  # Default to numerical features
        
        # Set attributes based on actual data (will be overridden later)
        settings['pipeline']['attributes'] = {
            'mitigating': 'gender',
            'discriminated': 'Female',
            'included': []  # For cohausz method
        }
        
        # Add stratification column
        settings['crossvalidation']['stratifier_col'] = 'binary_label'
        
        # Add preprocessors and predictors sections from experiment config
        if 'preprocessors' in exp_config:
            settings['preprocessors'] = exp_config['preprocessors']
        else:
            settings['preprocessors'] = {}
            
        if 'predictors' in exp_config:
            settings['predictors'] = exp_config['predictors']
        else:
            settings['predictors'] = {
                'decision_tree': {'max_depth': None}
            }
        
        # Add method-specific configurations that are missing or problematic
        if 'luong' not in settings['preprocessors']:
            settings['preprocessors']['luong'] = {}
        
        # Add feature type configurations for luong method
        settings['preprocessors']['luong']['continuous'] = list(range(10))  # Assume first 10 features are continuous
        settings['preprocessors']['luong']['categorical'] = []
        settings['preprocessors']['luong']['ordinal'] = []
        
        # Fix dablain configuration - ensure k is at least 1
        if 'dablain' not in settings['preprocessors']:
            settings['preprocessors']['dablain'] = {}
        
        # Ensure dablain has proper defaults
        dablain_config = settings['preprocessors']['dablain']
        if 'k' not in dablain_config or dablain_config['k'] <= 0:
            dablain_config['k'] = 5
        if 'proportion' not in dablain_config:
            dablain_config['proportion'] = 0.5
        
        return settings
    
    def apply_preprocessing(self, method_key: str, X_train: np.ndarray, 
                          y_train: np.ndarray, sensitive_attr: np.ndarray,
                          X_val: Optional[np.ndarray] = None, 
                          y_val: Optional[np.ndarray] = None,
                          sensitive_val: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply the specified preprocessing method
        
        Args:
            method_key: Key identifying the preprocessing method
            X_train: Training features
            y_train: Training labels
            sensitive_attr: Sensitive attribute values
            X_val: Validation features (optional)
            y_val: Validation labels (optional) 
            sensitive_val: Validation sensitive attributes (optional)
            
        Returns:
            Tuple of (preprocessed_X, preprocessed_y, preprocessed_sensitive)
        """
        if method_key == 'none':
            return X_train, y_train, sensitive_attr
        
        # Load the method if not already loaded
        preprocessor_class = self._load_method(method_key)
        
        if preprocessor_class is None:
            raise ValueError(f"Method '{method_key}' not available or failed to load. Available methods: {list(self.method_configs.keys())}")
        
        try:
            # Create comprehensive settings with all required fields
            settings = self.create_comprehensive_settings(method_key)
            
            # Validate data before applying preprocessing
            unique_labels = np.unique(y_train)
            unique_sensitive = np.unique(sensitive_attr)
            
            # Check for data issues that would cause specific methods to fail
            if method_key in ['iosifidis_smote_attr', 'iosifidis_resample_attr'] and len(unique_sensitive) < 2:
                print(f"⚠️  Skipping {method_key}: Requires multiple sensitive attribute values, got {len(unique_sensitive)}")
                return X_train, y_train, sensitive_attr
                
            if method_key.startswith('zelaya_') and (len(unique_sensitive) < 2 or len(unique_labels) < 2):
                print(f"⚠️  Skipping {method_key}: Requires balanced demographic groups, got {len(unique_sensitive)} sensitive groups, {len(unique_labels)} label classes")
                return X_train, y_train, sensitive_attr
            
            # Check if any demographic group is too small for Zelaya methods
            if method_key.startswith('zelaya_'):
                for sens_val in unique_sensitive:
                    group_size = np.sum(sensitive_attr == sens_val)
                    if group_size < 5:  # Minimum viable group size
                        print(f"⚠️  Skipping {method_key}: Demographic group '{sens_val}' too small ({group_size} samples)")
                        return X_train, y_train, sensitive_attr
            
            # Check dataset size for methods that need sufficient data
            if method_key == 'dablain' and len(X_train) < 50:
                print(f"⚠️  Skipping {method_key}: Dataset too small ({len(X_train)} samples), needs at least 50")
                return X_train, y_train, sensitive_attr
            
            # Update settings based on actual data
            if len(unique_sensitive) == 2:
                # Binary sensitive attribute
                settings['pipeline']['attributes']['mitigating'] = 'gender'
                settings['pipeline']['attributes']['discriminated'] = str(unique_sensitive[0])
            
            # Update feature count based on actual data
            n_features = X_train.shape[1]
            settings['pipeline']['features'] = ['numerical'] * n_features
            
            # Update feature indices for luong method
            if method_key == 'luong' and 'luong' in settings['preprocessors']:
                settings['preprocessors']['luong']['continuous'] = list(range(n_features))
                settings['preprocessors']['luong']['categorical'] = []
                settings['preprocessors']['luong']['ordinal'] = []
                
            # Adjust dablain parameters based on dataset size
            if method_key == 'dablain' and 'dablain' in settings['preprocessors']:
                min_class_size = min([np.sum(y_train == label) for label in unique_labels])
                # Ensure k is not larger than the smallest class
                safe_k = min(settings['preprocessors']['dablain']['k'], max(1, min_class_size - 1))
                settings['preprocessors']['dablain']['k'] = safe_k
            
            # Create preprocessor instance
            preprocessor = preprocessor_class(settings)
            
            # Convert numpy arrays to lists (format expected by framework)
            x_train_list = X_train.tolist() if isinstance(X_train, np.ndarray) else X_train
            y_train_list = y_train.tolist() if isinstance(y_train, np.ndarray) else y_train
            
            # Create demographics in expected format
            demo_train = []
            for i, sens_val in enumerate(sensitive_attr):
                demo_train.append({'gender': sens_val})
                
            # Special handling for methods with specific data requirements
            if method_key == 'zemel':
                # Zemel expects 2D sensitive attribute data
                sensitive_2d = sensitive_attr.reshape(-1, 1) if len(sensitive_attr.shape) == 1 else sensitive_attr
                demo_train = [{'gender': val[0] if isinstance(val, np.ndarray) else val} for val in sensitive_2d]
            
            # Prepare validation data for fit_transform
            if X_val is not None and y_val is not None and sensitive_val is not None:
                x_val_list = X_val.tolist() if isinstance(X_val, np.ndarray) else X_val
                y_val_list = y_val.tolist() if isinstance(y_val, np.ndarray) else y_val
                demo_val = [{'gender': val} for val in sensitive_val]
            else:
                # Use training data as validation if not provided
                x_val_list = x_train_list[:5] if len(x_train_list) > 5 else x_train_list
                y_val_list = y_train_list[:5] if len(y_train_list) > 5 else y_train_list
                demo_val = demo_train[:5] if len(demo_train) > 5 else demo_train
            
            # Apply preprocessing
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                x_preprocessed, y_preprocessed, demo_preprocessed = preprocessor.fit_transform(
                    x_train_list, y_train_list, demo_train,
                    x_val_list, y_val_list, demo_val
                )
            
            # Convert back to numpy arrays
            X_preprocessed = np.array(x_preprocessed) if isinstance(x_preprocessed, list) else x_preprocessed
            y_preprocessed = np.array(y_preprocessed) if isinstance(y_preprocessed, list) else y_preprocessed
            
            # Extract sensitive attributes from demographics
            if isinstance(demo_preprocessed, list) and len(demo_preprocessed) > 0:
                if isinstance(demo_preprocessed[0], dict):
                    sensitive_preprocessed = np.array([demo['gender'] for demo in demo_preprocessed])
                else:
                    # Fallback: repeat original sensitive attributes
                    n_samples = len(X_preprocessed)
                    sensitive_preprocessed = np.tile(sensitive_attr, (n_samples // len(sensitive_attr) + 1))[:n_samples]
            else:
                sensitive_preprocessed = sensitive_attr
            
            print(f"✓ Applied {self.method_configs[method_key]['name']}")
            print(f"  Original size: {len(X_train)} → Preprocessed size: {len(X_preprocessed)}")
            
            return X_preprocessed, y_preprocessed, sensitive_preprocessed
            
        except Exception as e:
            error_msg = str(e)
            
            # Provide specific guidance for common errors
            if "division by zero" in error_msg:
                print(f"⚠️  {method_key} failed: Insufficient data diversity for fairness correction")
                print("   Try using a larger dataset or different demographic groups")
            elif "needs to have more than 1 class" in error_msg:
                print(f"⚠️  {method_key} failed: Insufficient class diversity")
                print("   This method requires multiple classes in both labels and sensitive attributes")
            elif "n_neighbors" in error_msg and "must be an int in the range [1, inf)" in error_msg:
                print(f"⚠️  {method_key} failed: Invalid parameter configuration")
                print("   The method's neighbor parameter is misconfigured")
            elif "not enough values to unpack" in error_msg:
                print(f"⚠️  {method_key} failed: Data format incompatibility")
                print("   This method may require specific data structures not compatible with GUI")
            else:
                print(f"Error applying {method_key}: {error_msg}")
                if len(error_msg) < 100:  # Only show traceback for short errors
                    print(f"Traceback: {traceback.format_exc()}")
            
            # Return original data as fallback
            return X_train, y_train, sensitive_attr

# Global instance
_preprocessing_wrapper = None

def get_preprocessing_wrapper():
    """
    Get the global preprocessing wrapper instance
    """
    global _preprocessing_wrapper
    if _preprocessing_wrapper is None:
        _preprocessing_wrapper = DynamicPreprocessingWrapper()
    return _preprocessing_wrapper

# Convenience functions for GUI
def get_available_preprocessing_methods():
    """
    Get available preprocessing methods for the GUI dropdown
    """
    wrapper = get_preprocessing_wrapper()
    return wrapper.get_available_methods()

def apply_preprocessing_method(method_key, X_train, y_train, sensitive_attr, 
                             X_val=None, y_val=None, sensitive_val=None):
    """
    Apply preprocessing method - convenience function for GUI
    """
    wrapper = get_preprocessing_wrapper()
    return wrapper.apply_preprocessing(method_key, X_train, y_train, sensitive_attr,
                                     X_val, y_val, sensitive_val)

def get_import_status():
    """
    Get status of preprocessing method imports (lazy loading stats)
    """
    wrapper = get_preprocessing_wrapper()
    stats = wrapper.get_loading_stats()
    return {
        'configured': stats['configured'],
        'loaded': stats['loaded'], 
        'failed': stats['failed'],
        'loaded_methods': list(wrapper.loaded_methods.keys()),
        'failed_methods': wrapper.get_failed_imports()
    }

def get_loading_stats():
    """
    Get detailed loading statistics
    """
    wrapper = get_preprocessing_wrapper()
    return wrapper.get_loading_stats()

if __name__ == "__main__":
    # Test the lazy loading system
    print("Testing Dynamic Preprocessing System with LAZY LOADING")
    print("=" * 60)
    
    wrapper = get_preprocessing_wrapper()
    
    print(f"\nConfiguration loaded: {len(wrapper.method_configs)} methods")
    print("(No actual modules imported yet - FAST startup!)")
    
    print(f"\nAvailable method names:")
    for key, config in list(wrapper.method_configs.items())[:5]:  # Show first 5
        print(f"  • {key}: {config['name']}")
    print(f"  ... and {len(wrapper.method_configs) - 5} more")
    
    # Test lazy loading
    print(f"\nTesting lazy loading with 'rebalance' method...")
    stats_before = wrapper.get_loading_stats()
    print(f"Before: {stats_before['loaded']} methods loaded")
    
    # This should trigger lazy loading
    try:
        X_test = np.random.randn(10, 3)
        y_test = np.random.randint(0, 2, 10)
        sensitive_test = np.random.choice([0, 1], 10)
        
        X_proc, y_proc, sens_proc = apply_preprocessing_method(
            'rebalance', X_test, y_test, sensitive_test
        )
        
        stats_after = wrapper.get_loading_stats()
        print(f"After: {stats_after['loaded']} methods loaded")
        print(f"✓ Lazy loading test successful: {X_test.shape} → {X_proc.shape}")
        
    except Exception as e:
        print(f"✗ Lazy loading test failed: {e}")
    
    # Show final statistics
    final_stats = wrapper.get_loading_stats()
    print(f"\nFinal Statistics:")
    print(f"  Configured: {final_stats['configured']}")
    print(f"  Loaded: {final_stats['loaded']}")
    print(f"  Failed: {final_stats['failed']}")
    print(f"  Loading efficiency: {final_stats['loaded']}/{final_stats['configured']} methods loaded only when needed") 
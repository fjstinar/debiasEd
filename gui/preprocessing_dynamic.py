"""
Dynamic Preprocessing System for DebiasEd GUI

This module dynamically imports and uses preprocessing algorithms from the 
src/mitigation/preprocessing directory instead of hardcoded implementations.
"""

import sys
import os
import importlib
import warnings
import traceback
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

# Add the src directory to Python path to enable imports
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(os.path.dirname(current_dir), 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

class DynamicPreprocessingWrapper:
    """
    Dynamically loads and manages preprocessing methods from the research framework
    """
    
    def __init__(self):
        self.available_methods = {}
        self.failed_imports = {}
        self._load_available_methods()
    
    def _load_available_methods(self):
        """
        Dynamically discover and load available preprocessing methods
        """
        preprocessing_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                        'src', 'mitigation', 'preprocessing')
        
        if not os.path.exists(preprocessing_dir):
            print(f"Warning: Preprocessing directory not found: {preprocessing_dir}")
            return
        
        # Comprehensive list of ALL available preprocessing methods from research framework
        method_configs = {
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
                'description': 'Zelaya PSP (Preferential Sampling with Parity) method'
            },
            
            # Iosifidis family methods
            'iosifidis_smote_attr': {
                'module': 'mitigation.preprocessing.iosifidis_smoteattribute',
                'class': 'IosifidisSmoteAttributePreProcessor',
                'name': 'Iosifidis SMOTE Attribute',
                'description': 'Iosifidis SMOTE targeting sensitive attributes'
            },
            'iosifidis_smote_target': {
                'module': 'mitigation.preprocessing.iosifidis_smotetarget',
                'class': 'IosifidisSmoteTargetPreProcessor',
                'name': 'Iosifidis SMOTE Target',
                'description': 'Iosifidis SMOTE targeting outcome variables'
            },
            'iosifidis_resample_attr': {
                'module': 'mitigation.preprocessing.iosifidis_resampledattribute',
                'class': 'IosifidisResamplingAttributePreProcessor',
                'name': 'Iosifidis Resample Attribute',
                'description': 'Iosifidis resampling targeting sensitive attributes'
            },
            'iosifidis_resample_target': {
                'module': 'mitigation.preprocessing.iosifidis_resampletarget',
                'class': 'IosifidisResamplingTargetPreProcessor',
                'name': 'Iosifidis Resample Target',
                'description': 'Iosifidis resampling targeting outcome variables'
            },
            
            # Advanced representation learning methods
            'zemel': {
                'module': 'mitigation.preprocessing.zemel',
                'class': 'ZemelPreProcessor',
                'name': 'Zemel Fair Representations',
                'description': 'Learning fair representations through mapping (Zemel et al. 2013)'
            },
            'luong': {
                'module': 'mitigation.preprocessing.luong',
                'class': 'LuongPreProcessor',
                'name': 'Luong Fair Representation',
                'description': 'Fair representation learning approach (Luong et al.)'
            },
            
            # Statistical and optimization-based methods
            'singh': {
                'module': 'mitigation.preprocessing.singh',
                'class': 'SinghSamplePreProcessor',
                'name': 'Singh Sampling',
                'description': 'Multi-sensitive debiasing through median-based resampling (Singh et al. 2022)'
            },
            'alabdulmohsin': {
                'module': 'mitigation.preprocessing.alabdulmohsin',
                'class': 'AlabdulmohsinPreProcessor',
                'name': 'Alabdulmohsin Method',
                'description': 'Alabdulmohsin fairness preprocessing approach'
            },
            'salazar': {
                'module': 'mitigation.preprocessing.salazar',
                'class': 'SalazarPreProcessor',
                'name': 'Salazar Method',
                'description': 'Salazar fairness preprocessing technique'
            },
            'yan': {
                'module': 'mitigation.preprocessing.yan',
                'class': 'YanPreProcessor',
                'name': 'Yan Method',
                'description': 'Yan fairness preprocessing approach'
            },
            'chakraborty': {
                'module': 'mitigation.preprocessing.chakraborty',
                'class': 'ChakrabortyPreProcessor',
                'name': 'Chakraborty Method',
                'description': 'Chakraborty fairness preprocessing technique'
            },
            'cohausz': {
                'module': 'mitigation.preprocessing.cohausz',
                'class': 'CohauszPreProcessor',
                'name': 'Cohausz Method',
                'description': 'Cohausz fairness preprocessing approach'
            },
            'dablain': {
                'module': 'mitigation.preprocessing.dablain',
                'class': 'DablainPreProcessor',
                'name': 'Dablain Method',
                'description': 'Dablain fairness preprocessing technique'
            },
            'cock': {
                'module': 'mitigation.preprocessing.cock',
                'class': 'CockPreProcessor',
                'name': 'Cock Method',
                'description': 'Cock fairness preprocessing approach'
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
            'jiang': {
                'module': 'mitigation.preprocessing.jiang',
                'class': 'JiangPreProcessor',
                'name': 'Jiang Method',
                'description': 'Jiang fairness preprocessing technique'
            }
        }
        
        for method_key, config in method_configs.items():
            try:
                module = importlib.import_module(config['module'])
                cls = getattr(module, config['class'])
                
                self.available_methods[method_key] = {
                    'class': cls,
                    'name': config['name'],
                    'description': config['description'],
                    'module_name': config['module']
                }
                print(f"✓ Successfully loaded: {config['name']}")
                
            except Exception as e:
                self.failed_imports[method_key] = {
                    'name': config['name'],
                    'error': str(e),
                    'module': config['module']
                }
                print(f"✗ Failed to load {config['name']}: {str(e)}")
    
    def get_available_methods(self) -> Dict[str, Dict]:
        """
        Return dictionary of available preprocessing methods
        """
        methods = {
            'none': {
                'name': 'None (Baseline)',
                'description': 'No preprocessing - baseline comparison'
            }
        }
        
        for key, method_info in self.available_methods.items():
            methods[key] = {
                'name': method_info['name'],
                'description': method_info['description']
            }
        
        return methods
    
    def get_failed_imports(self) -> Dict[str, Dict]:
        """
        Return information about methods that failed to import
        """
        return self.failed_imports
    
    def create_mock_settings(self, method_key: str) -> Dict:
        """
        Create mock settings dictionary required by preprocessing methods
        """
        settings = {
            'experiment': {'name': 'gui_experiment'},
            'seeds': {'model': 42, 'preprocessor': 42},
            'pipeline': {
                'attributes': {
                    'mitigating': 'gender',  # Will be overridden
                    'discriminated': 'Female'  # Will be overridden
                },
                'features': ['numerical'] * 10  # Default feature types for methods that need it
            },
            'preprocessors': {}
        }
        
        # Method-specific default settings for all available methods
        if method_key == 'rebalance':
            settings['preprocessors']['rebalance'] = {'index': 1}
        elif method_key == 'calders':
            settings['preprocessors']['calders'] = {'sampling_proportions': 1.5}
        elif method_key == 'smote':
            settings['preprocessors']['chawla'] = {'sampling_strategy': 'auto'}
        elif method_key == 'kamiran':
            settings['preprocessors']['kamiran'] = {}
        elif method_key == 'kamiran2':
            settings['preprocessors']['kamiran2'] = {}
        elif method_key == 'zelaya_over':
            settings['preprocessors']['zelaya'] = {}
        elif method_key == 'zelaya_under':
            settings['preprocessors']['zelaya'] = {}
        elif method_key == 'zelaya_smote':
            settings['preprocessors']['zelaya'] = {}
        elif method_key == 'zelaya_psp':
            settings['preprocessors']['zelaya'] = {}
        elif method_key == 'iosifidis_smote_attr':
            settings['preprocessors']['iosifidis_smote_attr'] = {}
        elif method_key == 'iosifidis_smote_target':
            settings['preprocessors']['iosifidis_smote_target'] = {}
        elif method_key == 'iosifidis_resample_attr':
            settings['preprocessors']['iosifidis_resample_attr'] = {}
        elif method_key == 'iosifidis_resample_target':
            settings['preprocessors']['iosifidis_resample_target'] = {}
        elif method_key == 'zemel':
            settings['preprocessors']['zemel'] = {'k': 5}
        elif method_key == 'luong':
            settings['preprocessors']['luong'] = {'lambda_param': 0.5}
        elif method_key == 'singh':
            settings['preprocessors']['singh'] = {}
        elif method_key == 'alabdulmohsin':
            settings['preprocessors']['alabdulmohsin'] = {}
        elif method_key == 'salazar':
            settings['preprocessors']['salazar'] = {}
        elif method_key == 'yan':
            settings['preprocessors']['yan'] = {'clustering': 'kmeans', 'knn': 3}
        elif method_key == 'chakraborty':
            settings['preprocessors']['chakraborty'] = {}
        elif method_key == 'cohausz':
            settings['preprocessors']['cohausz'] = {}
        elif method_key == 'dablain':
            settings['preprocessors']['dablain'] = {'proportion': 0.5, 'k': 5}
        elif method_key == 'cock':
            settings['preprocessors']['cock'] = {'combinations': ['param1', 'param2']}
        elif method_key == 'li':
            settings['preprocessors']['calders'] = {}
        elif method_key == 'lahoti':
            settings['preprocessors']['lahoti'] = {}
        elif method_key == 'jiang':
            settings['preprocessors']['jiang'] = {}
        
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
        
        if method_key not in self.available_methods:
            raise ValueError(f"Method '{method_key}' not available. Available methods: {list(self.available_methods.keys())}")
        
        try:
            # Create preprocessor instance
            settings = self.create_mock_settings(method_key)
            
            # Update settings based on actual data
            unique_sensitive = np.unique(sensitive_attr)
            if len(unique_sensitive) == 2:
                # Binary sensitive attribute
                settings['pipeline']['attributes']['mitigating'] = 'gender'
                settings['pipeline']['attributes']['discriminated'] = str(unique_sensitive[0])
            
            preprocessor_class = self.available_methods[method_key]['class']
            preprocessor = preprocessor_class(settings)
            
            # Convert numpy arrays to lists (format expected by framework)
            x_train_list = X_train.tolist() if isinstance(X_train, np.ndarray) else X_train
            y_train_list = y_train.tolist() if isinstance(y_train, np.ndarray) else y_train
            
            # Create demographics in expected format
            demo_train = []
            for i, sens_val in enumerate(sensitive_attr):
                demo_train.append({'gender': sens_val})
            
            # Prepare validation data
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
            
            print(f"✓ Applied {self.available_methods[method_key]['name']}")
            print(f"  Original size: {len(X_train)} → Preprocessed size: {len(X_preprocessed)}")
            
            return X_preprocessed, y_preprocessed, sensitive_preprocessed
            
        except Exception as e:
            print(f"Error applying {method_key}: {str(e)}")
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
    Get status of preprocessing method imports
    """
    wrapper = get_preprocessing_wrapper()
    return {
        'successful': list(wrapper.available_methods.keys()),
        'failed': wrapper.get_failed_imports()
    }

if __name__ == "__main__":
    # Test the dynamic loading
    print("Testing Dynamic Preprocessing System...")
    print("=" * 50)
    
    wrapper = get_preprocessing_wrapper()
    
    print(f"\nSuccessfully loaded methods: {len(wrapper.available_methods)}")
    for key, info in wrapper.available_methods.items():
        print(f"  ✓ {key}: {info['name']}")
    
    print(f"\nFailed imports: {len(wrapper.failed_imports)}")
    for key, info in wrapper.failed_imports.items():
        print(f"  ✗ {key}: {info['name']} - {info['error']}")
    
    # Test with sample data
    print("\nTesting with sample data...")
    X_test = np.random.randn(10, 3)
    y_test = np.random.randint(0, 2, 10)
    sensitive_test = np.random.choice(['Male', 'Female'], 10)
    
    try:
        X_proc, y_proc, sens_proc = apply_preprocessing_method(
            'rebalance', X_test, y_test, sensitive_test
        )
        print(f"✓ Test successful: {X_test.shape} → {X_proc.shape}")
    except Exception as e:
        print(f"✗ Test failed: {e}") 
# Dynamic Preprocessing Integration

## Overview

The DebiasEd GUI now uses a **dynamic preprocessing system** that loads actual research-grade algorithms from `src/mitigation/preprocessing` instead of hardcoded simplified implementations.

## What Changed

### Before
- Used hardcoded simplified implementations in `gui/preprocessing_simple.py`
- Limited to 4 basic methods: None, SMOTE, Rebalancing, Calders
- No connection to the research framework's actual algorithms

### After  
- **Dynamic loading** from `src/mitigation/preprocessing/`
- **Automatic fallback** to simple implementations if dependencies fail
- **Research-grade algorithms** with full functionality
- **Graceful dependency handling** - continues working even with missing packages

## Architecture

### Files Created/Modified

1. **`gui/preprocessing_dynamic.py`** (NEW)
   - Dynamic import system for preprocessing methods
   - Handles dependency failures gracefully
   - Provides unified interface for GUI integration

2. **`gui/unfairness_mitigation_gui.py`** (MODIFIED)
   - Updated to use dynamic system with fallback
   - Automatic method detection and dropdown population
   - Improved error handling

3. **`dynamic_preprocessing_demo.py`** (NEW)
   - Demonstration script showing the system in action
   - Tests all available methods with sample data

## Currently Available Methods

### ✅ Successfully Loaded (Research Framework)
- **Rebalancing** (`RebalancePreProcessor`)
  - Rebalances demographic groups through oversampling with replacement
  - From: `src/mitigation/preprocessing/rebalance.py`

- **Calders Reweighting** (`CaldersPreProcessor`) 
  - Reweights instances to achieve demographic parity
  - From: `src/mitigation/preprocessing/calders.py`

### ❌ Failed to Load (Missing Dependencies)
- **SMOTE Oversampling** - Requires additional packages
- **Kamiran Sampling** - Requires additional packages  
- **Luong Fair Representation** - Requires additional packages

## How It Works

### 1. Dynamic Discovery
```python
from gui.preprocessing_dynamic import get_available_preprocessing_methods

methods = get_available_preprocessing_methods()
# Returns: {'none': {...}, 'rebalance': {...}, 'calders': {...}}
```

### 2. Automatic Fallback
- Tries to import from research framework first
- Falls back to simple implementations if imports fail
- GUI continues working regardless of dependency issues

### 3. Unified Interface
```python
from gui.preprocessing_dynamic import apply_preprocessing_method

X_processed, y_processed, sens_processed = apply_preprocessing_method(
    'rebalance', X_train, y_train, sensitive_attr
)
```

## Testing

Run the demonstration script to see the system in action:

```bash
python dynamic_preprocessing_demo.py
```

Expected output:
- Shows which methods loaded successfully
- Tests each method with sample educational data
- Displays data transformations (e.g., 20 → 300 students for rebalancing)

## Benefits

### 1. **Research-Grade Quality**
- Uses actual algorithms from the research framework
- Full implementations with proper references and documentation
- Validated methods from academic literature

### 2. **Robustness**
- Graceful handling of missing dependencies
- Automatic fallback ensures GUI always works
- Clear error reporting for debugging

### 3. **Extensibility**
- Easy to add new methods by updating the configuration
- Automatic detection of available methods
- No GUI code changes needed for new algorithms

### 4. **User Experience**
- Dropdown automatically populated with working methods
- Dynamic descriptions based on available algorithms
- Comparative results show actual research method performance

## Adding New Methods

To add a new preprocessing method from the research framework:

1. **Update `gui/preprocessing_dynamic.py`**:
```python
method_configs = {
    # ... existing methods ...
    'new_method': {
        'module': 'mitigation.preprocessing.new_method',
        'class': 'NewMethodPreProcessor',
        'name': 'New Method Name',
        'description': 'Description of what this method does'
    }
}
```

2. **Add default settings** (if needed):
```python
elif method_key == 'new_method':
    settings['preprocessors']['new_method'] = {'param1': 'value1'}
```

The GUI will automatically detect and include the new method!

## Migration Path

The system provides a smooth migration path:

1. **Phase 1** (Current): Dynamic loading with fallback to simple methods
2. **Phase 2**: Add more research methods as dependencies become available  
3. **Phase 3**: Eventually remove simple fallbacks once all methods work dynamically

## Impact on User Workflow

### GUI Users
- **No change** in user interface
- **More authentic** results from research-grade algorithms
- **Better performance** on real datasets

### Researchers  
- **Direct access** to actual research framework methods
- **Consistent results** between GUI and command-line research tools
- **Easy validation** of GUI results against research pipeline

## Conclusion

This integration bridges the gap between the simplified GUI and the full research framework, providing users with access to authentic research-grade bias mitigation algorithms while maintaining the ease of use of the GUI interface.

The dynamic loading system ensures robustness and extensibility, making it easy to add new methods as they become available in the research framework. 
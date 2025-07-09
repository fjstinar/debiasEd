import tkinter as tk
import pickle
import os
import sys
import numpy as np
from tkinter import messagebox
from tkinter import ttk  # For the Treeview widget and Combobox
import numpy
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier

# Add the src directory to the path so we can import the DTClassifier
project_root = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "src"))

# Now import the classes
from src.predictors.decision_tree import DTClassifier

# Import preprocessing techniques - try dynamic first, then fallback to simple
try:
    from gui.preprocessing_dynamic import get_available_preprocessing_methods, apply_preprocessing_method
    USING_DYNAMIC_PREPROCESSING = True
    print("✓ Using dynamic preprocessing system")
except ImportError as e:
    print(f"Dynamic preprocessing not available: {e}")
    from gui.preprocessing_simple import create_preprocessor
    USING_DYNAMIC_PREPROCESSING = False
    print("✓ Using simple preprocessing system")

class SimpleDTClassifier:
    """Simple decision tree classifier for the GUI"""
    def __init__(self, settings=None):
        max_depth = 5
        if settings and 'predictors' in settings and 'decision_tree' in settings['predictors']:
            max_depth = settings['predictors']['decision_tree'].get('max_depth', 5)
        
        self.model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    
    def fit(self, x_train, y_train, x_val=None, y_val=None):
        self.model.fit(x_train, y_train)
    
    def predict(self, x, y=None):
        predictions = self.model.predict(x)
        return predictions, y

class PreprocessingWrapper:
    """Wrapper to handle preprocessing techniques in the GUI"""
    
    def __init__(self):
        if USING_DYNAMIC_PREPROCESSING:
            # Get available methods from dynamic system
            methods_dict = get_available_preprocessing_methods()
            self.available_methods = list(methods_dict.keys())
            self.method_info = methods_dict
        else:
            # Fallback to simple system
            self.available_methods = [
                'none', 'smote', 'rebalance', 'calders'
            ]
            self.method_info = {
                'none': {'name': 'None (Baseline)', 'description': 'No preprocessing - train on original data'},
                'smote': {'name': 'SMOTE (Oversampling)', 'description': 'Synthetic Minority Oversampling Technique'},
                'rebalance': {'name': 'Rebalancing', 'description': 'Rebalances demographic groups through sampling'},
                'calders': {'name': 'Calders (Reweighting)', 'description': 'Reweights instances to achieve demographic independence'}
            }
        
    def get_method_names(self):
        return [self.method_info[key]['name'] for key in self.available_methods]
    
    def get_method_descriptions(self):
        return {self.method_info[key]['name']: self.method_info[key]['description'] 
                for key in self.available_methods}
    
    def get_method_key_from_name(self, method_name):
        """Convert display name back to method key"""
        for key in self.available_methods:
            if self.method_info[key]['name'] == method_name:
                return key
        return 'none'
    
    def apply_preprocessing(self, method_name, X_train, y_train, sensitive_attr):
        """Apply preprocessing method"""
        method_key = self.get_method_key_from_name(method_name)
        
        if USING_DYNAMIC_PREPROCESSING:
            return apply_preprocessing_method(method_key, X_train, y_train, sensitive_attr)
        else:
            # Fallback to simple system
            preprocessor = create_preprocessor(method_name)
            return preprocessor.apply_preprocessing(X_train, y_train, sensitive_attr)

class DataLoaderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Data Dictionary Loader with Fairness Preprocessing")
        self.root.geometry("700x500")
        self.current_data = None
        self.dataset_name = None
        self.preprocessing_wrapper = PreprocessingWrapper()
        
        # Create main frames
        self.header_frame = tk.Frame(root, pady=10)
        self.header_frame.pack(fill=tk.X)
        
        self.content_frame = tk.Frame(root)
        self.content_frame.pack(expand=True, fill=tk.BOTH, padx=20, pady=10)
        
        self.status_frame = tk.Frame(root, pady=10)
        self.status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        # Header
        tk.Label(
            self.header_frame, 
            text="Unfairness Mitigation - Dataset Loader & Preprocessing", 
            font=("Arial", 16, "bold")
        ).pack()
        
        # Dataset buttons section
        datasets_frame = tk.LabelFrame(self.content_frame, text="Available Datasets")
        datasets_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Get list of dataset folders
        notebook_dir = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
        print(notebook_dir)
        dataset_folders = self._get_dataset_folders(notebook_dir)
        print(dataset_folders)
        
        # Create a button for each dataset
        for dataset in dataset_folders:
            btn = tk.Button(
                datasets_frame,
                text=f"Load {dataset}",
                command=lambda ds=dataset: self.load_data_dictionary(ds),
                width=20,
                height=1,
                relief=tk.RAISED,
                bg="#e0e0e0"
            )
            btn.pack(pady=2, padx=10, anchor=tk.W)
        
        # Add preprocessing selection section
        preprocessing_frame = tk.LabelFrame(self.content_frame, text="Bias Mitigation Preprocessing")
        preprocessing_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Preprocessing method selection
        tk.Label(preprocessing_frame, text="Select Preprocessing Method:").pack(anchor=tk.W, padx=5, pady=2)
        
        self.preprocessing_var = tk.StringVar()
        # Set default to first available method name
        method_names = self.preprocessing_wrapper.get_method_names()
        default_method = method_names[0] if method_names else "No methods available"
        self.preprocessing_var.set(default_method)
        
        self.preprocessing_combo = ttk.Combobox(
            preprocessing_frame,
            textvariable=self.preprocessing_var,
            values=self.preprocessing_wrapper.get_method_names(),
            state="readonly",
            width=30
        )
        self.preprocessing_combo.pack(anchor=tk.W, padx=5, pady=2)
        
        # Get method descriptions from wrapper
        method_descriptions = self.preprocessing_wrapper.get_method_descriptions()
        
        # Set default description
        default_method = list(method_descriptions.keys())[0] if method_descriptions else "No methods available"
        default_description = method_descriptions.get(default_method, "No description available")
        
        self.description_label = tk.Label(
            preprocessing_frame, 
            text=default_description,
            wraplength=400,
            justify=tk.LEFT,
            fg="gray"
        )
        self.description_label.pack(anchor=tk.W, padx=5, pady=2)
        
        # Update description when selection changes
        def update_description(*args):
            method = self.preprocessing_var.get()
            self.description_label.config(text=method_descriptions.get(method, 'Unknown method'))
        
        self.preprocessing_var.trace('w', update_description)
        
        # Training button section - initially disabled
        training_frame = tk.Frame(self.content_frame)
        training_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.train_btn = tk.Button(
            training_frame,
            text="Train Decision Tree with Preprocessing",
            command=self.train_with_preprocessing,
            width=30,
            height=2,
            state=tk.DISABLED,
            bg="#4CAF50",
            fg="white",
            font=("Arial", 10, "bold")
        )
        self.train_btn.pack(pady=5)
        
        # Status indicator
        self.status_var = tk.StringVar()
        self.status_var.set("No dataset loaded")
        self.status_label = tk.Label(
            self.status_frame, 
            textvariable=self.status_var,
            bd=1,
            relief=tk.SUNKEN,
            anchor=tk.W,
            padx=10
        )
        self.status_label.pack(fill=tk.X)
    
    def _get_dataset_folders(self, notebooks_dir):
        """Get dataset folders that contain data_dictionary.pkl files"""
        datasets = []
        try:
            for folder in os.listdir(notebooks_dir):
                potential_data_path = os.path.join(notebooks_dir, folder, "data_dictionary.pkl")
                if os.path.isfile(potential_data_path):
                    datasets.append(folder)
        except Exception as e:
            print(f"Error scanning dataset folders: {e}")
        return datasets
    
    def load_data_dictionary(self, dataset_name):
        """Load the data dictionary for the specified dataset"""
        try:
            notebook_dir = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
            data_dict_path = os.path.join(notebook_dir, dataset_name, "data_dictionary.pkl")
            
            with open(data_dict_path, 'rb') as f:
                self.current_data = pickle.load(f)
            
            self.dataset_name = dataset_name
            self.status_var.set(f"Loaded dataset: {dataset_name}")
            messagebox.showinfo("Success", f"Successfully loaded {dataset_name} data dictionary")
            
            # Enable the training button
            self.train_btn.config(state=tk.NORMAL)
            
            # Display the data features table
            self.show_data_features(dataset_name)
            
            # Here you can add additional code to display or use the loaded data
            print(f"Loaded {dataset_name} data dictionary")
        except Exception as e:
            self.status_var.set(f"Error loading dataset: {dataset_name}")
            messagebox.showerror("Error", f"Failed to load {dataset_name}: {str(e)}")
    
    def show_data_features(self, dataset_name):
        """Show a table with all features in the data dictionary"""
        if not self.current_data:
            return
            
        # Create a new toplevel window
        features_window = tk.Toplevel(self.root)
        features_window.title(f"{dataset_name} - Data Features")
        features_window.geometry("800x600")
        
        # Add a frame for the table
        frame = tk.Frame(features_window)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create a scrollbar
        scrollbar_y = tk.Scrollbar(frame)
        scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
        
        scrollbar_x = tk.Scrollbar(frame, orient=tk.HORIZONTAL)
        scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Create the treeview (table)
        tree = ttk.Treeview(frame, yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)
        
        # Configure scrollbars
        scrollbar_y.config(command=tree.yview)
        scrollbar_x.config(command=tree.xview)
        
        # Define columns - removed 'value' column
        tree['columns'] = ('type', 'additional_info')
        
        # Format columns
        tree.column('#0', width=200, minwidth=150)  # Made wider since we removed a column
        tree.column('type', width=150, minwidth=100)
        tree.column('additional_info', width=450, minwidth=200)  # Made wider for more info
        
        # Create headings
        tree.heading('#0', text='Feature Name')
        tree.heading('type', text='Data Type')
        tree.heading('additional_info', text='Additional Info')
        
        try:
            # Check for 'data' key which might contain individual records
            if 'data' in self.current_data:
                # Get a sample record
                sample_record = None
                for idx, record in self.current_data['data'].items():
                    if isinstance(record, dict) and 'features' in record:
                        sample_record = record
                        break
                
                if sample_record:
                    # Process directly accessible features (like sex, age, etc.)
                    direct_features = [k for k in sample_record.keys() if k != 'features' and k != 'learner_id' and k != 'binary_label']
                    
                    for feature in direct_features:
                        value_type = type(sample_record[feature]).__name__
                        tree.insert('', tk.END, text=feature, 
                                    values=(value_type, "Direct feature"))
                    
                    # If there's a feature dictionary or mapping available elsewhere
                    feature_info = {}
                    if 'feature_info' in self.current_data:
                        feature_info = self.current_data['feature_info']
                    
                    # Process the features array
                    if isinstance(sample_record['features'], (list, tuple, numpy.ndarray)):
                        features_array = sample_record['features']
                        for i, value in enumerate(features_array):
                            feature_name = f"feature_{i}"
                            
                            # Try to get actual feature name if available
                            if feature_info and i in feature_info:
                                feature_name = feature_info[i].get('name', feature_name)
                                additional_info = feature_info[i].get('description', '')
                            else:
                                additional_info = f"Index {i} in features array"
                                
                            tree.insert('', tk.END, text=feature_name, 
                                        values=(type(value).__name__, additional_info))
                    
                    # Add binary label info
                    if 'binary_label' in sample_record:
                        tree.insert('', tk.END, text='binary_label', 
                                    values=(type(sample_record['binary_label']).__name__, "Target variable"))
            else:
                # Fallback for different structure - display all top-level keys
                for key, value in self.current_data.items():
                    if isinstance(value, dict):
                        child_keys = list(value.keys())[:3]
                        additional_info = f"Dictionary with {len(value)} items. Sample keys: {', '.join(map(str, child_keys))}..."
                    elif isinstance(value, list):
                        additional_info = f"List with {len(value)} items"
                    else:
                        additional_info = ""
                    
                    tree.insert('', tk.END, text=key, 
                                values=(type(value).__name__, additional_info))
        
        except Exception as e:
            # Display error and show raw structure
            error_label = tk.Label(frame, text=f"Error analyzing data structure: {str(e)}", fg="red")
            error_label.pack(before=tree)
            
            # Display raw structure of the data dictionary
            for key in self.current_data.keys():
                try:
                    value_type = type(self.current_data[key]).__name__
                    if isinstance(self.current_data[key], dict):
                        additional_info = f"Dict with keys: {', '.join(list(self.current_data[key].keys())[:3])}..."
                    elif isinstance(self.current_data[key], list):
                        additional_info = f"List with {len(self.current_data[key])} items"
                    else:
                        additional_info = ""
                except:
                    value_type = "Error"
                    additional_info = "Could not access"
                    
                tree.insert('', tk.END, text=key, values=(value_type, additional_info))
            
            print(f"Error in show_data_features: {e}")
        
        # Pack the tree
        tree.pack(fill=tk.BOTH, expand=True)
        
        # Add a close button
        close_btn = tk.Button(features_window, text="Close", command=features_window.destroy)
        close_btn.pack(pady=10)
    
    def train_with_preprocessing(self):
        """Train a decision tree classifier with preprocessing on the loaded data and show results"""
        if not self.current_data or 'data' not in self.current_data:
            messagebox.showerror("Error", "No valid data loaded for training")
            return
            
        try:
            # Extract features, labels, and demographics
            X = []
            y = []
            demographics = []
            
            for idx, record in self.current_data['data'].items():
                if 'features' in record and 'binary_label' in record:
                    X.append(record['features'])
                    y.append(record['binary_label'])
                    
                    # Extract demographics - build a dict with available demographics
                    demo_dict = {}
                    for demo_attr in self.current_data.get('available_demographics', []):
                        if demo_attr in record:
                            demo_dict[demo_attr] = record[demo_attr]
                    demographics.append(demo_dict)
            
            if not X or not y:
                messagebox.showerror("Error", "Could not extract features and labels from the data")
                return
                
            # Convert to numpy arrays
            X = np.array(X)
            y = np.array(y)
            
            # Split the data
            X_train, X_test, y_train, y_test, demo_train, demo_test = train_test_split(
                X, y, demographics, test_size=0.2, random_state=42
            )
            
            # Create settings dictionary
            settings = {
                'predictors': {
                    'decision_tree': {
                        'max_depth': 5
                    }
                },
                'experiment': {'name': 'gui_experiment'},
                'seeds': {'preprocessor': 42, 'model': 42},
                'pipeline': {
                    'attributes': {
                        'mitigating': 'gender',  # Default - could be made configurable
                        'discriminated': '_1'     # Default - could be made configurable
                    }
                }
            }
            
            # Store original data for comparison 
            X_train_original = X_train.copy()
            y_train_original = y_train.copy()
            demo_train_original = demo_train.copy()
            
            # Apply preprocessing if selected
            preprocessing_method = self.preprocessing_var.get()
            
            try:
                # Extract sensitive attribute from demographics
                sensitive_attr = np.array([demo['gender'] for demo in demo_train_original])
                
                # Apply preprocessing using the wrapper
                X_train, y_train, sensitive_processed = self.preprocessing_wrapper.apply_preprocessing(
                    preprocessing_method, X_train, y_train, sensitive_attr
                )
                
                # Update demographics with processed sensitive attributes
                demo_train = [{'gender': val} for val in sensitive_processed]
                
                preprocessing_info = f"Applied {preprocessing_method}"
                
            except Exception as e:
                messagebox.showwarning("Preprocessing Warning", 
                                     f"Preprocessing failed: {str(e)}\nUsing original data instead.")
                X_train = X_train_original
                y_train = y_train_original
                demo_train = demo_train_original
                preprocessing_info = f"Preprocessing failed, using baseline"
            
            # Train baseline model for comparison
            baseline_model = SimpleDTClassifier(settings)
            baseline_model.fit(X_train_original, y_train_original)
            baseline_predictions, _ = baseline_model.predict(X_test)
            
            baseline_accuracy = accuracy_score(y_test, baseline_predictions)
            baseline_precision = precision_score(y_test, baseline_predictions, zero_division=0)
            baseline_recall = recall_score(y_test, baseline_predictions, zero_division=0)
            baseline_f1 = f1_score(y_test, baseline_predictions, zero_division=0)
            baseline_conf_matrix = confusion_matrix(y_test, baseline_predictions)
            
            # Train model with preprocessing
            model = SimpleDTClassifier(settings)
            model.fit(X_train, y_train)
            
            # Get predictions
            predictions, _ = model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, predictions)
            precision = precision_score(y_test, predictions, zero_division=0)
            recall = recall_score(y_test, predictions, zero_division=0)
            f1 = f1_score(y_test, predictions, zero_division=0)
            conf_matrix = confusion_matrix(y_test, predictions)
            
            # Show comparative results
            self.show_comparative_results(
                # Baseline results
                baseline_accuracy, baseline_precision, baseline_recall, baseline_f1, baseline_conf_matrix,
                # Preprocessed results
                accuracy, precision, recall, f1, conf_matrix,
                preprocessing_info,
                len(X_train_original), len(X_train)  # Original vs preprocessed data sizes
            )
            
        except Exception as e:
            messagebox.showerror("Training Error", f"An error occurred during model training: {str(e)}")
            print(f"Training error: {e}")
    
    def show_comparative_results(self, 
                                baseline_acc, baseline_prec, baseline_rec, baseline_f1, baseline_cm,
                                processed_acc, processed_prec, processed_rec, processed_f1, processed_cm,
                                preprocessing_info, original_size, processed_size):
        """Display comparative results between baseline and preprocessed models"""
        results_window = tk.Toplevel(self.root)
        results_window.title(f"Fairness Results - {self.dataset_name}")
        results_window.geometry("800x600")
        
        # Title
        tk.Label(
            results_window, 
            text="Bias Mitigation Results: Baseline vs Preprocessed",
            font=("Arial", 16, "bold")
        ).pack(pady=10)
        
        # Preprocessing info
        tk.Label(
            results_window,
            text=f"Preprocessing Method: {preprocessing_info}",
            font=("Arial", 12),
            fg="blue"
        ).pack(pady=5)
        
        # Data size info
        tk.Label(
            results_window,
            text=f"Training Data: {original_size} → {processed_size} samples",
            font=("Arial", 10),
            fg="gray"
        ).pack(pady=2)
        
        # Create two columns for comparison
        comparison_frame = tk.Frame(results_window)
        comparison_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Baseline results (left column)
        baseline_frame = tk.LabelFrame(comparison_frame, text="Baseline (No Preprocessing)")
        baseline_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        baseline_metrics = [
            ("Accuracy:", f"{baseline_acc:.4f}"),
            ("Precision:", f"{baseline_prec:.4f}"),
            ("Recall:", f"{baseline_rec:.4f}"),
            ("F1 Score:", f"{baseline_f1:.4f}")
        ]
        
        for i, (label, value) in enumerate(baseline_metrics):
            tk.Label(baseline_frame, text=label, anchor=tk.W, width=12).grid(row=i, column=0, sticky=tk.W, padx=5, pady=3)
            tk.Label(baseline_frame, text=value, anchor=tk.W).grid(row=i, column=1, sticky=tk.W, padx=5, pady=3)
        
        # Baseline confusion matrix
        baseline_cm_frame = tk.Frame(baseline_frame)
        baseline_cm_frame.grid(row=len(baseline_metrics), column=0, columnspan=2, pady=10)
        
        tk.Label(baseline_cm_frame, text="Confusion Matrix", font=("Arial", 10, "bold")).pack()
        self._draw_confusion_matrix(baseline_cm_frame, baseline_cm, size=40)
        
        # Preprocessed results (right column)
        processed_frame = tk.LabelFrame(comparison_frame, text="With Preprocessing")
        processed_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        
        processed_metrics = [
            ("Accuracy:", f"{processed_acc:.4f}"),
            ("Precision:", f"{processed_prec:.4f}"),
            ("Recall:", f"{processed_rec:.4f}"),
            ("F1 Score:", f"{processed_f1:.4f}")
        ]
        
        for i, (label, value) in enumerate(processed_metrics):
            # Color-code improvements
            improvement = ""
            color = "black"
            if i < len(baseline_metrics):
                baseline_val = float(baseline_metrics[i][1])
                processed_val = float(value)
                if processed_val > baseline_val:
                    improvement = " ↑"
                    color = "green"
                elif processed_val < baseline_val:
                    improvement = " ↓"
                    color = "red"
                    
            tk.Label(processed_frame, text=label, anchor=tk.W, width=12).grid(row=i, column=0, sticky=tk.W, padx=5, pady=3)
            value_label = tk.Label(processed_frame, text=value + improvement, anchor=tk.W, fg=color)
            value_label.grid(row=i, column=1, sticky=tk.W, padx=5, pady=3)
        
        # Preprocessed confusion matrix
        processed_cm_frame = tk.Frame(processed_frame)
        processed_cm_frame.grid(row=len(processed_metrics), column=0, columnspan=2, pady=10)
        
        tk.Label(processed_cm_frame, text="Confusion Matrix", font=("Arial", 10, "bold")).pack()
        self._draw_confusion_matrix(processed_cm_frame, processed_cm, size=40)
        
        # Summary and fairness analysis
        summary_frame = tk.LabelFrame(results_window, text="Fairness Analysis Summary")
        summary_frame.pack(fill=tk.X, padx=20, pady=10)
        
        # Calculate improvements
        acc_improvement = processed_acc - baseline_acc
        f1_improvement = processed_f1 - baseline_f1
        
        summary_text = f"Accuracy Change: {acc_improvement:+.4f} | F1 Change: {f1_improvement:+.4f}"
        
        if processed_size != original_size:
            summary_text += f" | Data Size Changed: {original_size} → {processed_size}"
        
        tk.Label(summary_frame, text=summary_text, font=("Arial", 10)).pack(pady=5)
        
        # Interpretation
        if acc_improvement > 0.01:
            interpretation = "✅ Preprocessing improved model performance"
            color = "green"
        elif acc_improvement < -0.01:
            interpretation = "⚠️ Preprocessing reduced model performance - consider different method"
            color = "orange"
        else:
            interpretation = "ℹ️ Preprocessing had minimal impact on performance"
            color = "blue"
            
        tk.Label(summary_frame, text=interpretation, fg=color, font=("Arial", 10, "italic")).pack(pady=2)
        
        # Close button
        tk.Button(
            results_window, 
            text="Close", 
            command=results_window.destroy,
            width=15,
            bg="#f0f0f0"
        ).pack(pady=10)
    
    def _draw_confusion_matrix(self, parent, conf_matrix, size=40):
        """Helper method to draw a confusion matrix"""
        cm_canvas = tk.Canvas(parent, width=size*3, height=size*3, bg="white")
        cm_canvas.pack()
        
        margin = size//2
        
        # Draw the matrix
        for i in range(2):
            for j in range(2):
                x = margin + j * size
                y = margin + i * size
                
                # Color based on value (darker = higher)
                val = conf_matrix[i, j]
                max_val = np.max(conf_matrix)
                intensity = int(255 - (val / max_val) * 100) if max_val > 0 else 255
                color = f"#{intensity:02x}{intensity:02x}{intensity:02x}"
                
                # Draw cell
                cm_canvas.create_rectangle(x, y, x + size, y + size, fill=color, outline="black")
                
                # Draw text
                cm_canvas.create_text(x + size//2, y + size//2, text=str(val), font=("Arial", 8))
        
        # Labels
        cm_canvas.create_text(margin + size//2, margin - 10, text="0", font=("Arial", 8))
        cm_canvas.create_text(margin + size + size//2, margin - 10, text="1", font=("Arial", 8))
        cm_canvas.create_text(margin - 10, margin + size//2, text="0", font=("Arial", 8))
        cm_canvas.create_text(margin - 10, margin + size + size//2, text="1", font=("Arial", 8))
    
    def show_training_results(self, accuracy, precision, recall, f1, conf_matrix):
        """Display the training results in a new window (legacy method)"""
        results_window = tk.Toplevel(self.root)
        results_window.title(f"Decision Tree Results - {self.dataset_name}")
        results_window.geometry("500x400")
        
        # Title
        tk.Label(
            results_window, 
            text="Decision Tree Classification Results",
            font=("Arial", 14, "bold")
        ).pack(pady=10)
        
        # Metrics frame
        metrics_frame = tk.LabelFrame(results_window, text="Performance Metrics")
        metrics_frame.pack(fill=tk.X, padx=20, pady=10)
        
        # Display metrics
        metrics = [
            ("Accuracy:", f"{accuracy:.4f}"),
            ("Precision:", f"{precision:.4f}"),
            ("Recall:", f"{recall:.4f}"),
            ("F1 Score:", f"{f1:.4f}")
        ]
        
        for i, (label, value) in enumerate(metrics):
            tk.Label(metrics_frame, text=label, anchor=tk.W, width=10).grid(row=i, column=0, sticky=tk.W, padx=5, pady=3)
            tk.Label(metrics_frame, text=value, anchor=tk.W).grid(row=i, column=1, sticky=tk.W, padx=5, pady=3)
        
        # Confusion matrix
        cm_frame = tk.LabelFrame(results_window, text="Confusion Matrix")
        cm_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Create canvas for the confusion matrix
        cm_canvas = tk.Canvas(cm_frame, bg="white")
        cm_canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Draw confusion matrix
        cell_size = 50
        margin = 30
        
        # Draw labels
        cm_canvas.create_text(margin, margin - 15, text="Actual", anchor=tk.CENTER)
        cm_canvas.create_text(margin + cell_size * 2, margin - 15, text="Predicted", anchor=tk.CENTER)
        
        # Draw the matrix
        for i in range(2):
            for j in range(2):
                x = margin + j * cell_size
                y = margin + i * cell_size
                
                # Draw cell
                cm_canvas.create_rectangle(x, y, x + cell_size, y + cell_size, fill="#f0f0f0", outline="black")
                
                # Draw text
                cm_canvas.create_text(x + cell_size/2, y + cell_size/2, text=str(conf_matrix[i, j]), font=("Arial", 12))
            
            # Row labels (0, 1)
            cm_canvas.create_text(margin - 15, margin + i * cell_size + cell_size/2, text=str(i), font=("Arial", 10))
            
            # Column labels (0, 1)
            cm_canvas.create_text(margin + i * cell_size + cell_size/2, margin - 15, text=str(i), font=("Arial", 10))
        
        # Close button
        tk.Button(
            results_window, 
            text="Close", 
            command=results_window.destroy,
            width=10
        ).pack(pady=10)

if __name__ == "__main__":
    root = tk.Tk()
    app = DataLoaderApp(root)
    root.mainloop()
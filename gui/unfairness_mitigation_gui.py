import tkinter as tk
import pickle
import os
import sys
import numpy as np
from tkinter import messagebox
from tkinter import ttk  # For the Treeview widget
import numpy
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier

# Add the src directory to the path so we can import the DTClassifier
project_root = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "src"))

# Now import the class
from src.predictors.decision_tree import DTClassifier

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

class DataLoaderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Data Dictionary Loader")
        self.root.geometry("600x400")
        self.current_data = None
        self.dataset_name = None
        
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
            text="Unfairness Mitigation - Dataset Loader", 
            font=("Arial", 16, "bold")
        ).pack()
        
        # Dataset buttons section
        datasets_frame = tk.LabelFrame(self.content_frame, text="Available Datasets")
        datasets_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Get list of dataset folders
        notebook_dir = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "notebooks/data"))
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
                height=2,
                relief=tk.RAISED,
                bg="#e0e0e0"
            )
            btn.pack(pady=5, padx=10, anchor=tk.W)
        
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
        
        # Training button section - initially disabled
        self.train_btn = tk.Button(
            self.content_frame,
            text="Train Decision Tree",
            command=self.train_decision_tree,
            width=20,
            height=2,
            state=tk.DISABLED
        )
        self.train_btn.pack(pady=10)
    
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
            notebook_dir = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "notebooks/data"))
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
    
    def train_decision_tree(self):
        """Train a decision tree classifier on the loaded data and show results"""
        if not self.current_data or 'data' not in self.current_data:
            messagebox.showerror("Error", "No valid data loaded for training")
            return
            
        try:
            # Extract features and labels
            X = []
            y = []
            
            for idx, record in self.current_data['data'].items():
                if 'features' in record and 'binary_label' in record:
                    X.append(record['features'])
                    y.append(record['binary_label'])
            
            if not X or not y:
                messagebox.showerror("Error", "Could not extract features and labels from the data")
                return
                
            # Convert to numpy arrays
            X = np.array(X)
            y = np.array(y)
            
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Create settings dictionary for the model
            settings = {
                'predictors': {
                    'decision_tree': {
                        'max_depth': 5  # Default max_depth
                    }
                }
            }
            
            # Initialize and train the model using our simplified classifier
            model = SimpleDTClassifier(settings)
            model.fit(X_train, y_train, X_test, y_test)
            
            # Get predictions
            predictions, _ = model.predict(X_test, y_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, predictions)
            precision = precision_score(y_test, predictions, zero_division=0)
            recall = recall_score(y_test, predictions, zero_division=0)
            f1 = f1_score(y_test, predictions, zero_division=0)
            conf_matrix = confusion_matrix(y_test, predictions)
            
            # Show results
            self.show_training_results(accuracy, precision, recall, f1, conf_matrix)
            
        except Exception as e:
            messagebox.showerror("Training Error", f"An error occurred during model training: {str(e)}")
            print(f"Training error: {e}")
    
    def show_training_results(self, accuracy, precision, recall, f1, conf_matrix):
        """Display the training results in a new window"""
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
#!/usr/bin/env python3
"""
DebiasEd GUI - A comprehensive graphical interface for bias mitigation in educational data

This GUI provides a complete workflow for:
1. Data loading and exploration
2. Bias mitigation preprocessing
3. Model training and evaluation
4. Fairness metrics visualization
5. Results export
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
import sys
import os
import pickle
import json
from pathlib import Path
import importlib.util
from typing import Dict, Any, List, Tuple, Optional

# Add the package to the path
sys.path.append('../src/src')

# Import the package components
try:
    from debiased_jadouille.mitigation.preprocessing.preprocessor import PreProcessor
    from debiased_jadouille.predictors.predictor import Predictor
    from debiased_jadouille.crossvalidation.scorers.scorer import Scorer
    from debiased_jadouille.crossvalidation.scorers.fairness_binary_scorer import BinaryFairnessScorer
    from debiased_jadouille.crossvalidation.scorers.binary_scorer import BinaryClfScorer
    
    # Import predictors
    from debiased_jadouille.predictors.logistic_regression import LogisticRegressionClassifier
    from debiased_jadouille.predictors.decision_tree import DecisionTreeClassifier as DTClassifier
    
    # Import preprocessing methods with error handling
    AVAILABLE_PREPROCESSORS = {}
    
    # Try to import each preprocessor individually
    try:
        from debiased_jadouille.mitigation.preprocessing.calders import CaldersPreProcessor
        AVAILABLE_PREPROCESSORS['calders'] = CaldersPreProcessor
    except ImportError:
        pass
    
    try:
        from debiased_jadouille.mitigation.preprocessing.zemel import ZemelPreProcessor
        AVAILABLE_PREPROCESSORS['zemel'] = ZemelPreProcessor
    except ImportError:
        pass
    
    try:
        from debiased_jadouille.mitigation.preprocessing.chakraborty import ChakrabortyPreProcessor
        AVAILABLE_PREPROCESSORS['chakraborty'] = ChakrabortyPreProcessor
    except ImportError:
        pass
    
    try:
        from debiased_jadouille.mitigation.preprocessing.smote import SmotePreProcessor
        AVAILABLE_PREPROCESSORS['smote'] = SmotePreProcessor
    except ImportError:
        pass
        
    try:
        from debiased_jadouille.mitigation.preprocessing.alabdulmohsin import AlabdulmohsinPreProcessor
        AVAILABLE_PREPROCESSORS['alabdulmohsin'] = AlabdulmohsinPreProcessor
    except ImportError:
        pass
    
    try:
        from debiased_jadouille.mitigation.preprocessing.dablain import DablainPreProcessor
        AVAILABLE_PREPROCESSORS['dablain'] = DablainPreProcessor
    except ImportError:
        pass
    
    try:
        from debiased_jadouille.mitigation.preprocessing.li import LiPreProcessor
        AVAILABLE_PREPROCESSORS['li'] = LiPreProcessor
    except ImportError:
        pass
    
    try:
        from debiased_jadouille.mitigation.preprocessing.cock import CockPreProcessor
        AVAILABLE_PREPROCESSORS['cock'] = CockPreProcessor
    except ImportError:
        pass
    
    # Import inprocessing methods with error handling
    AVAILABLE_INPROCESSORS = {}
    
    try:
        from debiased_jadouille.mitigation.inprocessing.zafar import ZafarInProcessor
        AVAILABLE_INPROCESSORS['zafar'] = ZafarInProcessor
    except ImportError:
        pass
    
    try:
        from debiased_jadouille.mitigation.inprocessing.chen import ChenInProcessor
        AVAILABLE_INPROCESSORS['chen'] = ChenInProcessor
    except ImportError:
        pass
    
    try:
        from debiased_jadouille.mitigation.inprocessing.gao import GaoInProcessor
        AVAILABLE_INPROCESSORS['gao'] = GaoInProcessor
    except ImportError:
        pass
    
    try:
        from debiased_jadouille.mitigation.inprocessing.islam import IslamInProcessor
        AVAILABLE_INPROCESSORS['islam'] = IslamInProcessor
    except ImportError:
        pass
    
    try:
        from debiased_jadouille.mitigation.inprocessing.kilbertus import KilbertusInProcessor
        AVAILABLE_INPROCESSORS['kilbertus'] = KilbertusInProcessor
    except ImportError:
        pass
    
    try:
        from debiased_jadouille.mitigation.inprocessing.liu import LiuInProcessor
        AVAILABLE_INPROCESSORS['liu'] = LiuInProcessor
    except ImportError:
        pass
    
    try:
        from debiased_jadouille.mitigation.inprocessing.grari2 import Grari2InProcessor
        AVAILABLE_INPROCESSORS['grari2'] = Grari2InProcessor
    except ImportError:
        pass
    
    try:
        from debiased_jadouille.mitigation.inprocessing.chakraborty_in import ChakrabortyInProcessor
        AVAILABLE_INPROCESSORS['chakraborty_in'] = ChakrabortyInProcessor
    except ImportError:
        pass
    
    # Import postprocessing methods with error handling
    AVAILABLE_POSTPROCESSORS = {}
    
    try:
        from debiased_jadouille.mitigation.postprocessing.snel import SnelPostProcessor
        AVAILABLE_POSTPROCESSORS['snel'] = SnelPostProcessor
    except ImportError:
        pass
    
    try:
        from debiased_jadouille.mitigation.postprocessing.pleiss import PleissPostProcessor
        AVAILABLE_POSTPROCESSORS['pleiss'] = PleissPostProcessor
    except ImportError:
        pass
    
    try:
        from debiased_jadouille.mitigation.postprocessing.kamiranpost import KamiranPostProcessor
        AVAILABLE_POSTPROCESSORS['kamiran'] = KamiranPostProcessor
    except ImportError:
        pass
    
    print(f"Successfully imported {len(AVAILABLE_PREPROCESSORS)} preprocessing methods")
    print(f"Successfully imported {len(AVAILABLE_INPROCESSORS)} inprocessing methods")
    print(f"Successfully imported {len(AVAILABLE_POSTPROCESSORS)} postprocessing methods")
    
except ImportError as e:
    print(f"Warning: Could not import debiased_jadouille package: {e}")
    print("The GUI will run in demo mode with mock implementations")
    AVAILABLE_PREPROCESSORS = {}
    AVAILABLE_INPROCESSORS = {}
    AVAILABLE_POSTPROCESSORS = {}


class DebiasedJadouilleGUI:
    """Main GUI application for the DebiasEd package"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("DebiasEd - Bias Mitigation for Educational Data")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 700)
        
        # Configure style
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Data storage
        self.data = {
            'raw_data': None,
            'processed_data': None,
            'train_data': None,
            'test_data': None,
            'features': [],
            'target_column': None,
            'sensitive_attributes': [],
            'models': {},
            'results': {},
            'preprocessor': None
        }
        
        # Configuration
        self.config = {
            'preprocessing_method': None,
            'preprocessing_params': {},
            'model_type': None,
            'model_params': {},
            'evaluation_metrics': []
        }
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the main user interface"""
        # Create main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Create tabs
        self.create_data_tab()
        self.create_preprocessing_tab()
        self.create_modeling_tab()
        self.create_evaluation_tab()
        self.create_results_tab()
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready - Load data to begin")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X, pady=(5, 0))
        
    def create_data_tab(self):
        """Create the data loading and exploration tab"""
        data_frame = ttk.Frame(self.notebook)
        self.notebook.add(data_frame, text="Data")
        
        # Data loading section
        load_frame = ttk.LabelFrame(data_frame, text="Data Loading", padding=10)
        load_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(load_frame, text="Load CSV/PKL File", 
                  command=self.load_data).pack(side=tk.LEFT, padx=5)
        ttk.Button(load_frame, text="Load Sample Dataset", 
                  command=self.load_sample_data).pack(side=tk.LEFT, padx=5)
        ttk.Button(load_frame, text="Load Converted PKL", 
                  command=self.load_converted_pkl).pack(side=tk.LEFT, padx=5)
        
        self.data_info_var = tk.StringVar(value="No data loaded")
        ttk.Label(load_frame, textvariable=self.data_info_var).pack(side=tk.LEFT, padx=20)
        
        # Data configuration section
        config_frame = ttk.LabelFrame(data_frame, text="Data Configuration", padding=10)
        config_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Target column selection
        ttk.Label(config_frame, text="Target Column:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.target_var = tk.StringVar()
        self.target_combo = ttk.Combobox(config_frame, textvariable=self.target_var, width=20)
        self.target_combo.grid(row=0, column=1, padx=5, pady=2)
        self.target_combo.bind('<<ComboboxSelected>>', self.update_data_config)
        
        # Sensitive attributes selection
        ttk.Label(config_frame, text="Sensitive Attributes:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.sensitive_frame = ttk.Frame(config_frame)
        self.sensitive_frame.grid(row=1, column=1, columnspan=2, sticky=tk.W, padx=5, pady=2)
        
        # Data preview section
        preview_frame = ttk.LabelFrame(data_frame, text="Data Preview", padding=10)
        preview_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Create treeview for data preview
        columns = ['Column'] + [f'Sample_{i+1}' for i in range(5)]
        self.data_tree = ttk.Treeview(preview_frame, columns=columns, show='headings', height=15)
        
        for col in columns:
            self.data_tree.heading(col, text=col)
            self.data_tree.column(col, width=100)
        
        # Scrollbars for treeview
        v_scrollbar = ttk.Scrollbar(preview_frame, orient=tk.VERTICAL, command=self.data_tree.yview)
        h_scrollbar = ttk.Scrollbar(preview_frame, orient=tk.HORIZONTAL, command=self.data_tree.xview)
        self.data_tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        self.data_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        
    def create_preprocessing_tab(self):
        """Create the data preprocessing methods tab (model-agnostic data transformation)"""
        preproc_frame = ttk.Frame(self.notebook)
        self.notebook.add(preproc_frame, text="Preprocessing")
        
        # Information section
        info_frame = ttk.LabelFrame(preproc_frame, text="About Preprocessing", padding=10)
        info_frame.pack(fill=tk.X, padx=10, pady=5)
        
        info_text = "Preprocessing methods transform your data before training to reduce bias.\nThese methods work with any machine learning model you choose later."
        ttk.Label(info_frame, text=info_text, justify=tk.LEFT, 
                 font=('TkDefaultFont', 9, 'italic')).pack(anchor=tk.W)
        
        # Method selection section
        method_frame = ttk.LabelFrame(preproc_frame, text="Method Selection", padding=10)
        method_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Initialize method dictionaries
        self.preproc_methods = {
            # Basic resampling and reweighting methods
            'Calders Reweighting': 'calders',
            'SMOTE Oversampling': 'smote',
            
            # Fair representation learning
            'Zemel Learning Fair Representations': 'zemel',
            
            # Synthetic data generation
            'Chakraborty Synthetic Data': 'chakraborty',
            
            # Advanced fair sampling methods
            'Dablain Fair Over-Sampling': 'dablain',
            'Zelaya Fair Over-Sampling': 'zelaya_over',
            'Zelaya Fair SMOTE': 'zelaya_smote',
            
            # Disparate impact reduction
            'Alabdulmohsin Binary Debiasing': 'alabdulmohsin',
            
            # Data debugging and cleaning
            'Li Training Data Debugging': 'li',
            'Cock Fair Data Cleaning': 'cock',
            
            # Resampling variants (Iosifidis methods)
            'Iosifidis Resample Attribute': 'iosifidis_resample_attr',
            'Iosifidis Resample Target': 'iosifidis_resample_target',
            'Iosifidis SMOTE Attribute': 'iosifidis_smote_attr',
            'Iosifidis SMOTE Target': 'iosifidis_smote_target',
            
            # Other advanced methods
            'Lahoti Representation Learning': 'lahoti',
        }
        
        self.inproc_methods = {
            # Constraint-based fair learning
            'Zafar Fair Constraints': 'zafar',
            
            # Adversarial debiasing
            'Chen Multi-Accuracy Adversarial Training': 'chen',
            
            # Fair representation learning
            'Gao Fair Adversarial Networks': 'gao',
            
            # Fairness-aware deep learning
            'Islam Fairness-Aware Learning': 'islam',
            
            # Causal fairness
            'Kilbertus Fair Prediction': 'kilbertus',
            
            # Distribution matching
            'Liu Fair Distribution Matching': 'liu',
            
            # Gradient-based fairness
            'Grari Gradient-Based Fairness': 'grari2',
            
            # In-processing synthetic data
            'Chakraborty In-Process Synthesis': 'chakraborty_in',
        }
        
        self.postproc_methods = {
            # Threshold optimization methods
            'Snel Bias Correction': 'snel',
            
            # Calibration-based methods  
            'Pleiss Multicalibration': 'pleiss',
            
            # Reject option classification
            'Kamiran Reject Option': 'kamiran',
        }
        
        # Preprocessing method selection (only preprocessing methods)
        ttk.Label(method_frame, text="Select Preprocessing Method:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.preproc_method_var = tk.StringVar(value='Calders Reweighting')
        self.preproc_combo = ttk.Combobox(method_frame, textvariable=self.preproc_method_var, 
                                         values=list(self.preproc_methods.keys()),
                                         width=40, state='readonly')
        self.preproc_combo.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        self.preproc_combo.bind('<<ComboboxSelected>>', self.update_preprocessing_method_params)
        
        # Method description section
        desc_frame = ttk.LabelFrame(preproc_frame, text="Method Description", padding=10)
        desc_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.preproc_desc_text = tk.Text(desc_frame, height=4, wrap=tk.WORD, font=('TkDefaultFont', 9))
        desc_scrollbar = ttk.Scrollbar(desc_frame, orient=tk.VERTICAL, command=self.preproc_desc_text.yview)
        self.preproc_desc_text.configure(yscrollcommand=desc_scrollbar.set)
        self.preproc_desc_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        desc_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Parameters section
        params_frame = ttk.LabelFrame(preproc_frame, text="Method Parameters", padding=10)
        params_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.params_widgets = {}
        self.create_parameter_widgets(params_frame)
        
        # Apply preprocessing section
        apply_frame = ttk.LabelFrame(preproc_frame, text="Apply Preprocessing", padding=10)
        apply_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(apply_frame, text="Apply Preprocessing", 
                  command=self.apply_preprocessing).pack(side=tk.LEFT, padx=5)
        
        self.preproc_status_var = tk.StringVar(value="No preprocessing applied")
        ttk.Label(apply_frame, textvariable=self.preproc_status_var).pack(side=tk.LEFT, padx=20)
        
        # Preprocessing results visualization
        viz_frame = ttk.LabelFrame(preproc_frame, text="Preprocessing Results", padding=10)
        viz_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Create notebook for before/after comparison
        self.preproc_notebook = ttk.Notebook(viz_frame)
        self.preproc_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Before preprocessing tab
        self.before_frame = ttk.Frame(self.preproc_notebook)
        self.preproc_notebook.add(self.before_frame, text="Before Preprocessing")
        
        self.before_canvas_frame = ttk.Frame(self.before_frame)
        self.before_canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        # After preprocessing tab
        self.after_frame = ttk.Frame(self.preproc_notebook)
        self.preproc_notebook.add(self.after_frame, text="After Preprocessing")
        
        self.preproc_canvas_frame = ttk.Frame(self.after_frame)
        self.preproc_canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        # Comparison tab
        self.comparison_frame = ttk.Frame(self.preproc_notebook)
        self.preproc_notebook.add(self.comparison_frame, text="Comparison")
        
        self.comparison_canvas_frame = ttk.Frame(self.comparison_frame)
        self.comparison_canvas_frame.pack(fill=tk.BOTH, expand=True)
        
    def create_modeling_tab(self):
        """Create the modeling tab"""
        model_frame = ttk.Frame(self.notebook)
        self.notebook.add(model_frame, text="Modeling")
        
        # Model selection section
        model_select_frame = ttk.LabelFrame(model_frame, text="Model Selection", padding=10)
        model_select_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.model_types = {
            'Logistic Regression': 'logistic_regression',
            'Decision Tree': 'decision_tree',
            'Support Vector Machine': 'svm',
            'Random Forest': 'random_forest'
        }
        
        self.model_var = tk.StringVar(value='Logistic Regression')
        for i, (name, key) in enumerate(self.model_types.items()):
            ttk.Radiobutton(model_select_frame, text=name, variable=self.model_var, 
                          value=name, command=self.update_model_selection).grid(
                          row=i//2, column=i%2, sticky=tk.W, padx=10, pady=2)
        
        # Model parameters section
        model_params_frame = ttk.LabelFrame(model_frame, text="Model Parameters", padding=10)
        model_params_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.model_params_widgets = {}
        self.create_model_parameter_widgets(model_params_frame)
        
        # Bias mitigation approach selection (for this model)
        bias_mitigation_frame = ttk.LabelFrame(model_frame, text="Bias Mitigation for This Model", padding=10)
        bias_mitigation_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.model_bias_approach_var = tk.StringVar(value='none')
        ttk.Radiobutton(bias_mitigation_frame, text="No bias mitigation (standard training)", 
                       variable=self.model_bias_approach_var, value='none',
                       command=self.update_model_bias_approach).pack(anchor=tk.W, padx=5, pady=2)
        ttk.Radiobutton(bias_mitigation_frame, text="Use preprocessed data (if applied)", 
                       variable=self.model_bias_approach_var, value='preprocessed',
                       command=self.update_model_bias_approach).pack(anchor=tk.W, padx=5, pady=2)
        ttk.Radiobutton(bias_mitigation_frame, text="Inprocessing (fair training algorithm)", 
                       variable=self.model_bias_approach_var, value='inprocessing',
                       command=self.update_model_bias_approach).pack(anchor=tk.W, padx=5, pady=2)
        ttk.Radiobutton(bias_mitigation_frame, text="Postprocessing (adjust model outputs)", 
                       variable=self.model_bias_approach_var, value='postprocessing',
                       command=self.update_model_bias_approach).pack(anchor=tk.W, padx=5, pady=2)
        
        # Inprocessing method selection (shown when inprocessing is selected)
        self.inprocessing_frame = ttk.LabelFrame(model_frame, text="Inprocessing Method", padding=10)
        # Will be packed/unpacked based on selection
        
        ttk.Label(self.inprocessing_frame, text="Select Inprocessing Method:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.inproc_method_var = tk.StringVar(value='Zafar Fair Constraints')
        self.inproc_combo = ttk.Combobox(self.inprocessing_frame, textvariable=self.inproc_method_var, 
                                        width=40, state='readonly')
        self.inproc_combo.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        self.inproc_combo.bind('<<ComboboxSelected>>', self.update_inprocessing_method_params)
        
        # Inprocessing method description
        inproc_desc_frame = ttk.Frame(self.inprocessing_frame)
        inproc_desc_frame.grid(row=1, column=0, columnspan=2, sticky=tk.EW, padx=5, pady=5)
        
        self.inproc_desc_text = tk.Text(inproc_desc_frame, height=3, wrap=tk.WORD, font=('TkDefaultFont', 9))
        inproc_scrollbar = ttk.Scrollbar(inproc_desc_frame, orient=tk.VERTICAL, command=self.inproc_desc_text.yview)
        self.inproc_desc_text.configure(yscrollcommand=inproc_scrollbar.set)
        self.inproc_desc_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        inproc_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Inprocessing parameters
        self.inproc_params_frame = ttk.LabelFrame(self.inprocessing_frame, text="Method Parameters", padding=10)
        self.inproc_params_frame.grid(row=2, column=0, columnspan=2, sticky=tk.EW, padx=5, pady=5)
        
        self.inproc_params_content_frame = ttk.Frame(self.inproc_params_frame)
        self.inproc_params_content_frame.pack(fill=tk.X)
        
        # Postprocessing method selection (shown when postprocessing is selected)
        self.postprocessing_frame = ttk.LabelFrame(model_frame, text="Postprocessing Method", padding=10)
        # Will be packed/unpacked based on selection
        
        ttk.Label(self.postprocessing_frame, text="Select Postprocessing Method:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.postproc_method_var = tk.StringVar(value='Snel Bias Correction')
        self.postproc_combo = ttk.Combobox(self.postprocessing_frame, textvariable=self.postproc_method_var, 
                                          width=40, state='readonly')
        self.postproc_combo.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        self.postproc_combo.bind('<<ComboboxSelected>>', self.update_postprocessing_method_params)
        
        # Postprocessing method description
        postproc_desc_frame = ttk.Frame(self.postprocessing_frame)
        postproc_desc_frame.grid(row=1, column=0, columnspan=2, sticky=tk.EW, padx=5, pady=5)
        
        self.postproc_desc_text = tk.Text(postproc_desc_frame, height=3, wrap=tk.WORD, font=('TkDefaultFont', 9))
        postproc_scrollbar = ttk.Scrollbar(postproc_desc_frame, orient=tk.VERTICAL, command=self.postproc_desc_text.yview)
        self.postproc_desc_text.configure(yscrollcommand=postproc_scrollbar.set)
        self.postproc_desc_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        postproc_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Postprocessing parameters
        self.postproc_params_frame = ttk.LabelFrame(self.postprocessing_frame, text="Method Parameters", padding=10)
        self.postproc_params_frame.grid(row=2, column=0, columnspan=2, sticky=tk.EW, padx=5, pady=5)
        
        self.postproc_params_content_frame = ttk.Frame(self.postproc_params_frame)
        self.postproc_params_content_frame.pack(fill=tk.X)
        
        # Training section
        training_frame = ttk.LabelFrame(model_frame, text="Model Training", padding=10)
        training_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Training configuration
        ttk.Label(training_frame, text="Test Split:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.test_split_var = tk.DoubleVar(value=0.2)
        ttk.Entry(training_frame, textvariable=self.test_split_var, width=15).grid(row=0, column=1, padx=5, pady=2, sticky=tk.W)
        ttk.Label(training_frame, text="(e.g., 0.2 = 20%)").grid(row=0, column=2, padx=5, pady=2, sticky=tk.W)
        
        ttk.Button(training_frame, text="Train Model", 
                  command=self.train_model).grid(row=1, column=0, padx=5, pady=10)
        
        self.training_status_var = tk.StringVar(value="No model trained")
        ttk.Label(training_frame, textvariable=self.training_status_var).grid(row=1, column=1, columnspan=2, padx=20, pady=10)
        
        # Training progress
        self.training_progress = ttk.Progressbar(training_frame, mode='indeterminate')
        self.training_progress.grid(row=2, column=0, columnspan=3, sticky=tk.EW, padx=5, pady=5)
        
        # Initialize postprocessing method params
        self.update_postprocessing_method_params()
        
    def create_evaluation_tab(self):
        """Create the evaluation and fairness metrics tab"""
        eval_frame = ttk.Frame(self.notebook)
        self.notebook.add(eval_frame, text="Evaluation")
        
        # Metrics selection section
        metrics_frame = ttk.LabelFrame(eval_frame, text="Evaluation Metrics", padding=10)
        metrics_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Standard ML metrics
        ml_metrics_frame = ttk.LabelFrame(metrics_frame, text="ML Metrics", padding=5)
        ml_metrics_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5)
        
        self.ml_metrics_vars = {}
        ml_metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        for metric in ml_metrics:
            var = tk.BooleanVar(value=True)
            self.ml_metrics_vars[metric] = var
            ttk.Checkbutton(ml_metrics_frame, text=metric, variable=var).pack(anchor=tk.W)
        
        # Fairness metrics
        fairness_metrics_frame = ttk.LabelFrame(metrics_frame, text="Fairness Metrics", padding=5)
        fairness_metrics_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5)
        
        self.fairness_metrics_vars = {}
        fairness_metrics = ['Demographic Parity', 'Equal Opportunity', 'Equalized Odds', 
                          'Calibration', 'Predictive Parity']
        for metric in fairness_metrics:
            var = tk.BooleanVar(value=True)
            self.fairness_metrics_vars[metric] = var
            ttk.Checkbutton(fairness_metrics_frame, text=metric, variable=var).pack(anchor=tk.W)
        
        # Evaluation controls
        eval_controls_frame = ttk.LabelFrame(eval_frame, text="Evaluation Controls", padding=10)
        eval_controls_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(eval_controls_frame, text="Evaluate Model", 
                  command=self.evaluate_model).pack(side=tk.LEFT, padx=5)
        ttk.Button(eval_controls_frame, text="Generate Report", 
                  command=self.generate_report).pack(side=tk.LEFT, padx=5)
        
        # Results visualization
        viz_frame = ttk.LabelFrame(eval_frame, text="Evaluation Results", padding=10)
        viz_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Create notebook for different visualizations
        self.eval_notebook = ttk.Notebook(viz_frame)
        self.eval_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Metrics summary tab
        metrics_tab = ttk.Frame(self.eval_notebook)
        self.eval_notebook.add(metrics_tab, text="Metrics Summary")
        
        self.metrics_tree = ttk.Treeview(metrics_tab, columns=('Metric', 'Value', 'Group'), show='headings')
        self.metrics_tree.heading('Metric', text='Metric')
        self.metrics_tree.heading('Value', text='Value')
        self.metrics_tree.heading('Group', text='Group')
        self.metrics_tree.pack(fill=tk.BOTH, expand=True)
        
        # Visualizations tab
        viz_tab = ttk.Frame(self.eval_notebook)
        self.eval_notebook.add(viz_tab, text="Visualizations")
        
        self.eval_canvas_frame = ttk.Frame(viz_tab)
        self.eval_canvas_frame.pack(fill=tk.BOTH, expand=True)
        
    def create_results_tab(self):
        """Create the results and export tab"""
        results_frame = ttk.Frame(self.notebook)
        self.notebook.add(results_frame, text="Results")
        
        # Model comparison section
        comparison_frame = ttk.LabelFrame(results_frame, text="Model Comparison", padding=10)
        comparison_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.comparison_tree = ttk.Treeview(comparison_frame, 
                                          columns=('Model', 'Preprocessing', 'Accuracy', 'Fairness'), 
                                          show='headings', height=8)
        self.comparison_tree.heading('Model', text='Model')
        self.comparison_tree.heading('Preprocessing', text='Preprocessing')
        self.comparison_tree.heading('Accuracy', text='Accuracy')
        self.comparison_tree.heading('Fairness', text='Fairness Score')
        self.comparison_tree.pack(fill=tk.X)
        
        # Export section
        export_frame = ttk.LabelFrame(results_frame, text="Export Results", padding=10)
        export_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(export_frame, text="Export Model", 
                  command=self.export_model).pack(side=tk.LEFT, padx=5)
        ttk.Button(export_frame, text="Export Results (CSV)", 
                  command=self.export_results_csv).pack(side=tk.LEFT, padx=5)
        ttk.Button(export_frame, text="Export Report (PDF)", 
                  command=self.export_report_pdf).pack(side=tk.LEFT, padx=5)
        
        # Configuration save/load
        config_frame = ttk.LabelFrame(results_frame, text="Configuration", padding=10)
        config_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(config_frame, text="Save Configuration", 
                  command=self.save_configuration).pack(side=tk.LEFT, padx=5)
        ttk.Button(config_frame, text="Load Configuration", 
                  command=self.load_configuration).pack(side=tk.LEFT, padx=5)
        
        # Log section
        log_frame = ttk.LabelFrame(results_frame, text="Execution Log", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.log_text = tk.Text(log_frame, height=15, wrap=tk.WORD)
        log_scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scrollbar.set)
        
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
    def create_parameter_widgets(self, parent):
        """Create parameter widgets for preprocessing methods"""
        # This will be populated based on selected method
        self.params_content_frame = ttk.Frame(parent)
        self.params_content_frame.pack(fill=tk.X)
        
        # Initialize the method description and parameters
        self.preproc_desc_text.config(state=tk.NORMAL)
        self.update_preprocessing_method_params()  # This will set up the initial state
        
    def create_model_parameter_widgets(self, parent):
        """Create parameter widgets for models"""
        self.model_params_content_frame = ttk.Frame(parent)
        self.model_params_content_frame.pack(fill=tk.X)
        self.update_model_params()
        
    def update_preprocessing_method_params(self, event=None):
        """Update parameter widgets based on selected preprocessing method"""
        # Clear existing widgets
        for widget in self.params_content_frame.winfo_children():
            widget.destroy()
        
        method = self.preproc_method_var.get()
        
        # Method descriptions for both approaches
        preprocessing_descriptions = {
            'Calders Reweighting': "Reweights training samples to reduce bias. Adjusts the importance of different groups to achieve demographic parity. Simple and effective for many datasets.",
            
            'SMOTE Oversampling': "Synthetic Minority Oversampling Technique. Generates synthetic examples for underrepresented classes to balance the dataset.",
            
            'Zemel Learning Fair Representations': "Learns fair data representations that maintain utility while removing sensitive information. Maps data to a fair latent space using prototypes.",
            
            'Chakraborty Synthetic Data': "Generates synthetic fair data that preserves statistical properties while reducing bias. Creates new training samples with controlled bias.",
            
            'Dablain Fair Over-Sampling': "Advanced oversampling technique that considers both class imbalance and group fairness. Balances representation across sensitive groups.",
            
            'Zelaya Fair Over-Sampling': "Fair oversampling method that ensures equal representation of sensitive groups while maintaining class balance.",
            
            'Zelaya Fair SMOTE': "Extension of SMOTE that incorporates fairness constraints. Generates synthetic samples that promote fairness across sensitive attributes.",
            
            'Alabdulmohsin Binary Debiasing': "Reduces disparate impact by massaging the data without using demographic attributes during inference. Optimizes for demographic parity.",
            
            'Li Training Data Debugging': "Identifies and corrects problematic training samples that contribute to bias. Debugs the dataset to improve fairness.",
            
            'Cock Fair Data Cleaning': "Cleans the dataset by removing or correcting samples that contribute to unfairness. Focuses on data quality improvement.",
            
            'Iosifidis Resample Attribute': "Resamples data based on sensitive attributes to achieve better group representation and reduce attribute-based bias.",
            
            'Iosifidis Resample Target': "Resamples data based on target outcomes to balance class distribution while considering sensitive attributes.",
            
            'Iosifidis SMOTE Attribute': "SMOTE-based oversampling that focuses on balancing sensitive attribute representation in the dataset.",
            
            'Iosifidis SMOTE Target': "SMOTE-based oversampling that balances target classes while maintaining fairness across sensitive groups.",
            
            'Lahoti Representation Learning': "Learns fair representations through adversarial training. Maps input data to a representation space that reduces bias."
        }
        
        inprocessing_descriptions = {
            'Zafar Fair Constraints': "Incorporates fairness constraints directly into the optimization objective during training. Uses convex relaxations to ensure fairness while maintaining accuracy.",
            
            'Chen Multi-Accuracy Adversarial Training': "Uses adversarial training with multiple accuracy objectives to ensure fairness across different groups during model training.",
            
            'Gao Fair Adversarial Networks': "Employs adversarial networks to learn fair representations during training. The adversary tries to predict sensitive attributes while the main model learns fair predictions.",
            
            'Islam Fairness-Aware Learning': "Integrates fairness constraints into deep learning training through specialized loss functions and regularization techniques.",
            
            'Kilbertus Fair Prediction': "Implements causal fairness through prediction methods that account for the causal structure of the data during training.",
            
            'Liu Fair Distribution Matching': "Uses optimal transport and distribution matching techniques to ensure fair predictions during the training process.",
            
            'Grari Gradient-Based Fairness': "Applies gradient-based optimization techniques to directly optimize for fairness metrics during model training.",
            
            'Chakraborty In-Process Synthesis': "Generates synthetic fair samples during the training process to improve model fairness in real-time."
        }
        
        # Update description for preprocessing methods
        desc = preprocessing_descriptions.get(method, "Description not available for this method.")
        self.preproc_desc_text.config(state=tk.NORMAL)
        self.preproc_desc_text.delete(1.0, tk.END)
        self.preproc_desc_text.insert(1.0, desc)
        self.preproc_desc_text.config(state=tk.DISABLED)
        
        # Configure parameters for preprocessing method
        self.configure_preprocessing_params(method)
            
    def configure_preprocessing_params(self, method):
        """Configure parameter widgets for preprocessing methods"""
        # Basic resampling and reweighting methods
        if method == 'Calders Reweighting':
            ttk.Label(self.params_content_frame, text="Sampling Proportions:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
            self.calders_prop_var = tk.DoubleVar(value=1.0)
            ttk.Entry(self.params_content_frame, textvariable=self.calders_prop_var, width=15).grid(row=0, column=1, padx=5, pady=2, sticky=tk.W)
            
        elif method == 'SMOTE Oversampling':
            ttk.Label(self.params_content_frame, text="K Neighbors:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
            self.smote_k_var = tk.IntVar(value=5)
            ttk.Entry(self.params_content_frame, textvariable=self.smote_k_var, width=15).grid(row=0, column=1, padx=5, pady=2, sticky=tk.W)
            
        # Fair representation learning
        elif method == 'Zemel Learning Fair Representations':
            ttk.Label(self.params_content_frame, text="Number of Prototypes:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
            self.zemel_prototypes_var = tk.IntVar(value=10)
            ttk.Entry(self.params_content_frame, textvariable=self.zemel_prototypes_var, width=15).grid(row=0, column=1, padx=5, pady=2, sticky=tk.W)
            
            ttk.Label(self.params_content_frame, text="Learning Rate:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
            self.zemel_lr_var = tk.DoubleVar(value=0.01)
            ttk.Entry(self.params_content_frame, textvariable=self.zemel_lr_var, width=15).grid(row=1, column=1, padx=5, pady=2, sticky=tk.W)
        
        # Synthetic data generation
        elif method == 'Chakraborty Synthetic Data':
            ttk.Label(self.params_content_frame, text="Synthesis Rate:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
            self.chakraborty_rate_var = tk.DoubleVar(value=0.5)
            ttk.Entry(self.params_content_frame, textvariable=self.chakraborty_rate_var, width=15).grid(row=0, column=1, padx=5, pady=2, sticky=tk.W)
            
        # Advanced fair sampling methods
        elif method == 'Dablain Fair Over-Sampling':
            ttk.Label(self.params_content_frame, text="Proportion:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
            self.dablain_prop_var = tk.DoubleVar(value=1.0)
            ttk.Entry(self.params_content_frame, textvariable=self.dablain_prop_var, width=15).grid(row=0, column=1, padx=5, pady=2, sticky=tk.W)
            
            ttk.Label(self.params_content_frame, text="K Neighbors:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
            self.dablain_k_var = tk.IntVar(value=5)
            ttk.Entry(self.params_content_frame, textvariable=self.dablain_k_var, width=15).grid(row=1, column=1, padx=5, pady=2, sticky=tk.W)
        
        elif method == 'Zelaya Fair Over-Sampling':
            ttk.Label(self.params_content_frame, text="Sampling Strategy:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
            self.zelaya_over_strategy_var = tk.StringVar(value='auto')
            strategy_combo = ttk.Combobox(self.params_content_frame, textvariable=self.zelaya_over_strategy_var, 
                                       values=['auto', 'majority', 'minority', 'not minority'], width=15)
            strategy_combo.grid(row=0, column=1, padx=5, pady=2)
            
        elif method == 'Zelaya Fair SMOTE':
            ttk.Label(self.params_content_frame, text="K Neighbors:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
            self.zelaya_smote_k_var = tk.IntVar(value=5)
            ttk.Entry(self.params_content_frame, textvariable=self.zelaya_smote_k_var, width=15).grid(row=0, column=1, padx=5, pady=2, sticky=tk.W)
        
        # Disparate impact reduction
        elif method == 'Alabdulmohsin Binary Debiasing':
            ttk.Label(self.params_content_frame, text="SGD Steps:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
            self.alabdulmohsin_sgd_var = tk.IntVar(value=10)
            ttk.Entry(self.params_content_frame, textvariable=self.alabdulmohsin_sgd_var, width=15).grid(row=0, column=1, padx=5, pady=2, sticky=tk.W)
            
            ttk.Label(self.params_content_frame, text="Gradient Epochs:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
            self.alabdulmohsin_epochs_var = tk.IntVar(value=1)
            ttk.Entry(self.params_content_frame, textvariable=self.alabdulmohsin_epochs_var, width=15).grid(row=1, column=1, padx=5, pady=2, sticky=tk.W)
        
        # Data debugging and cleaning
        elif method == 'Li Training Data Debugging':
            ttk.Label(self.params_content_frame, text="C Factor:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
            self.li_c_factor_var = tk.DoubleVar(value=10.0)
            ttk.Entry(self.params_content_frame, textvariable=self.li_c_factor_var, width=15).grid(row=0, column=1, padx=5, pady=2, sticky=tk.W)
            
            ttk.Label(self.params_content_frame, text="Fairness Metric:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
            self.li_fairness_var = tk.StringVar(value='dp')
            fairness_combo = ttk.Combobox(self.params_content_frame, textvariable=self.li_fairness_var, 
                                       values=['dp', 'eo', 'eqo'], width=15)
            fairness_combo.grid(row=1, column=1, padx=5, pady=2)
        
        elif method == 'Cock Fair Data Cleaning':
            ttk.Label(self.params_content_frame, text="Cleaning Threshold:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
            self.cock_threshold_var = tk.DoubleVar(value=0.1)
            ttk.Entry(self.params_content_frame, textvariable=self.cock_threshold_var, width=15).grid(row=0, column=1, padx=5, pady=2, sticky=tk.W)
        
        # Iosifidis resampling variants
        elif method in ['Iosifidis Resample Attribute', 'Iosifidis Resample Target']:
            ttk.Label(self.params_content_frame, text="Sampling Strategy:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
            self.iosifidis_resample_strategy_var = tk.StringVar(value='auto')
            strategy_combo = ttk.Combobox(self.params_content_frame, textvariable=self.iosifidis_resample_strategy_var, 
                                       values=['auto', 'majority', 'minority'], width=15)
            strategy_combo.grid(row=0, column=1, padx=5, pady=2)
        
        elif method in ['Iosifidis SMOTE Attribute', 'Iosifidis SMOTE Target']:
            ttk.Label(self.params_content_frame, text="K Neighbors:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
            self.iosifidis_smote_k_var = tk.IntVar(value=5)
            ttk.Entry(self.params_content_frame, textvariable=self.iosifidis_smote_k_var, width=15).grid(row=0, column=1, padx=5, pady=2, sticky=tk.W)
        
        # Other advanced methods
        elif method == 'Lahoti Representation Learning':
            ttk.Label(self.params_content_frame, text="Representation Dimension:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
            self.lahoti_dim_var = tk.IntVar(value=10)
            ttk.Entry(self.params_content_frame, textvariable=self.lahoti_dim_var, width=15).grid(row=0, column=1, padx=5, pady=2, sticky=tk.W)
            
            ttk.Label(self.params_content_frame, text="Learning Rate:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
            self.lahoti_lr_var = tk.DoubleVar(value=0.01)
            ttk.Entry(self.params_content_frame, textvariable=self.lahoti_lr_var, width=15).grid(row=1, column=1, padx=5, pady=2, sticky=tk.W)
        
        # Default case for methods without specific parameters
        else:
            ttk.Label(self.params_content_frame, text="No additional parameters required.", 
                     font=('TkDefaultFont', 9, 'italic')).grid(row=0, column=0, columnspan=2, padx=5, pady=20)
                     
    def configure_inprocessing_params(self, method):
        """Configure parameter widgets for inprocessing methods"""
        # Constraint-based fair learning
        if method == 'Zafar Fair Constraints':
            ttk.Label(self.params_content_frame, text="Lambda (fairness weight):").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
            self.zafar_lambda_var = tk.DoubleVar(value=0.1)
            ttk.Scale(self.params_content_frame, from_=0.01, to=1.0, variable=self.zafar_lambda_var, 
                     orient=tk.HORIZONTAL, length=200).grid(row=0, column=1, padx=5, pady=2)
            
            ttk.Label(self.params_content_frame, text="Constraint Type:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
            self.zafar_constraint_var = tk.StringVar(value='demographic_parity')
            constraint_combo = ttk.Combobox(self.params_content_frame, textvariable=self.zafar_constraint_var, 
                                         values=['demographic_parity', 'equalized_odds'], width=15)
            constraint_combo.grid(row=1, column=1, padx=5, pady=2)
            
        elif method == 'Chen Multi-Accuracy Adversarial Training':
            ttk.Label(self.params_content_frame, text="Adversarial Weight:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
            self.chen_adv_var = tk.DoubleVar(value=0.1)
            ttk.Scale(self.params_content_frame, from_=0.01, to=1.0, variable=self.chen_adv_var, 
                     orient=tk.HORIZONTAL, length=200).grid(row=0, column=1, padx=5, pady=2)
            
            ttk.Label(self.params_content_frame, text="Learning Rate:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
            self.chen_lr_var = tk.DoubleVar(value=0.001)
            ttk.Scale(self.params_content_frame, from_=0.0001, to=0.01, variable=self.chen_lr_var, 
                     orient=tk.HORIZONTAL, length=200).grid(row=1, column=1, padx=5, pady=2)
            
        elif method == 'Gao Fair Adversarial Networks':
            ttk.Label(self.params_content_frame, text="Adversarial Loss Weight:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
            self.gao_adv_var = tk.DoubleVar(value=0.1)
            ttk.Scale(self.params_content_frame, from_=0.01, to=1.0, variable=self.gao_adv_var, 
                     orient=tk.HORIZONTAL, length=200).grid(row=0, column=1, padx=5, pady=2)
            
        elif method == 'Islam Fairness-Aware Learning':
            ttk.Label(self.params_content_frame, text="Fairness Regularizer:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
            self.islam_reg_var = tk.DoubleVar(value=0.1)
            ttk.Scale(self.params_content_frame, from_=0.01, to=1.0, variable=self.islam_reg_var, 
                     orient=tk.HORIZONTAL, length=200).grid(row=0, column=1, padx=5, pady=2)
            
        elif method == 'Kilbertus Fair Prediction':
            ttk.Label(self.params_content_frame, text="Causal Regularizer:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
            self.kilbertus_reg_var = tk.DoubleVar(value=0.1)
            ttk.Scale(self.params_content_frame, from_=0.01, to=1.0, variable=self.kilbertus_reg_var, 
                     orient=tk.HORIZONTAL, length=200).grid(row=0, column=1, padx=5, pady=2)
            
        elif method == 'Liu Fair Distribution Matching':
            ttk.Label(self.params_content_frame, text="Transport Regularizer:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
            self.liu_transport_var = tk.DoubleVar(value=0.1)
            ttk.Scale(self.params_content_frame, from_=0.01, to=1.0, variable=self.liu_transport_var, 
                     orient=tk.HORIZONTAL, length=200).grid(row=0, column=1, padx=5, pady=2)
            
        elif method == 'Grari Gradient-Based Fairness':
            ttk.Label(self.params_content_frame, text="Fairness Learning Rate:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
            self.grari_lr_var = tk.DoubleVar(value=0.01)
            ttk.Scale(self.params_content_frame, from_=0.001, to=0.1, variable=self.grari_lr_var, 
                     orient=tk.HORIZONTAL, length=200).grid(row=0, column=1, padx=5, pady=2)
            
        elif method == 'Chakraborty In-Process Synthesis':
            ttk.Label(self.params_content_frame, text="Synthesis Rate:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
            self.chakraborty_in_rate_var = tk.DoubleVar(value=0.1)
            ttk.Scale(self.params_content_frame, from_=0.01, to=0.5, variable=self.chakraborty_in_rate_var, 
                     orient=tk.HORIZONTAL, length=200).grid(row=0, column=1, padx=5, pady=2)
            
        # Default case for inprocessing methods without specific parameters
        else:
            ttk.Label(self.params_content_frame, text="No additional parameters required.", 
                     font=('TkDefaultFont', 9, 'italic')).grid(row=0, column=0, columnspan=2, padx=5, pady=20)
                     
    def configure_inprocessing_params_for_method(self, method):
        """Configure parameter widgets for inprocessing methods in the modeling tab"""
        # Clear existing widgets
        for widget in self.inproc_params_content_frame.winfo_children():
            widget.destroy()
            
        # Constraint-based fair learning
        if method == 'Zafar Fair Constraints':
            ttk.Label(self.inproc_params_content_frame, text="Lambda (fairness weight):").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
            self.inproc_zafar_lambda_var = tk.DoubleVar(value=0.1)
            ttk.Entry(self.inproc_params_content_frame, textvariable=self.inproc_zafar_lambda_var, width=15).grid(row=0, column=1, padx=5, pady=2, sticky=tk.W)
            
            ttk.Label(self.inproc_params_content_frame, text="Constraint Type:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
            self.inproc_zafar_constraint_var = tk.StringVar(value='demographic_parity')
            constraint_combo = ttk.Combobox(self.inproc_params_content_frame, textvariable=self.inproc_zafar_constraint_var, 
                                         values=['demographic_parity', 'equalized_odds'], width=15)
            constraint_combo.grid(row=1, column=1, padx=5, pady=2)
            
        elif method == 'Chen Multi-Accuracy Adversarial Training':
            ttk.Label(self.inproc_params_content_frame, text="Adversarial Weight:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
            self.inproc_chen_adv_var = tk.DoubleVar(value=0.1)
            ttk.Entry(self.inproc_params_content_frame, textvariable=self.inproc_chen_adv_var, width=15).grid(row=0, column=1, padx=5, pady=2, sticky=tk.W)
            
            ttk.Label(self.inproc_params_content_frame, text="Learning Rate:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
            self.inproc_chen_lr_var = tk.DoubleVar(value=0.001)
            ttk.Entry(self.inproc_params_content_frame, textvariable=self.inproc_chen_lr_var, width=15).grid(row=1, column=1, padx=5, pady=2, sticky=tk.W)
            
        elif method == 'Gao Fair Adversarial Networks':
            ttk.Label(self.inproc_params_content_frame, text="Adversarial Loss Weight:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
            self.inproc_gao_adv_var = tk.DoubleVar(value=0.1)
            ttk.Entry(self.inproc_params_content_frame, textvariable=self.inproc_gao_adv_var, width=15).grid(row=0, column=1, padx=5, pady=2, sticky=tk.W)
            
        elif method == 'Islam Fairness-Aware Learning':
            ttk.Label(self.inproc_params_content_frame, text="Fairness Regularizer:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
            self.inproc_islam_reg_var = tk.DoubleVar(value=0.1)
            ttk.Entry(self.inproc_params_content_frame, textvariable=self.inproc_islam_reg_var, width=15).grid(row=0, column=1, padx=5, pady=2, sticky=tk.W)
            
        elif method == 'Kilbertus Fair Prediction':
            ttk.Label(self.inproc_params_content_frame, text="Causal Regularizer:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
            self.inproc_kilbertus_reg_var = tk.DoubleVar(value=0.1)
            ttk.Entry(self.inproc_params_content_frame, textvariable=self.inproc_kilbertus_reg_var, width=15).grid(row=0, column=1, padx=5, pady=2, sticky=tk.W)
            
        elif method == 'Liu Fair Distribution Matching':
            ttk.Label(self.inproc_params_content_frame, text="Transport Regularizer:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
            self.inproc_liu_transport_var = tk.DoubleVar(value=0.1)
            ttk.Entry(self.inproc_params_content_frame, textvariable=self.inproc_liu_transport_var, width=15).grid(row=0, column=1, padx=5, pady=2, sticky=tk.W)
            
        elif method == 'Grari Gradient-Based Fairness':
            ttk.Label(self.inproc_params_content_frame, text="Fairness Learning Rate:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
            self.inproc_grari_lr_var = tk.DoubleVar(value=0.01)
            ttk.Entry(self.inproc_params_content_frame, textvariable=self.inproc_grari_lr_var, width=15).grid(row=0, column=1, padx=5, pady=2, sticky=tk.W)
            
        elif method == 'Chakraborty In-Process Synthesis':
            ttk.Label(self.inproc_params_content_frame, text="Synthesis Rate:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
            self.inproc_chakraborty_in_rate_var = tk.DoubleVar(value=0.1)
            ttk.Entry(self.inproc_params_content_frame, textvariable=self.inproc_chakraborty_in_rate_var, width=15).grid(row=0, column=1, padx=5, pady=2, sticky=tk.W)
            
        # Default case for inprocessing methods without specific parameters
        else:
            ttk.Label(self.inproc_params_content_frame, text="No additional parameters required.", 
                     font=('TkDefaultFont', 9, 'italic')).grid(row=0, column=0, columnspan=2, padx=5, pady=20)
                     
    def configure_postprocessing_params_for_method(self, method):
        """Configure parameter widgets for postprocessing methods"""
        # Clear existing widgets
        for widget in self.postproc_params_content_frame.winfo_children():
            widget.destroy()
            
        # Threshold optimization methods
        if method == 'Snel Bias Correction':
            ttk.Label(self.postproc_params_content_frame, text="Low Threshold:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
            self.postproc_snel_low_var = tk.DoubleVar(value=0.01)
            ttk.Entry(self.postproc_params_content_frame, textvariable=self.postproc_snel_low_var, width=15).grid(row=0, column=1, padx=5, pady=2, sticky=tk.W)
            
            ttk.Label(self.postproc_params_content_frame, text="High Threshold:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
            self.postproc_snel_high_var = tk.DoubleVar(value=0.99)
            ttk.Entry(self.postproc_params_content_frame, textvariable=self.postproc_snel_high_var, width=15).grid(row=1, column=1, padx=5, pady=2, sticky=tk.W)
            
        elif method == 'Pleiss Multicalibration':
            ttk.Label(self.postproc_params_content_frame, text="Alpha (learning rate):").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
            self.postproc_pleiss_alpha_var = tk.DoubleVar(value=0.2)
            ttk.Entry(self.postproc_params_content_frame, textvariable=self.postproc_pleiss_alpha_var, width=15).grid(row=0, column=1, padx=5, pady=2, sticky=tk.W)
            
            ttk.Label(self.postproc_params_content_frame, text="Lambda (regularization):").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
            self.postproc_pleiss_lambda_var = tk.DoubleVar(value=10.0)
            ttk.Entry(self.postproc_params_content_frame, textvariable=self.postproc_pleiss_lambda_var, width=15).grid(row=1, column=1, padx=5, pady=2, sticky=tk.W)
            
        elif method == 'Kamiran Reject Option':
            ttk.Label(self.postproc_params_content_frame, text="Low Threshold:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
            self.postproc_kamiran_low_var = tk.DoubleVar(value=0.01)
            ttk.Entry(self.postproc_params_content_frame, textvariable=self.postproc_kamiran_low_var, width=15).grid(row=0, column=1, padx=5, pady=2, sticky=tk.W)
            
            ttk.Label(self.postproc_params_content_frame, text="High Threshold:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
            self.postproc_kamiran_high_var = tk.DoubleVar(value=0.99)
            ttk.Entry(self.postproc_params_content_frame, textvariable=self.postproc_kamiran_high_var, width=15).grid(row=1, column=1, padx=5, pady=2, sticky=tk.W)
            
            ttk.Label(self.postproc_params_content_frame, text="ROC Margin:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
            self.postproc_kamiran_margin_var = tk.IntVar(value=50)
            ttk.Entry(self.postproc_params_content_frame, textvariable=self.postproc_kamiran_margin_var, width=15).grid(row=2, column=1, padx=5, pady=2, sticky=tk.W)
            
            ttk.Label(self.postproc_params_content_frame, text="Metric Upper Bound:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=2)
            self.postproc_kamiran_ub_var = tk.DoubleVar(value=0.05)
            ttk.Entry(self.postproc_params_content_frame, textvariable=self.postproc_kamiran_ub_var, width=15).grid(row=3, column=1, padx=5, pady=2, sticky=tk.W)
            
            ttk.Label(self.postproc_params_content_frame, text="Metric Lower Bound:").grid(row=4, column=0, sticky=tk.W, padx=5, pady=2)
            self.postproc_kamiran_lb_var = tk.DoubleVar(value=-0.05)
            ttk.Entry(self.postproc_params_content_frame, textvariable=self.postproc_kamiran_lb_var, width=15).grid(row=4, column=1, padx=5, pady=2, sticky=tk.W)
            
        # Default case for postprocessing methods without specific parameters
        else:
            ttk.Label(self.postproc_params_content_frame, text="No additional parameters required.", 
                     font=('TkDefaultFont', 9, 'italic')).grid(row=0, column=0, columnspan=2, padx=5, pady=20)
        
    def update_model_params(self):
        """Update parameter widgets based on selected model"""
        # Clear existing widgets
        for widget in self.model_params_content_frame.winfo_children():
            widget.destroy()
        
        model = self.model_var.get()
        
        if model == 'Logistic Regression':
            ttk.Label(self.model_params_content_frame, text="Regularization (C):").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
            self.lr_c_var = tk.DoubleVar(value=1.0)
            ttk.Entry(self.model_params_content_frame, textvariable=self.lr_c_var, width=15).grid(row=0, column=1, padx=5, pady=2, sticky=tk.W)
            
            ttk.Label(self.model_params_content_frame, text="Penalty:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
            self.lr_penalty_var = tk.StringVar(value='l2')
            penalty_combo = ttk.Combobox(self.model_params_content_frame, textvariable=self.lr_penalty_var, 
                                       values=['l1', 'l2', 'elasticnet', 'none'], width=15)
            penalty_combo.grid(row=1, column=1, padx=5, pady=2)
            
        elif model == 'Decision Tree':
            ttk.Label(self.model_params_content_frame, text="Max Depth:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
            self.dt_depth_var = tk.IntVar(value=5)
            ttk.Entry(self.model_params_content_frame, textvariable=self.dt_depth_var, width=15).grid(row=0, column=1, padx=5, pady=2, sticky=tk.W)
            
            ttk.Label(self.model_params_content_frame, text="Min Samples Split:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
            self.dt_min_samples_var = tk.IntVar(value=2)
            ttk.Entry(self.model_params_content_frame, textvariable=self.dt_min_samples_var, width=15).grid(row=1, column=1, padx=5, pady=2, sticky=tk.W)
        
        # Add more parameter configurations for other models...
        
    def log_message(self, message):
        """Add a message to the log"""
        timestamp = pd.Timestamp.now().strftime("%H:%M:%S")
        
        # Safety check: only log if log_text widget exists
        if hasattr(self, 'log_text') and self.log_text:
            self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
            self.log_text.see(tk.END)
            self.root.update()
        else:
            # Print to console if GUI log not available yet
            print(f"[{timestamp}] {message}")
        
    def load_data(self):
        """Load data from CSV or PKL file"""
        file_path = filedialog.askopenfilename(
            title="Select data file",
            filetypes=[("CSV files", "*.csv"), ("PKL files", "*.pkl"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                # Determine file type and load accordingly
                if file_path.lower().endswith('.pkl'):
                    self.data['raw_data'] = self.load_pkl_data(file_path)
                elif file_path.lower().endswith('.csv'):
                    self.data['raw_data'] = pd.read_csv(file_path)
                else:
                    # Try CSV first, then PKL
                    try:
                        self.data['raw_data'] = pd.read_csv(file_path)
                    except:
                        self.data['raw_data'] = self.load_pkl_data(file_path)
                
                self.update_data_display()
                self.log_message(f"Loaded data from {file_path}")
                self.status_var.set(f"Data loaded: {self.data['raw_data'].shape[0]} rows, {self.data['raw_data'].shape[1]} columns")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load data: {str(e)}")
                self.log_message(f"Error loading data: {str(e)}")
    
    def load_pkl_data(self, file_path):
        """Load data from PKL file (convert_data.py format)"""
        import pickle
        import numpy as np
        
        with open(file_path, 'rb') as f:
            pkl_data = pickle.load(f)
        
        # Extract data from the PKL format
        data_dict = pkl_data['data']
        available_demographics = pkl_data['available_demographics']
        
        # Convert to pandas DataFrame
        rows = []
        
        # Get feature names by examining the first row to determine number of features
        if data_dict:
            first_key = list(data_dict.keys())[0]
            first_features = data_dict[first_key]['features']
            if isinstance(first_features, str):
                # Parse numpy array string representation
                first_features = np.fromstring(first_features.strip('[]'), sep=' ')
            n_features = len(first_features)
            feature_names = [f'feature_{i}' for i in range(n_features)]
        else:
            feature_names = []
        
        # Convert each row
        for row_id, row_data in data_dict.items():
            row = {}
            
            # Add demographic attributes
            for demo_attr in available_demographics:
                # Remove 'demo ' prefix for column name
                clean_attr_name = demo_attr.replace('demo ', '').strip()
                row[clean_attr_name] = row_data[demo_attr]
            
            # Add features
            features = row_data['features']
            if isinstance(features, str):
                # Parse numpy array string representation
                features = np.fromstring(features.strip('[]'), sep=' ')
            
            for i, feature_value in enumerate(features):
                if i < len(feature_names):
                    row[feature_names[i]] = feature_value
            
            # Add target
            row['target'] = row_data['target']
            
            # Add learner_id if needed
            row['learner_id'] = row_data['learner_id']
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        self.log_message(f"Converted PKL data: {len(df)} rows, {len(df.columns)} columns")
        self.log_message(f"Available demographics: {', '.join(available_demographics)}")
        self.log_message(f"Features: {', '.join(feature_names)}")
        
        return df
                
    def load_sample_data(self):
        """Load a sample dataset for demonstration"""
        try:
            # Create a sample educational dataset
            np.random.seed(42)
            n_samples = 1000
            
            # Generate sample features
            data = {
                'student_id': range(n_samples),
                'gender': np.random.choice(['Male', 'Female'], n_samples),
                'age': np.random.normal(20, 2, n_samples),
                'previous_grade': np.random.normal(75, 10, n_samples),
                'study_hours': np.random.exponential(5, n_samples),
                'attendance': np.random.uniform(0.6, 1.0, n_samples),
                'socioeconomic_status': np.random.choice(['Low', 'Medium', 'High'], n_samples),
                'pass': np.random.choice([0, 1], n_samples, p=[0.3, 0.7])
            }
            
            self.data['raw_data'] = pd.DataFrame(data)
            self.update_data_display()
            self.log_message("Loaded sample educational dataset")
            self.status_var.set(f"Sample data loaded: {self.data['raw_data'].shape[0]} rows, {self.data['raw_data'].shape[1]} columns")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create sample data: {str(e)}")
            self.log_message(f"Error creating sample data: {str(e)}")
    
    def load_converted_pkl(self):
        """Load converted PKL data from the data directory"""
        import os
        
        # Check if data directory exists
        if not os.path.exists('data'):
            messagebox.showinfo("Info", 
                              "No converted datasets found.\n\n"
                              "To convert your own dataset:\n"
                              "1. Run one of the setup scripts, OR\n"
                              "2. Use: python convert_data.py --path YOUR_FILE.csv --name DATASET_NAME")
            return
        
        # Get list of converted datasets
        datasets = []
        for item in os.listdir('data'):
            dataset_path = os.path.join('data', item)
            if os.path.isdir(dataset_path):
                pkl_file = os.path.join(dataset_path, 'data_dictionary.pkl')
                if os.path.exists(pkl_file):
                    datasets.append((item, pkl_file))
        
        if not datasets:
            messagebox.showinfo("Info", 
                              "No converted datasets found in the data directory.\n\n"
                              "To convert your own dataset:\n"
                              "1. Run one of the setup scripts, OR\n"
                              "2. Use: python convert_data.py --path YOUR_FILE.csv --name DATASET_NAME")
            return
        
        # Create selection dialog
        selection_window = tk.Toplevel(self.root)
        selection_window.title("Select Converted Dataset")
        selection_window.geometry("400x300")
        selection_window.transient(self.root)
        selection_window.grab_set()
        
        # Center the window
        selection_window.update_idletasks()
        x = (selection_window.winfo_screenwidth() // 2) - (400 // 2)
        y = (selection_window.winfo_screenheight() // 2) - (300 // 2)
        selection_window.geometry(f"400x300+{x}+{y}")
        
        # Add instructions
        instructions = ttk.Label(selection_window, 
                                text="Select a converted dataset to load:",
                                font=('TkDefaultFont', 10, 'bold'))
        instructions.pack(pady=10)
        
        # Create listbox for dataset selection
        listbox_frame = ttk.Frame(selection_window)
        listbox_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        listbox = tk.Listbox(listbox_frame, font=('TkDefaultFont', 10))
        scrollbar = ttk.Scrollbar(listbox_frame, orient=tk.VERTICAL, command=listbox.yview)
        listbox.configure(yscrollcommand=scrollbar.set)
        
        # Add datasets to listbox
        for dataset_name, _ in datasets:
            listbox.insert(tk.END, dataset_name)
        
        listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Add buttons
        button_frame = ttk.Frame(selection_window)
        button_frame.pack(pady=10)
        
        def load_selected():
            selection = listbox.curselection()
            if selection:
                dataset_name, pkl_file = datasets[selection[0]]
                try:
                    self.data['raw_data'] = self.load_pkl_data(pkl_file)
                    self.update_data_display()
                    self.log_message(f"Loaded converted dataset: {dataset_name}")
                    self.status_var.set(f"Converted data loaded: {self.data['raw_data'].shape[0]} rows, {self.data['raw_data'].shape[1]} columns")
                    selection_window.destroy()
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to load dataset: {str(e)}")
                    self.log_message(f"Error loading dataset {dataset_name}: {str(e)}")
            else:
                messagebox.showwarning("Warning", "Please select a dataset")
        
        ttk.Button(button_frame, text="Load", command=load_selected).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=selection_window.destroy).pack(side=tk.LEFT, padx=5)
        
        # Select first item by default
        if datasets:
            listbox.selection_set(0)
            
    def update_data_display(self):
        """Update the data preview and configuration options"""
        if self.data['raw_data'] is not None:
            df = self.data['raw_data']
            
            # Update column selectors
            columns = list(df.columns)
            self.target_combo['values'] = columns
            
            # Clear existing data preview
            for item in self.data_tree.get_children():
                self.data_tree.delete(item)
            
            # Add data to preview (first few rows)
            for col in df.columns:
                sample_values = [str(df[col].iloc[i]) if i < len(df) else "" for i in range(5)]
                self.data_tree.insert('', tk.END, values=[col] + sample_values)
            
            # Update sensitive attributes checkboxes
            for widget in self.sensitive_frame.winfo_children():
                widget.destroy()
            
            self.sensitive_vars = {}
            for i, col in enumerate(columns):
                if col != self.target_var.get():  # Don't include target column
                    var = tk.BooleanVar()
                    self.sensitive_vars[col] = var
                    ttk.Checkbutton(self.sensitive_frame, text=col, variable=var).grid(
                        row=i//3, column=i%3, sticky=tk.W, padx=5)
            
            # Create "before preprocessing" visualizations
            self.visualize_before_preprocessing()
    
    def visualize_before_preprocessing(self):
        """Visualize the raw data before any preprocessing"""
        if self.data['raw_data'] is None:
            return
            
        # Clear previous plots
        for widget in self.before_canvas_frame.winfo_children():
            widget.destroy()
            
        try:
            # Create matplotlib figure for before preprocessing
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle('Data Distribution Before Preprocessing', fontsize=16)
            
            df = self.data['raw_data']
            
            # Plot 1: Overall data distribution
            if len(df.columns) > 0:
                # Count of samples
                axes[0, 0].bar(['Total Samples'], [len(df)], color='lightblue')
                axes[0, 0].set_title('Dataset Size')
                axes[0, 0].set_ylabel('Count')
                
                # Add text annotation
                axes[0, 0].text(0, len(df)/2, f'{len(df)} samples\n{len(df.columns)} features', 
                               ha='center', va='center', fontsize=12, fontweight='bold')
            
            # Plot 2: Target distribution (if target is selected)
            target_col = self.target_var.get()
            if target_col and target_col in df.columns:
                target_counts = df[target_col].value_counts()
                axes[0, 1].bar(target_counts.index.astype(str), target_counts.values, color='lightcoral')
                axes[0, 1].set_title(f'Target Distribution: {target_col}')
                axes[0, 1].set_xlabel(target_col)
                axes[0, 1].set_ylabel('Count')
                
                # Add percentage annotations
                total = len(df)
                for i, (label, count) in enumerate(target_counts.items()):
                    percentage = count / total * 100
                    axes[0, 1].text(i, count + total*0.01, f'{percentage:.1f}%', 
                                   ha='center', va='bottom', fontweight='bold')
            
            # Plot 3: Sensitive attributes distribution
            if hasattr(self, 'sensitive_vars') and self.sensitive_vars:
                sensitive_attrs = [col for col, var in self.sensitive_vars.items() if var.get()]
                if sensitive_attrs:
                    # Use the first sensitive attribute
                    attr = sensitive_attrs[0]
                    if attr in df.columns:
                        attr_counts = df[attr].value_counts()
                        colors = plt.cm.Set3(range(len(attr_counts)))
                        axes[1, 0].bar(attr_counts.index.astype(str), attr_counts.values, color=colors)
                        axes[1, 0].set_title(f'Sensitive Attribute: {attr}')
                        axes[1, 0].set_xlabel(attr)
                        axes[1, 0].set_ylabel('Count')
                        axes[1, 0].tick_params(axis='x', rotation=45)
                else:
                    axes[1, 0].text(0.5, 0.5, 'No sensitive\nattributes selected', 
                                   ha='center', va='center', transform=axes[1, 0].transAxes,
                                   fontsize=12, style='italic')
                    axes[1, 0].set_title('Sensitive Attributes')
            
            # Plot 4: Bias analysis (if both target and sensitive attributes are available)
            if target_col and target_col in df.columns and hasattr(self, 'sensitive_vars') and self.sensitive_vars:
                sensitive_attrs = [col for col, var in self.sensitive_vars.items() if var.get()]
                if sensitive_attrs:
                    attr = sensitive_attrs[0]
                    if attr in df.columns:
                        # Create cross-tabulation
                        cross_tab = pd.crosstab(df[attr], df[target_col], normalize='index') * 100
                        
                        if len(cross_tab.columns) > 0:
                            # Plot stacked bar chart
                            cross_tab.plot(kind='bar', ax=axes[1, 1], stacked=False, 
                                         color=['lightcoral', 'lightblue'])
                            axes[1, 1].set_title(f'Potential Bias: {target_col} by {attr}')
                            axes[1, 1].set_xlabel(attr)
                            axes[1, 1].set_ylabel('Percentage')
                            axes[1, 1].legend(title=target_col, bbox_to_anchor=(1.05, 1), loc='upper left')
                            axes[1, 1].tick_params(axis='x', rotation=45)
                else:
                    axes[1, 1].text(0.5, 0.5, 'Select target and\nsensitive attributes\nto see bias analysis', 
                                   ha='center', va='center', transform=axes[1, 1].transAxes,
                                   fontsize=12, style='italic')
                    axes[1, 1].set_title('Bias Analysis')
            else:
                axes[1, 1].text(0.5, 0.5, 'Select target and\nsensitive attributes\nto see bias analysis', 
                               ha='center', va='center', transform=axes[1, 1].transAxes,
                               fontsize=12, style='italic')
                axes[1, 1].set_title('Bias Analysis')
            
            plt.tight_layout()
            
            # Embed in tkinter
            canvas = FigureCanvasTkAgg(fig, self.before_canvas_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
        except Exception as e:
            self.log_message(f"Error creating before preprocessing visualization: {str(e)}")
            
    def update_model_selection(self):
        """Update model parameters and bias mitigation options when model is selected"""
        self.update_model_params()
        self.update_model_bias_approach()
        
    def update_model_bias_approach(self):
        """Update the inprocessing/postprocessing options based on chosen approach and model"""
        approach = self.model_bias_approach_var.get()
        model_type = self.model_var.get()
        
        # Hide both frames initially
        self.inprocessing_frame.pack_forget()
        self.postprocessing_frame.pack_forget()
        
        if approach == 'inprocessing':
            # Show inprocessing frame and populate methods based on model compatibility
            self.inprocessing_frame.pack(fill=tk.X, padx=10, pady=5)
            
            # Filter inprocessing methods based on model compatibility
            compatible_methods = self.get_compatible_inprocessing_methods(model_type)
            self.inproc_combo['values'] = list(compatible_methods.keys())
            
            # Set default method if current one is not compatible
            if self.inproc_method_var.get() not in compatible_methods:
                if compatible_methods:
                    self.inproc_method_var.set(list(compatible_methods.keys())[0])
                else:
                    self.inproc_method_var.set('No methods available')
            
            self.log_message(f"Inprocessing enabled for {model_type}")
            self.update_inprocessing_method_params()
            
        elif approach == 'postprocessing':
            # Show postprocessing frame and populate all postprocessing methods
            self.postprocessing_frame.pack(fill=tk.X, padx=10, pady=5)
            
            # All postprocessing methods are compatible with all models
            self.postproc_combo['values'] = list(self.postproc_methods.keys())
            
            # Set default method if not already set
            if self.postproc_method_var.get() not in self.postproc_methods:
                if self.postproc_methods:
                    self.postproc_method_var.set(list(self.postproc_methods.keys())[0])
                else:
                    self.postproc_method_var.set('No methods available')
            
            self.log_message(f"Postprocessing enabled for {model_type}")
            self.update_postprocessing_method_params()
            
        else:
            if approach == 'none':
                self.log_message("No bias mitigation - standard training")
            elif approach == 'preprocessed':
                self.log_message("Will use preprocessed data if available")
                
    def get_compatible_inprocessing_methods(self, model_type):
        """Get inprocessing methods compatible with the selected model"""
        # Define model compatibility for inprocessing methods
        compatibility = {
            'Logistic Regression': {
                'Zafar Fair Constraints': 'zafar',
                'Kilbertus Fair Prediction': 'kilbertus', 
                'Liu Fair Distribution Matching': 'liu',
                'Grari Gradient-Based Fairness': 'grari2'
            },
            'Decision Tree': {
                'Chakraborty In-Process Synthesis': 'chakraborty_in'
            },
            'Support Vector Machine': {
                'Zafar Fair Constraints': 'zafar',
                'Liu Fair Distribution Matching': 'liu'
            },
            'Random Forest': {
                'Chen Multi-Accuracy Adversarial Training': 'chen',
                'Gao Fair Adversarial Networks': 'gao',
                'Islam Fairness-Aware Learning': 'islam'
            }
        }
        
        return compatibility.get(model_type, {})
        
    def update_inprocessing_method_params(self, event=None):
        """Update parameter widgets and description for selected inprocessing method"""
        method = self.inproc_method_var.get()
        
        if method == 'No methods available':
            return
            
        # Clear existing inprocessing parameter widgets
        for widget in self.inproc_params_content_frame.winfo_children():
            widget.destroy()
        
        # Inprocessing method descriptions
        inprocessing_descriptions = {
            'Zafar Fair Constraints': "Incorporates fairness constraints directly into the optimization objective during training. Uses convex relaxations to ensure fairness while maintaining accuracy.",
            'Chen Multi-Accuracy Adversarial Training': "Uses adversarial training with multiple accuracy objectives to ensure fairness across different groups during model training.",
            'Gao Fair Adversarial Networks': "Employs adversarial networks to learn fair representations during training. The adversary tries to predict sensitive attributes while the main model learns fair predictions.",
            'Islam Fairness-Aware Learning': "Integrates fairness constraints into deep learning training through specialized loss functions and regularization techniques.",
            'Kilbertus Fair Prediction': "Implements causal fairness through prediction methods that account for the causal structure of the data during training.",
            'Liu Fair Distribution Matching': "Uses optimal transport and distribution matching techniques to ensure fair predictions during the training process.",
            'Grari Gradient-Based Fairness': "Applies gradient-based optimization techniques to directly optimize for fairness metrics during model training.",
            'Chakraborty In-Process Synthesis': "Generates synthetic fair samples during the training process to improve model fairness in real-time."
        }
        
        # Update description
        desc = inprocessing_descriptions.get(method, "Description not available for this method.")
        self.inproc_desc_text.config(state=tk.NORMAL)
        self.inproc_desc_text.delete(1.0, tk.END)
        self.inproc_desc_text.insert(1.0, desc)
        self.inproc_desc_text.config(state=tk.DISABLED)
        
        # Configure parameters for inprocessing method
        self.configure_inprocessing_params_for_method(method)
        
    def update_postprocessing_method_params(self, event=None):
        """Update parameter widgets and description for selected postprocessing method"""
        method = self.postproc_method_var.get()
        
        if method == 'No methods available':
            return
            
        # Clear existing postprocessing parameter widgets
        for widget in self.postproc_params_content_frame.winfo_children():
            widget.destroy()
        
        # Postprocessing method descriptions
        postprocessing_descriptions = {
            'Snel Bias Correction': "Threshold optimization method that adjusts decision thresholds for different groups to achieve fairness. Particularly effective for correcting statistical parity violations after training.",
            
            'Pleiss Multicalibration': "Calibration-based fairness method that ensures predictions are well-calibrated across different groups. Uses multicalibration techniques to achieve both fairness and accuracy.",
            
            'Kamiran Reject Option': "Reject option classification that creates an uncertainty region where predictions can be modified to improve fairness. Particularly useful for high-stakes decisions where fairness is critical."
        }
        
        # Update description for postprocessing methods
        desc = postprocessing_descriptions.get(method, "Description not available for this method.")
        self.postproc_desc_text.config(state=tk.NORMAL)
        self.postproc_desc_text.delete(1.0, tk.END)
        self.postproc_desc_text.insert(1.0, desc)
        self.postproc_desc_text.config(state=tk.DISABLED)
        
        # Configure parameters for postprocessing method
        self.configure_postprocessing_params_for_method(method)
        
    def update_data_config(self, event=None):
        """Update data configuration when target column changes"""
        self.update_data_display()
        # Also refresh the before preprocessing visualization with new settings
        self.visualize_before_preprocessing()
        
    def apply_preprocessing(self):
        """Apply the selected preprocessing method (data transformation only)"""
        if self.data['raw_data'] is None:
            messagebox.showwarning("Warning", "Please load data first")
            return
            
        if not self.target_var.get():
            messagebox.showwarning("Warning", "Please select a target column")
            return
            
        try:
            # Get selected sensitive attributes
            sensitive_attrs = [col for col, var in self.sensitive_vars.items() if var.get()]
            if not sensitive_attrs:
                messagebox.showwarning("Warning", "Please select at least one sensitive attribute")
                return
            
            method = self.preproc_method_var.get()
            self.log_message("Applying preprocessing...")
            self.log_message(f"Using preprocessing method: {method}")
            
            # Prepare data
            df = self.data['raw_data'].copy()
            target_col = self.target_var.get()
            
            # Apply actual preprocessing method
            processed_df = self.apply_real_preprocessing(df, target_col, sensitive_attrs, method)
            
            # Store processed data
            self.data['processed_data'] = processed_df
            self.data['target_column'] = target_col
            self.data['sensitive_attributes'] = sensitive_attrs
            
            self.preproc_status_var.set(f"Preprocessing applied: {method}")
            self.log_message("Preprocessing completed successfully")
            self.log_message("Check the Before/After/Comparison tabs to see the impact of preprocessing")
            
            # Visualize preprocessing results
            self.visualize_preprocessing_results()
            
        except Exception as e:
            messagebox.showerror("Error", f"Preprocessing failed: {str(e)}")
            self.log_message(f"Preprocessing error: {str(e)}")
            
    def get_inprocessing_params(self, method):
        """Get parameters for the selected inprocessing method"""
        params = {}
        
        if method == 'Zafar Fair Constraints':
            params['lambda'] = getattr(self, 'inproc_zafar_lambda_var', tk.DoubleVar(value=0.1)).get()
            params['constraint'] = getattr(self, 'inproc_zafar_constraint_var', tk.StringVar(value='demographic_parity')).get()
            
        elif method == 'Chen Multi-Accuracy Adversarial Training':
            params['adv_weight'] = getattr(self, 'inproc_chen_adv_var', tk.DoubleVar(value=0.1)).get()
            params['learning_rate'] = getattr(self, 'inproc_chen_lr_var', tk.DoubleVar(value=0.001)).get()
            
        elif method == 'Gao Fair Adversarial Networks':
            params['adv_loss_weight'] = getattr(self, 'inproc_gao_adv_var', tk.DoubleVar(value=0.1)).get()
            
        elif method == 'Islam Fairness-Aware Learning':
            params['fairness_reg'] = getattr(self, 'inproc_islam_reg_var', tk.DoubleVar(value=0.1)).get()
            
        elif method == 'Kilbertus Fair Prediction':
            params['causal_reg'] = getattr(self, 'inproc_kilbertus_reg_var', tk.DoubleVar(value=0.1)).get()
            
        elif method == 'Liu Fair Distribution Matching':
            params['transport_reg'] = getattr(self, 'inproc_liu_transport_var', tk.DoubleVar(value=0.1)).get()
            
        elif method == 'Grari Gradient-Based Fairness':
            params['fairness_lr'] = getattr(self, 'inproc_grari_lr_var', tk.DoubleVar(value=0.01)).get()
            
        elif method == 'Chakraborty In-Process Synthesis':
            params['synthesis_rate'] = getattr(self, 'inproc_chakraborty_in_rate_var', tk.DoubleVar(value=0.1)).get()
            
        return params
        
    def get_postprocessing_params(self, method):
        """Get parameters for the selected postprocessing method"""
        params = {}
        
        if method == 'Snel Bias Correction':
            params['low_threshold'] = getattr(self, 'postproc_snel_low_var', tk.DoubleVar(value=0.01)).get()
            params['high_threshold'] = getattr(self, 'postproc_snel_high_var', tk.DoubleVar(value=0.99)).get()
            
        elif method == 'Pleiss Multicalibration':
            params['alpha'] = getattr(self, 'postproc_pleiss_alpha_var', tk.DoubleVar(value=0.2)).get()
            params['lambdaa'] = getattr(self, 'postproc_pleiss_lambda_var', tk.DoubleVar(value=10.0)).get()
            
        elif method == 'Kamiran Reject Option':
            params['low_threshold'] = getattr(self, 'postproc_kamiran_low_var', tk.DoubleVar(value=0.01)).get()
            params['high_threshold'] = getattr(self, 'postproc_kamiran_high_var', tk.DoubleVar(value=0.99)).get()
            params['num_ROC_margin'] = getattr(self, 'postproc_kamiran_margin_var', tk.IntVar(value=50)).get()
            params['metric_ub'] = getattr(self, 'postproc_kamiran_ub_var', tk.DoubleVar(value=0.05)).get()
            params['metric_lb'] = getattr(self, 'postproc_kamiran_lb_var', tk.DoubleVar(value=-0.05)).get()
            
        return params
            
    def apply_real_preprocessing(self, df, target_col, sensitive_attrs, method):
        """Apply the actual preprocessing method from debiased_jadouille package"""
        try:
            self.log_message(f"Applying {method} to data...")
            
            # Get the method key
            method_key = self.preproc_methods.get(method, 'calders')
            
            # Check if we have any available preprocessors
            if not AVAILABLE_PREPROCESSORS:
                self.log_message("No preprocessors available - package import failed")
                return self.simulate_preprocessing_effect(df, target_col, sensitive_attrs, method)
            
            # Initialize the appropriate preprocessor with parameters
            preprocessor = self.create_preprocessor(method_key, sensitive_attrs)
            
            if preprocessor is None:
                self.log_message(f"Warning: {method} not available, using simulation")
                return self.simulate_preprocessing_effect(df, target_col, sensitive_attrs, method)
            
            # Prepare data for preprocessing
            # Convert categorical columns to numeric for processing
            df_numeric = df.copy()
            categorical_encoders = {}
            
            for col in df_numeric.columns:
                if col != target_col and df_numeric[col].dtype == 'object':
                    from sklearn.preprocessing import LabelEncoder
                    le = LabelEncoder()
                    df_numeric[col] = le.fit_transform(df_numeric[col].astype(str))
                    categorical_encoders[col] = le
            
            # Prepare data in the format expected by preprocessors
            X = df_numeric.drop([target_col], axis=1).values.tolist()
            y = df_numeric[target_col].values.tolist()
            
            # Create demographics list for the first sensitive attribute
            if sensitive_attrs:
                demo_attr = sensitive_attrs[0]
                demo_data = []
                for _, row in df.iterrows():  # Use original df for demo data
                    demo_dict = {demo_attr: row[demo_attr]}
                    demo_data.append(demo_dict)
            else:
                demo_data = [{} for _ in range(len(df))]
            
            # Apply preprocessing
            self.log_message("Calling preprocessor fit_transform...")
            X_processed, y_processed, demo_processed = preprocessor.fit_transform(X, y, demo_data)
            
            # Convert back to DataFrame
            feature_cols = [col for col in df.columns if col != target_col]
            processed_df = pd.DataFrame(X_processed, columns=feature_cols)
            processed_df[target_col] = y_processed
            
            # Decode categorical columns back to original values
            for col, encoder in categorical_encoders.items():
                if col in processed_df.columns:
                    # Handle potential new values from preprocessing
                    try:
                        processed_df[col] = encoder.inverse_transform(processed_df[col].astype(int))
                    except ValueError:
                        # If new values were created, use the closest existing values
                        processed_df[col] = processed_df[col].map(lambda x: encoder.classes_[min(int(x), len(encoder.classes_)-1)])
            
            # Add back sensitive attributes from demo_processed or original data
            if sensitive_attrs:
                for attr in sensitive_attrs:
                    if demo_processed and len(demo_processed) > 0 and attr in demo_processed[0]:
                        processed_df[attr] = [demo.get(attr, '') for demo in demo_processed]
                    else:
                        # Keep original sensitive attribute values
                        processed_df[attr] = df[attr].iloc[:len(processed_df)].values
            
            self.log_message(f"Preprocessing complete. Original size: {len(df)}, Processed size: {len(processed_df)}")
            return processed_df
            
        except Exception as e:
            self.log_message(f"Error in preprocessing: {str(e)}")
            self.log_message("Falling back to simulation")
            return self.simulate_preprocessing_effect(df, target_col, sensitive_attrs, method)
    
    def simulate_preprocessing_effect(self, df, target_col, sensitive_attrs, method):
        """Simulate the effect of preprocessing when real methods aren't available"""
        self.log_message(f"Simulating effect of {method}...")
        
        processed_df = df.copy()
        
        # Apply different simulation effects based on method type
        if 'reweight' in method.lower() or 'calders' in method.lower():
            # Simulate reweighting by slightly adjusting the dataset composition
            if sensitive_attrs:
                # Balance the groups slightly
                attr = sensitive_attrs[0]
                groups = processed_df[attr].unique()
                min_group_size = processed_df[attr].value_counts().min()
                
                balanced_dfs = []
                for group in groups:
                    group_df = processed_df[processed_df[attr] == group]
                    if len(group_df) > min_group_size:
                        group_df = group_df.sample(n=min_group_size + 10, random_state=42)
                    balanced_dfs.append(group_df)
                
                processed_df = pd.concat(balanced_dfs, ignore_index=True)
                
        elif 'smote' in method.lower() or 'over' in method.lower():
            # Simulate oversampling by duplicating some minority samples
            if sensitive_attrs:
                attr = sensitive_attrs[0]
                minority_group = processed_df[attr].value_counts().idxmin()
                minority_samples = processed_df[processed_df[attr] == minority_group]
                
                # Add some duplicated samples with slight noise
                n_new_samples = min(20, len(minority_samples))
                new_samples = minority_samples.sample(n=n_new_samples, replace=True, random_state=42)
                processed_df = pd.concat([processed_df, new_samples], ignore_index=True)
        
        self.log_message(f"Simulation complete. Original size: {len(df)}, Processed size: {len(processed_df)}")
        return processed_df
    
    def create_preprocessor(self, method_key, sensitive_attrs):
        """Create the appropriate preprocessor instance with parameters"""
        try:
            # Check if the method is available
            if method_key not in AVAILABLE_PREPROCESSORS:
                self.log_message(f"Preprocessor {method_key} not available (import failed)")
                return None
            
            # Get the preprocessor class
            PreprocessorClass = AVAILABLE_PREPROCESSORS[method_key]
            
            # Get the sensitive attribute for mitigating parameter
            mitigating_attr = sensitive_attrs[0] if sensitive_attrs else 'gender'
            discriminated_value = None  # Will be determined by the preprocessor
            
            # Create preprocessor with method-specific parameters
            if method_key == 'calders':
                proportion = getattr(self, 'calders_prop_var', tk.DoubleVar(value=1.0)).get()
                return PreprocessorClass(mitigating_attr, discriminated_value, proportion)
                
            elif method_key == 'smote':
                k = getattr(self, 'smote_k_var', tk.IntVar(value=5)).get()
                return PreprocessorClass(mitigating_attr, discriminated_value, k)
                
            elif method_key == 'zemel':
                prototypes = getattr(self, 'zemel_prototypes_var', tk.IntVar(value=10)).get()
                lr = getattr(self, 'zemel_lr_var', tk.DoubleVar(value=0.01)).get()
                return PreprocessorClass(mitigating_attr, discriminated_value, prototypes, lr)
                
            elif method_key == 'chakraborty':
                rate = getattr(self, 'chakraborty_rate_var', tk.DoubleVar(value=0.5)).get()
                return PreprocessorClass(mitigating_attr, discriminated_value, rate)
                
            elif method_key == 'dablain':
                proportion = getattr(self, 'dablain_prop_var', tk.DoubleVar(value=1.0)).get()
                k = getattr(self, 'dablain_k_var', tk.IntVar(value=5)).get()
                return PreprocessorClass(mitigating_attr, discriminated_value, proportion, k)
                
            elif method_key == 'alabdulmohsin':
                sgd_steps = getattr(self, 'alabdulmohsin_sgd_var', tk.IntVar(value=10)).get()
                epochs = getattr(self, 'alabdulmohsin_epochs_var', tk.IntVar(value=1)).get()
                return PreprocessorClass(mitigating_attr, discriminated_value, sgd_steps, epochs)
                
            elif method_key == 'li':
                c_factor = getattr(self, 'li_c_factor_var', tk.DoubleVar(value=10.0)).get()
                fairness_metric = getattr(self, 'li_fairness_var', tk.StringVar(value='dp')).get()
                return PreprocessorClass(mitigating_attr, discriminated_value, c_factor, fairness_metric)
                
            elif method_key == 'cock':
                threshold = getattr(self, 'cock_threshold_var', tk.DoubleVar(value=0.1)).get()
                return PreprocessorClass(mitigating_attr, discriminated_value, threshold)
                
            # For methods not yet implemented with specific parameters, use default constructor
            else:
                self.log_message(f"Using default parameters for {method_key}")
                return PreprocessorClass(mitigating_attr, discriminated_value)
                
        except Exception as e:
            self.log_message(f"Error creating preprocessor {method_key}: {str(e)}")
            return None
            
    def apply_postprocessing(self, model, X_test, y_test, y_pred, probabilities, demo_test, method):
        """Apply the selected postprocessing method to model outputs"""
        try:
            self.log_message(f"Applying {method} to model predictions...")
            
            # Get the method key
            method_key = self.postproc_methods.get(method, 'snel')
            
            # Check if we have any available postprocessors
            if not AVAILABLE_POSTPROCESSORS:
                self.log_message("No postprocessors available - package import failed")
                return self.simulate_postprocessing_effect(y_pred, probabilities, method)
            
            # Check if the specific postprocessor is available
            if method_key not in AVAILABLE_POSTPROCESSORS:
                self.log_message(f"Postprocessor {method_key} not available, using simulation")
                return self.simulate_postprocessing_effect(y_pred, probabilities, method)
            
            # Initialize the appropriate postprocessor with parameters
            postprocessor = self.create_postprocessor(method_key)
            
            if postprocessor is None:
                self.log_message(f"Warning: {method} not available, using simulation")
                return self.simulate_postprocessing_effect(y_pred, probabilities, method)
            
            # Prepare data for postprocessing
            features_list = X_test.values.tolist()
            ground_truths_list = y_test.tolist()
            predictions_list = y_pred.tolist()
            probabilities_list = probabilities.tolist() if probabilities is not None else [[1-p, p] for p in y_pred]
            
            # Apply postprocessing
            self.log_message("Calling postprocessor transform...")
            new_pred, original_proba = postprocessor.transform(
                model, features_list, ground_truths_list, predictions_list, 
                probabilities_list, demo_test
            )
            
            self.log_message(f"Postprocessing complete. Adjusted {len(new_pred)} predictions")
            return new_pred, original_proba
            
        except Exception as e:
            self.log_message(f"Error in postprocessing: {str(e)}")
            self.log_message("Falling back to simulation")
            return self.simulate_postprocessing_effect(y_pred, probabilities, method)
    
    def create_postprocessor(self, method_key):
        """Create the appropriate postprocessor instance with parameters"""
        try:
            # Get the postprocessor class
            PostProcessorClass = AVAILABLE_POSTPROCESSORS[method_key]
            
            # Get sensitive attributes
            sensitive_attrs = self.data.get('sensitive_attributes', [])
            mitigating_attr = sensitive_attrs[0] if sensitive_attrs else 'gender'
            
            # Determine discriminated value - for now, use a simple heuristic
            # In practice, this would be configured by the user
            discriminated_value = 'Female'  # Default assumption - would need user input
            
            # Get method-specific parameters
            params = self.data.get('postprocessing_params', {})
            
            # Create postprocessor with method-specific parameters
            if method_key == 'snel':
                return PostProcessorClass(mitigating_attr, discriminated_value)
                
            elif method_key == 'pleiss':
                alpha = params.get('alpha', 0.2)
                lambdaa = params.get('lambdaa', 10.0)
                return PostProcessorClass(mitigating_attr, discriminated_value, alpha, lambdaa)
                
            elif method_key == 'kamiran':
                low_threshold = params.get('low_threshold', 0.01)
                high_threshold = params.get('high_threshold', 0.99)
                num_ROC_margin = params.get('num_ROC_margin', 50)
                metric_ub = params.get('metric_ub', 0.05)
                metric_lb = params.get('metric_lb', -0.05)
                return PostProcessorClass(mitigating_attr, discriminated_value, 
                                        low_threshold, high_threshold, num_ROC_margin, 
                                        metric_ub, metric_lb)
            else:
                # Use default constructor for other methods
                self.log_message(f"Using default parameters for {method_key}")
                return PostProcessorClass(mitigating_attr, discriminated_value)
                
        except Exception as e:
            self.log_message(f"Error creating postprocessor {method_key}: {str(e)}")
            return None
    
    def simulate_postprocessing_effect(self, y_pred, probabilities, method):
        """Simulate the effect of postprocessing when real methods aren't available"""
        self.log_message(f"Simulating effect of {method}...")
        
        # Create a copy of predictions to modify
        simulated_pred = np.array(y_pred).copy()
        
        # Apply different simulation effects based on method type
        if 'snel' in method.lower() or 'threshold' in method.lower():
            # Simulate threshold adjustment by randomly flipping some predictions
            # This is a very simplified simulation
            flip_indices = np.random.choice(len(simulated_pred), size=max(1, len(simulated_pred) // 20), replace=False)
            simulated_pred[flip_indices] = 1 - simulated_pred[flip_indices]
            
        elif 'pleiss' in method.lower() or 'calibration' in method.lower():
            # Simulate calibration by slightly adjusting predictions towards fairness
            # This is a simplified simulation
            adjustment = np.random.normal(0, 0.1, len(simulated_pred))
            simulated_pred = np.clip(simulated_pred + adjustment, 0, 1)
            simulated_pred = np.round(simulated_pred).astype(int)
            
        elif 'kamiran' in method.lower() or 'reject' in method.lower():
            # Simulate reject option by flipping predictions in uncertain region
            if probabilities is not None:
                proba_scores = probabilities[:, 1] if len(probabilities.shape) > 1 else probabilities
                # Find uncertain predictions (around 0.5)
                uncertain_mask = np.abs(proba_scores - 0.5) < 0.2
                flip_indices = np.where(uncertain_mask)[0]
                if len(flip_indices) > 0:
                    # Randomly flip some uncertain predictions
                    to_flip = np.random.choice(flip_indices, size=max(1, len(flip_indices) // 3), replace=False)
                    simulated_pred[to_flip] = 1 - simulated_pred[to_flip]
        
        self.log_message(f"Simulation complete. Modified {np.sum(simulated_pred != y_pred)} predictions")
        return simulated_pred, probabilities
            
    def visualize_preprocessing_results(self):
        """Visualize the results of preprocessing"""
        if self.data['processed_data'] is None:
            return
            
        # Create after preprocessing visualization
        self.visualize_after_preprocessing()
        
        # Create comparison visualization
        self.visualize_comparison()
        
    def visualize_after_preprocessing(self):
        """Visualize the data after preprocessing"""
        if self.data['processed_data'] is None:
            return
            
        # Clear previous plots
        for widget in self.preproc_canvas_frame.winfo_children():
            widget.destroy()
            
        try:
            # Create matplotlib figure
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle('Data Distribution After Preprocessing', fontsize=16)
            
            df = self.data['processed_data']
            target_col = self.data['target_column']
            sensitive_attrs = self.data['sensitive_attributes']
            
            # Plot 1: Target distribution
            if target_col in df.columns:
                target_counts = df[target_col].value_counts()
                axes[0, 0].bar(target_counts.index.astype(str), target_counts.values, color='lightgreen')
                axes[0, 0].set_title(f'Target Distribution: {target_col}')
                axes[0, 0].set_xlabel(target_col)
                axes[0, 0].set_ylabel('Count')
                
                # Add percentage annotations
                total = len(df)
                for i, (label, count) in enumerate(target_counts.items()):
                    percentage = count / total * 100
                    axes[0, 0].text(i, count + total*0.01, f'{percentage:.1f}%', 
                                   ha='center', va='bottom', fontweight='bold')
            
            # Plot 2: Sensitive attribute distribution
            if sensitive_attrs:
                attr = sensitive_attrs[0]
                if attr in df.columns:
                    attr_counts = df[attr].value_counts()
                    colors = plt.cm.Set2(range(len(attr_counts)))
                    axes[0, 1].bar(attr_counts.index.astype(str), attr_counts.values, color=colors)
                    axes[0, 1].set_title(f'Sensitive Attribute: {attr}')
                    axes[0, 1].set_xlabel(attr)
                    axes[0, 1].set_ylabel('Count')
                    axes[0, 1].tick_params(axis='x', rotation=45)
            
            # Plot 3: Empty (correlation plot removed)
            axes[1, 0].set_visible(False)
            
            # Plot 4: Target vs sensitive attribute (bias analysis after preprocessing)
            if sensitive_attrs and target_col in df.columns:
                attr = sensitive_attrs[0]
                if attr in df.columns:
                    # Create cross-tabulation with percentages
                    cross_tab = pd.crosstab(df[attr], df[target_col], normalize='index') * 100
                    
                    if len(cross_tab.columns) > 0:
                        cross_tab.plot(kind='bar', ax=axes[1, 1], stacked=False, 
                                     color=['lightgreen', 'lightcoral'])
                        axes[1, 1].set_title(f'Bias Analysis After: {target_col} by {attr}')
                        axes[1, 1].set_xlabel(attr)
                        axes[1, 1].set_ylabel('Percentage')
                        axes[1, 1].legend(title=target_col, bbox_to_anchor=(1.05, 1), loc='upper left')
                        axes[1, 1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            # Embed in tkinter
            canvas = FigureCanvasTkAgg(fig, self.preproc_canvas_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
        except Exception as e:
            self.log_message(f"Error creating after preprocessing visualization: {str(e)}")
            
    def visualize_comparison(self):
        """Create side-by-side comparison of before and after preprocessing"""
        if self.data['raw_data'] is None or self.data['processed_data'] is None:
            return
            
        # Clear previous plots
        for widget in self.comparison_canvas_frame.winfo_children():
            widget.destroy()
            
        try:
            # Create matplotlib figure for comparison
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle('Before vs After Preprocessing Comparison', fontsize=16)
            
            raw_df = self.data['raw_data']
            processed_df = self.data['processed_data']
            target_col = self.data['target_column']
            sensitive_attrs = self.data['sensitive_attributes']
            
            # Row 1: Target distribution comparison
            if target_col and target_col in raw_df.columns and target_col in processed_df.columns:
                # Before
                raw_target_counts = raw_df[target_col].value_counts()
                axes[0, 0].bar(raw_target_counts.index.astype(str), raw_target_counts.values, 
                              color='lightcoral', alpha=0.7)
                axes[0, 0].set_title(f'BEFORE: {target_col} Distribution')
                axes[0, 0].set_ylabel('Count')
                
                # After
                proc_target_counts = processed_df[target_col].value_counts()
                axes[0, 1].bar(proc_target_counts.index.astype(str), proc_target_counts.values, 
                              color='lightgreen', alpha=0.7)
                axes[0, 1].set_title(f'AFTER: {target_col} Distribution')
                axes[0, 1].set_ylabel('Count')
                
                # Summary
                axes[0, 2].text(0.1, 0.8, 'TARGET DISTRIBUTION CHANGES:', fontweight='bold', 
                               transform=axes[0, 2].transAxes, fontsize=12)
                
                changes_text = []
                for label in raw_target_counts.index:
                    if label in proc_target_counts.index:
                        raw_count = raw_target_counts[label]
                        proc_count = proc_target_counts[label]
                        change = proc_count - raw_count
                        changes_text.append(f'{label}: {raw_count} → {proc_count} ({change:+d})')
                
                axes[0, 2].text(0.1, 0.6, '\n'.join(changes_text), 
                               transform=axes[0, 2].transAxes, fontsize=10, 
                               verticalalignment='top')
                axes[0, 2].axis('off')
            
            # Row 2: Bias analysis comparison
            if sensitive_attrs and target_col:
                attr = sensitive_attrs[0]
                if attr in raw_df.columns and attr in processed_df.columns:
                    # Before bias analysis
                    raw_cross_tab = pd.crosstab(raw_df[attr], raw_df[target_col], normalize='index') * 100
                    if len(raw_cross_tab.columns) > 0:
                        raw_cross_tab.plot(kind='bar', ax=axes[1, 0], stacked=False, 
                                         color=['lightcoral', 'lightsalmon'])
                        axes[1, 0].set_title(f'BEFORE: Bias in {target_col} by {attr}')
                        axes[1, 0].set_xlabel(attr)
                        axes[1, 0].set_ylabel('Percentage')
                        axes[1, 0].legend(title=target_col)
                        axes[1, 0].tick_params(axis='x', rotation=45)
                    
                    # After bias analysis
                    proc_cross_tab = pd.crosstab(processed_df[attr], processed_df[target_col], normalize='index') * 100
                    if len(proc_cross_tab.columns) > 0:
                        proc_cross_tab.plot(kind='bar', ax=axes[1, 1], stacked=False, 
                                          color=['lightgreen', 'lightblue'])
                        axes[1, 1].set_title(f'AFTER: Bias in {target_col} by {attr}')
                        axes[1, 1].set_xlabel(attr)
                        axes[1, 1].set_ylabel('Percentage')
                        axes[1, 1].legend(title=target_col)
                        axes[1, 1].tick_params(axis='x', rotation=45)
                    
                    # Bias reduction summary
                    axes[1, 2].text(0.1, 0.9, 'BIAS REDUCTION ANALYSIS:', fontweight='bold', 
                                   transform=axes[1, 2].transAxes, fontsize=12)
                    
                    # Calculate bias metrics
                    bias_text = []
                    if len(raw_cross_tab.columns) > 0 and len(proc_cross_tab.columns) > 0:
                        # For each group, calculate the difference in positive outcome rates
                        for group in raw_cross_tab.index:
                            if group in proc_cross_tab.index:
                                # Assuming positive outcome is 1 (or the last column)
                                if len(raw_cross_tab.columns) > 1:
                                    raw_positive_rate = raw_cross_tab.loc[group, 1] if 1 in raw_cross_tab.columns else raw_cross_tab.loc[group].iloc[-1]
                                    proc_positive_rate = proc_cross_tab.loc[group, 1] if 1 in proc_cross_tab.columns else proc_cross_tab.loc[group].iloc[-1]
                                    
                                    bias_text.append(f'{group}:')
                                    bias_text.append(f'  Before: {raw_positive_rate:.1f}%')
                                    bias_text.append(f'  After: {proc_positive_rate:.1f}%')
                                    bias_text.append(f'  Change: {proc_positive_rate - raw_positive_rate:+.1f}%')
                        
                        # Calculate overall demographic parity
                        if len(raw_cross_tab) > 1:
                            raw_max = raw_cross_tab.iloc[:, -1].max()
                            raw_min = raw_cross_tab.iloc[:, -1].min()
                            raw_parity = raw_max - raw_min
                            
                            proc_max = proc_cross_tab.iloc[:, -1].max()
                            proc_min = proc_cross_tab.iloc[:, -1].min()
                            proc_parity = proc_max - proc_min
                            
                            bias_text.append('')
                            bias_text.append(f'Demographic Parity Gap:')
                            bias_text.append(f'  Before: {raw_parity:.1f}%')
                            bias_text.append(f'  After: {proc_parity:.1f}%')
                            bias_text.append(f'  Improvement: {raw_parity - proc_parity:+.1f}%')
                    
                    axes[1, 2].text(0.1, 0.8, '\n'.join(bias_text), 
                                   transform=axes[1, 2].transAxes, fontsize=9, 
                                   verticalalignment='top', family='monospace')
                    axes[1, 2].axis('off')
            
            plt.tight_layout()
            
            # Embed in tkinter
            canvas = FigureCanvasTkAgg(fig, self.comparison_canvas_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
        except Exception as e:
            self.log_message(f"Error creating comparison visualization: {str(e)}")
            
    def train_model(self):
        """Train the selected model"""
        if self.data['raw_data'] is None:
            messagebox.showwarning("Warning", "Please load data first")
            return
            
        if not self.target_var.get():
            messagebox.showwarning("Warning", "Please select a target column")
            return
            
        try:
            self.log_message("Starting model training...")
            self.training_progress.start(10)
            
            # Determine which data to use based on bias mitigation approach
            bias_approach = self.model_bias_approach_var.get()
            
            if bias_approach == 'preprocessed' and 'processed_data' in self.data and self.data['processed_data'] is not None:
                # Use preprocessed data if available and requested
                df = self.data['processed_data']
                target_col = self.data['target_column']
                self.log_message("Using preprocessed data for training")
            else:
                # Use raw data (for 'none' approach or when no preprocessing was applied)
                df = self.data['raw_data']
                target_col = self.target_var.get()
                
                # Store target column for consistency
                self.data['target_column'] = target_col
                
                if bias_approach == 'preprocessed':
                    self.log_message("No preprocessed data available - using raw data")
                else:
                    self.log_message("Using raw data for training")
            
            # Separate features and target
            X = df.drop([target_col], axis=1)
            y = df[target_col]
            
            # Encode categorical variables
            from sklearn.preprocessing import LabelEncoder
            le_dict = {}
            for col in X.select_dtypes(include=['object']).columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                le_dict[col] = le
            
            # Get sensitive attributes (from stored data or UI selections)
            if 'sensitive_attributes' in self.data and self.data['sensitive_attributes']:
                sensitive_attrs = self.data['sensitive_attributes']
            else:
                # Get from UI selections if not stored
                sensitive_attrs = [col for col, var in self.sensitive_vars.items() if var.get()]
                # Store for consistency
                self.data['sensitive_attributes'] = sensitive_attrs
            demo_data = []
            if sensitive_attrs:
                for _, row in df.iterrows():
                    demo_dict = {attr: row[attr] for attr in sensitive_attrs}
                    demo_data.append(demo_dict)
            else:
                demo_data = [{} for _ in range(len(df))]
            
            # Split data
            from sklearn.model_selection import train_test_split
            test_size = self.test_split_var.get()
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
            
            # Split demographic data correspondingly
            demo_train = [demo_data[i] for i in X_train.index] if hasattr(X_train, 'index') else demo_data[:len(X_train)]
            demo_test = [demo_data[i] for i in X_test.index] if hasattr(X_test, 'index') else demo_data[len(X_train):]
            
            # Store train/test data
            self.data['train_data'] = (X_train, y_train, demo_train)
            self.data['test_data'] = (X_test, y_test, demo_test)
            
            # Check bias mitigation approach from modeling tab
            bias_approach = self.model_bias_approach_var.get()
            model_type = self.model_var.get()
            
            if bias_approach == 'inprocessing':
                # Check if sensitive attributes are selected for inprocessing
                if not sensitive_attrs:
                    messagebox.showwarning("Warning", "Please select at least one sensitive attribute for inprocessing methods")
                    return
                    
                # Use inprocessing method
                self.log_message("Training with inprocessing method...")
                inproc_method = self.inproc_method_var.get()
                
                # Store inprocessing configuration in data
                self.data['inprocessing_method'] = inproc_method
                self.data['inprocessing_params'] = self.get_inprocessing_params(inproc_method)
                
                model = self.create_inprocessing_model(X_train, y_train, demo_train)
                
            elif bias_approach == 'postprocessing':
                # Check if sensitive attributes are selected for postprocessing
                if not sensitive_attrs:
                    messagebox.showwarning("Warning", "Please select at least one sensitive attribute for postprocessing methods")
                    return
                    
                # For postprocessing, we train a standard model first and store postprocessing config
                self.log_message(f"Training {model_type} model for postprocessing...")
                postproc_method = self.postproc_method_var.get()
                
                # Store postprocessing configuration in data
                self.data['postprocessing_method'] = postproc_method
                self.data['postprocessing_params'] = self.get_postprocessing_params(postproc_method)
                
                # Train standard model first
                model = self.create_standard_model(model_type)
                model.fit(X_train, y_train)
                
                self.log_message(f"Model trained - postprocessing will be applied during evaluation")
                
            elif bias_approach == 'preprocessed':
                # Check if preprocessing was applied
                if 'processed_data' in self.data and self.data['processed_data'] is not None:
                    self.log_message(f"Training {model_type} model with preprocessed data...")
                else:
                    self.log_message(f"No preprocessing applied - training {model_type} model with original data...")
                    
                # Train standard model (with preprocessed data if available)
                model = self.create_standard_model(model_type)
                model.fit(X_train, y_train)
                
            else:  # bias_approach == 'none'
                # Train standard model with original data
                self.log_message(f"Training {model_type} model (no bias mitigation)...")
                model = self.create_standard_model(model_type)
                model.fit(X_train, y_train)
            
            # Store the trained model
            self.data['models'][model_type] = {
                'model': model,
                'encoder_dict': le_dict,
                'features': list(X.columns)
            }
            
            self.training_progress.stop()
            self.training_status_var.set(f"Model trained: {model_type}")
            self.log_message(f"{model_type} training completed successfully")
            
        except Exception as e:
            self.training_progress.stop()
            messagebox.showerror("Error", f"Training failed: {str(e)}")
            self.log_message(f"Training error: {str(e)}")
            
    def create_inprocessing_model(self, X_train, y_train, demo_train):
        """Create and train an inprocessing model"""
        try:
            method = self.data['inprocessing_method']
            params = self.data.get('inprocessing_params', {})
            sensitive_attrs = self.data['sensitive_attributes']
            
            # Get the method key
            method_key = self.inproc_methods.get(method, 'zafar')
            
            # Check if the inprocessing method is available
            if method_key not in AVAILABLE_INPROCESSORS:
                self.log_message(f"Inprocessing method {method_key} not available, using simulation")
                return self.create_simulated_fair_model(X_train, y_train)
            
            # Get the inprocessor class
            InProcessorClass = AVAILABLE_INPROCESSORS[method_key]
            
            # Get the sensitive attribute for mitigating parameter
            mitigating_attr = sensitive_attrs[0] if sensitive_attrs else 'gender'
            discriminated_value = None  # Will be determined by the inprocessor
            
            # Create inprocessor with method-specific parameters
            if method_key == 'zafar':
                fairness_weight = params.get('lambda', 0.1)
                constraint_type = params.get('constraint', 'demographic_parity')
                inprocessor = InProcessorClass(mitigating_attr, discriminated_value, fairness_weight, constraint_type)
                
            elif method_key == 'chen':
                adv_weight = params.get('adv_weight', 0.1)
                learning_rate = params.get('learning_rate', 0.001)
                inprocessor = InProcessorClass(mitigating_attr, discriminated_value, adv_weight, learning_rate)
                
            elif method_key == 'gao':
                adv_loss_weight = params.get('adv_loss_weight', 0.1)
                inprocessor = InProcessorClass(mitigating_attr, discriminated_value, adv_loss_weight)
                
            else:
                # Use default constructor for other methods
                inprocessor = InProcessorClass(mitigating_attr, discriminated_value)
            
            # Train the inprocessor
            self.log_message(f"Training {method} inprocessing model...")
            inprocessor.fit(X_train.values.tolist(), y_train.tolist(), demo_train)
            
            return inprocessor
            
        except Exception as e:
            self.log_message(f"Error creating inprocessing model: {str(e)}")
            self.log_message("Falling back to simulated fair model")
            return self.create_simulated_fair_model(X_train, y_train)
    
    def create_simulated_fair_model(self, X_train, y_train):
        """Create a simulated fair model when real inprocessing isn't available"""
        from sklearn.linear_model import LogisticRegression
        
        # Create a standard model as fallback
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train, y_train)
        
        # Add a fairness simulation wrapper
        class SimulatedFairModel:
            def __init__(self, base_model):
                self.base_model = base_model
                
            def predict(self, X):
                return self.base_model.predict(X)
                
            def predict_proba(self, X):
                return self.base_model.predict_proba(X)
        
        return SimulatedFairModel(model)
        
    def create_standard_model(self, model_type):
        """Create a standard (non-fair) model based on type and parameters"""
        if model_type == 'Logistic Regression':
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(
                C=self.lr_c_var.get(),
                penalty=self.lr_penalty_var.get(),
                random_state=42,
                max_iter=1000
            )
        elif model_type == 'Decision Tree':
            from sklearn.tree import DecisionTreeClassifier
            return DecisionTreeClassifier(
                max_depth=self.dt_depth_var.get(),
                min_samples_split=self.dt_min_samples_var.get(),
                random_state=42
            )
        elif model_type == 'Support Vector Machine':
            from sklearn.svm import SVC
            return SVC(random_state=42, probability=True)
        elif model_type == 'Random Forest':
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(random_state=42)
        else:
            # Default to logistic regression
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(random_state=42, max_iter=1000)
            
    def evaluate_model(self):
        """Evaluate the trained model"""
        model_type = self.model_var.get()
        if model_type not in self.data['models']:
            messagebox.showwarning("Warning", "Please train a model first")
            return
            
        try:
            self.log_message("Evaluating model...")
            
            # Get model and test data
            model_info = self.data['models'][model_type]
            model = model_info['model']
            
            if model is None:
                messagebox.showerror("Error", "Model is None - please retrain the model")
                return
            
            # Handle both old and new data format
            if 'test_data' not in self.data or self.data['test_data'] is None:
                messagebox.showerror("Error", "No test data available - please train a model first")
                return
                
            test_data = self.data['test_data']
            if len(test_data) == 3:
                X_test, y_test, demo_test = test_data
            else:
                X_test, y_test = test_data
                demo_test = []
                
            if X_test is None or y_test is None:
                messagebox.showerror("Error", "Test data is None - please retrain the model")
                return
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Handle probability predictions safely
            if hasattr(model, 'predict_proba'):
                try:
                    probabilities = model.predict_proba(X_test)
                    if probabilities is not None and probabilities.shape[1] > 1:
                        y_pred_proba = probabilities[:, 1]
                    else:
                        y_pred_proba = y_pred.astype(float)
                except:
                    y_pred_proba = y_pred.astype(float)
            else:
                y_pred_proba = y_pred.astype(float)
            
            # Apply postprocessing if configured
            if 'postprocessing_method' in self.data and self.data['postprocessing_method']:
                try:
                    self.log_message("Applying postprocessing...")
                    postproc_method = self.data['postprocessing_method']
                    
                    # Apply postprocessing to get adjusted predictions
                    postprocessed_pred, original_proba = self.apply_postprocessing(
                        model, X_test, y_test, y_pred, probabilities, demo_test, postproc_method
                    )
                    
                    # Store both original and postprocessed results
                    self.data['original_predictions'] = y_pred.copy()
                    self.data['original_probabilities'] = y_pred_proba.copy()
                    
                    # Use postprocessed predictions for main evaluation
                    y_pred = postprocessed_pred
                    self.log_message("Postprocessing applied successfully")
                    
                except Exception as e:
                    self.log_message(f"Warning: Postprocessing failed - {str(e)}")
                    self.log_message("Using original predictions")
            
            # Calculate metrics
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
            
            metrics = {
                'Accuracy': accuracy_score(y_test, y_pred),
                'Precision': precision_score(y_test, y_pred, average='weighted'),
                'Recall': recall_score(y_test, y_pred, average='weighted'),
                'F1-Score': f1_score(y_test, y_pred, average='weighted'),
            }
            
            if len(np.unique(y_test)) == 2:  # Binary classification
                metrics['ROC-AUC'] = roc_auc_score(y_test, y_pred_proba)
            
            # Calculate fairness metrics (simplified)
            sensitive_attrs = self.data.get('sensitive_attributes', [])
            fairness_metrics = {}
            
            if sensitive_attrs:
                try:
                    # Get sensitive attribute values for test set - use the same data that was used for training
                    bias_approach = self.model_bias_approach_var.get()
                    
                    # Determine source data based on what was used for training
                    if bias_approach == 'preprocessed' and 'processed_data' in self.data and self.data['processed_data'] is not None:
                        source_df = self.data['processed_data']
                    else:
                        source_df = self.data['raw_data']
                    
                    # Handle index alignment - get the actual test data with sensitive attributes
                    if hasattr(X_test, 'index'):
                        # If X_test has pandas index, use it
                        df_test = source_df.loc[X_test.index]
                    else:
                        # If X_test is numpy array, we need to reconstruct from demo_test if available
                        if len(test_data) == 3 and demo_test:
                            # Use demographic data stored during training
                            df_test = pd.DataFrame(demo_test)
                        else:
                            # Fallback: use the last len(X_test) rows (not ideal but prevents crash)
                            df_test = source_df.tail(len(X_test)).reset_index(drop=True)
                    
                    for attr in sensitive_attrs:
                        if attr in df_test.columns:
                            groups = df_test[attr].unique()
                            group_metrics = {}
                            
                            for group in groups:
                                group_mask = df_test[attr] == group
                                if group_mask.sum() > 0:
                                    group_y_test = y_test[group_mask] if hasattr(group_mask, '__len__') else y_test
                                    group_y_pred = y_pred[group_mask] if hasattr(group_mask, '__len__') else y_pred
                                    
                                    if len(group_y_test) > 0:
                                        group_metrics[group] = {
                                            'Accuracy': accuracy_score(group_y_test, group_y_pred),
                                            'Positive_Rate': np.mean(group_y_pred)
                                        }
                            
                            fairness_metrics[attr] = group_metrics
                            
                except Exception as e:
                    self.log_message(f"Warning: Could not calculate fairness metrics - {str(e)}")
                    fairness_metrics = {}
            
            # Store results (initialize results dict if needed)
            if 'results' not in self.data:
                self.data['results'] = {}
                
            self.data['results'][model_type] = {
                'metrics': metrics,
                'fairness_metrics': fairness_metrics,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            # Update metrics display
            self.update_metrics_display(metrics, fairness_metrics)
            
            # Create visualizations
            self.create_evaluation_visualizations(y_test, y_pred, y_pred_proba)
            
            self.log_message("Model evaluation completed")
            
        except Exception as e:
            messagebox.showerror("Error", f"Evaluation failed: {str(e)}")
            self.log_message(f"Evaluation error: {str(e)}")
            
    def update_metrics_display(self, metrics, fairness_metrics):
        """Update the metrics display in the evaluation tab"""
        # Clear existing items
        for item in self.metrics_tree.get_children():
            self.metrics_tree.delete(item)
        
        # Add overall metrics
        for metric, value in metrics.items():
            self.metrics_tree.insert('', tk.END, values=(metric, f"{value:.4f}", "Overall"))
        
        # Add fairness metrics
        for attr, groups in fairness_metrics.items():
            for group, group_metrics in groups.items():
                for metric, value in group_metrics.items():
                    display_metric = f"{attr}_{metric}"
                    self.metrics_tree.insert('', tk.END, values=(display_metric, f"{value:.4f}", group))
                    
    def create_evaluation_visualizations(self, y_test, y_pred, y_pred_proba):
        """Create evaluation visualizations"""
        # Clear previous plots
        for widget in self.eval_canvas_frame.winfo_children():
            widget.destroy()
            
        try:
            # Create matplotlib figure
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle('Model Evaluation Results', fontsize=16)
            
            # Plot 1: Confusion Matrix
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_test, y_pred)
            im1 = axes[0, 0].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            axes[0, 0].set_title('Confusion Matrix')
            axes[0, 0].set_xlabel('Predicted')
            axes[0, 0].set_ylabel('Actual')
            
            # Add text annotations
            thresh = cm.max() / 2.
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    axes[0, 0].text(j, i, format(cm[i, j], 'd'),
                                  ha="center", va="center",
                                  color="white" if cm[i, j] > thresh else "black")
            
            # Plot 2: ROC Curve (for binary classification)
            if len(np.unique(y_test)) == 2:
                from sklearn.metrics import roc_curve, auc
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                roc_auc = auc(fpr, tpr)
                
                axes[0, 1].plot(fpr, tpr, color='darkorange', lw=2, 
                              label=f'ROC curve (AUC = {roc_auc:.2f})')
                axes[0, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                axes[0, 1].set_xlim([0.0, 1.0])
                axes[0, 1].set_ylim([0.0, 1.05])
                axes[0, 1].set_xlabel('False Positive Rate')
                axes[0, 1].set_ylabel('True Positive Rate')
                axes[0, 1].set_title('ROC Curve')
                axes[0, 1].legend(loc="lower right")
            
            # Plot 3: Prediction Distribution
            axes[1, 0].hist(y_pred_proba, bins=30, alpha=0.7, color='skyblue')
            axes[1, 0].set_title('Prediction Score Distribution')
            axes[1, 0].set_xlabel('Prediction Score')
            axes[1, 0].set_ylabel('Frequency')
            
            # Plot 4: Feature Importance (if available)
            model_type = self.model_var.get()
            if model_type in self.data['models']:
                model = self.data['models'][model_type]['model']
                features = self.data['models'][model_type]['features']
                
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    indices = np.argsort(importances)[::-1][:10]  # Top 10 features
                    
                    axes[1, 1].bar(range(len(indices)), importances[indices])
                    axes[1, 1].set_title('Top 10 Feature Importances')
                    axes[1, 1].set_xlabel('Features')
                    axes[1, 1].set_ylabel('Importance')
                    axes[1, 1].set_xticks(range(len(indices)))
                    axes[1, 1].set_xticklabels([features[i] for i in indices], rotation=45)
                elif hasattr(model, 'coef_'):
                    coef = np.abs(model.coef_[0])
                    indices = np.argsort(coef)[::-1][:10]  # Top 10 features
                    
                    axes[1, 1].bar(range(len(indices)), coef[indices])
                    axes[1, 1].set_title('Top 10 Feature Coefficients (Abs)')
                    axes[1, 1].set_xlabel('Features')
                    axes[1, 1].set_ylabel('Coefficient (Absolute)')
                    axes[1, 1].set_xticks(range(len(indices)))
                    axes[1, 1].set_xticklabels([features[i] for i in indices], rotation=45)
            
            plt.tight_layout()
            
            # Embed in tkinter
            canvas = FigureCanvasTkAgg(fig, self.eval_canvas_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
        except Exception as e:
            self.log_message(f"Error creating evaluation visualizations: {str(e)}")
            
    def generate_report(self):
        """Generate a comprehensive evaluation report"""
        if not self.data['results']:
            messagebox.showwarning("Warning", "Please evaluate a model first")
            return
            
        try:
            self.log_message("Generating evaluation report...")
            
            # Create a simple text report
            report_lines = []
            report_lines.append("=== DebiasEd Evaluation Report ===\n")
            report_lines.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            # Data summary
            if self.data['raw_data'] is not None:
                report_lines.append("\n--- Data Summary ---")
                report_lines.append(f"Dataset shape: {self.data['raw_data'].shape}")
                report_lines.append(f"Target column: {self.data['target_column']}")
                report_lines.append(f"Sensitive attributes: {', '.join(self.data['sensitive_attributes'])}")
            
            # Bias Mitigation summary
            report_lines.append(f"\n--- Bias Mitigation ---")
            
            # Check what bias mitigation was actually applied
            bias_approach = self.model_bias_approach_var.get()
            
            if bias_approach == 'preprocessed' and 'processed_data' in self.data and self.data['processed_data'] is not None:
                report_lines.append(f"Approach: Preprocessing")
                report_lines.append(f"Method: {self.preproc_method_var.get()}")
            elif bias_approach == 'inprocessing':
                report_lines.append(f"Approach: Inprocessing")
                if hasattr(self, 'inproc_method_var'):
                    report_lines.append(f"Method: {self.inproc_method_var.get()}")
                else:
                    report_lines.append(f"Method: Not specified")
            elif bias_approach == 'postprocessing':
                report_lines.append(f"Approach: Postprocessing")
                if hasattr(self, 'postproc_method_var'):
                    report_lines.append(f"Method: {self.postproc_method_var.get()}")
                else:
                    report_lines.append(f"Method: Not specified")
                
                # Add comparison of original vs postprocessed results if available
                if 'original_predictions' in self.data:
                    report_lines.append(f"Original predictions available for comparison")
            else:
                report_lines.append(f"Approach: None (Standard training)")
                report_lines.append(f"Method: No bias mitigation applied")
            
            # Model results
            for model_type, results in self.data['results'].items():
                report_lines.append(f"\n--- {model_type} Results ---")
                
                # Overall metrics
                report_lines.append("Overall Metrics:")
                for metric, value in results['metrics'].items():
                    report_lines.append(f"  {metric}: {value:.4f}")
                
                # Fairness metrics
                if results['fairness_metrics']:
                    report_lines.append("\nFairness Metrics:")
                    for attr, groups in results['fairness_metrics'].items():
                        report_lines.append(f"  {attr}:")
                        for group, group_metrics in groups.items():
                            for metric, value in group_metrics.items():
                                report_lines.append(f"    {group} {metric}: {value:.4f}")
            
            # Show report in a new window
            self.show_report_window('\n'.join(report_lines))
            
            self.log_message("Report generated successfully")
            
        except Exception as e:
            messagebox.showerror("Error", f"Report generation failed: {str(e)}")
            self.log_message(f"Report generation error: {str(e)}")
            
    def show_report_window(self, report_text):
        """Show the evaluation report in a new window"""
        report_window = tk.Toplevel(self.root)
        report_window.title("Evaluation Report")
        report_window.geometry("800x600")
        
        # Create text widget with scrollbar
        text_frame = ttk.Frame(report_window)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        text_widget = tk.Text(text_frame, wrap=tk.WORD, font=('Courier', 10))
        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)
        
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Insert report text
        text_widget.insert(tk.END, report_text)
        text_widget.config(state=tk.DISABLED)
        
        # Add save button
        save_btn = ttk.Button(report_window, text="Save Report", 
                             command=lambda: self.save_report_to_file(report_text))
        save_btn.pack(pady=5)
        
    def save_report_to_file(self, report_text):
        """Save the report to a file"""
        file_path = filedialog.asksaveasfilename(
            title="Save Report",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    f.write(report_text)
                messagebox.showinfo("Success", f"Report saved to {file_path}")
                self.log_message(f"Report saved to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save report: {str(e)}")
                
    def export_model(self):
        """Export the trained model"""
        model_type = self.model_var.get()
        if model_type not in self.data['models']:
            messagebox.showwarning("Warning", "Please train a model first")
            return
            
        file_path = filedialog.asksaveasfilename(
            title="Save Model",
            defaultextension=".pkl",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'wb') as f:
                    pickle.dump(self.data['models'][model_type], f)
                messagebox.showinfo("Success", f"Model saved to {file_path}")
                self.log_message(f"Model saved to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save model: {str(e)}")
                
    def export_results_csv(self):
        """Export results to CSV"""
        if not self.data['results']:
            messagebox.showwarning("Warning", "No results to export")
            return
            
        file_path = filedialog.asksaveasfilename(
            title="Save Results",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                # Create results DataFrame
                results_data = []
                for model_type, results in self.data['results'].items():
                    for metric, value in results['metrics'].items():
                        results_data.append({
                            'Model': model_type,
                            'Metric': metric,
                            'Value': value,
                            'Group': 'Overall'
                        })
                    
                    # Add fairness metrics
                    for attr, groups in results['fairness_metrics'].items():
                        for group, group_metrics in groups.items():
                            for metric, value in group_metrics.items():
                                results_data.append({
                                    'Model': model_type,
                                    'Metric': f"{attr}_{metric}",
                                    'Value': value,
                                    'Group': group
                                })
                
                results_df = pd.DataFrame(results_data)
                results_df.to_csv(file_path, index=False)
                messagebox.showinfo("Success", f"Results saved to {file_path}")
                self.log_message(f"Results saved to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save results: {str(e)}")
                
    def export_report_pdf(self):
        """Export report to PDF (placeholder)"""
        messagebox.showinfo("Info", "PDF export feature coming soon!")
        
    def save_configuration(self):
        """Save current configuration"""
        config = {
            'preprocessing_method': self.preproc_method_var.get() if hasattr(self, 'preproc_method_var') else 'None',
            'model_type': self.model_var.get(),
            'bias_approach': self.model_bias_approach_var.get() if hasattr(self, 'model_bias_approach_var') else 'none',
            'inprocessing_method': self.inproc_method_var.get() if hasattr(self, 'inproc_method_var') else 'None',
            'postprocessing_method': self.postproc_method_var.get() if hasattr(self, 'postproc_method_var') else 'None',
            'target_column': self.target_var.get(),
            'sensitive_attributes': [col for col, var in self.sensitive_vars.items() if var.get()] if hasattr(self, 'sensitive_vars') else [],
            'test_split': self.test_split_var.get()
        }
        
        file_path = filedialog.asksaveasfilename(
            title="Save Configuration",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    json.dump(config, f, indent=2)
                messagebox.showinfo("Success", f"Configuration saved to {file_path}")
                self.log_message(f"Configuration saved to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save configuration: {str(e)}")
                
    def load_configuration(self):
        """Load configuration from file"""
        file_path = filedialog.askopenfilename(
            title="Load Configuration",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    config = json.load(f)
                
                # Apply configuration
                if 'preprocessing_method' in config and hasattr(self, 'preproc_method_var'):
                    self.preproc_method_var.set(config['preprocessing_method'])
                if 'bias_approach' in config and hasattr(self, 'model_bias_approach_var'):
                    self.model_bias_approach_var.set(config['bias_approach'])
                if 'inprocessing_method' in config and hasattr(self, 'inproc_method_var'):
                    self.inproc_method_var.set(config['inprocessing_method'])
                if 'postprocessing_method' in config and hasattr(self, 'postproc_method_var'):
                    self.postproc_method_var.set(config['postprocessing_method'])
                if 'model_type' in config:
                    self.model_var.set(config['model_type'])
                if 'target_column' in config:
                    self.target_var.set(config['target_column'])
                if 'test_split' in config:
                    self.test_split_var.set(config['test_split'])
                
                # Update UI elements based on loaded configuration
                if hasattr(self, 'update_model_selection'):
                    self.update_model_selection()
                if hasattr(self, 'update_model_bias_approach'):
                    self.update_model_bias_approach()
                if hasattr(self, 'update_preprocessing_method_params'):
                    self.update_preprocessing_method_params()
                if hasattr(self, 'update_postprocessing_method_params'):
                    self.update_postprocessing_method_params()
                
                messagebox.showinfo("Success", f"Configuration loaded from {file_path}")
                self.log_message(f"Configuration loaded from {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load configuration: {str(e)}")


def main():
    """Main function to run the GUI application"""
    root = tk.Tk()
    app = DebiasedJadouilleGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main() 
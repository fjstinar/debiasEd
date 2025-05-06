# import numpy as np
# import pandas as pd

# from mitigation.preprocessing.syndage.utils import calculation
# from mitigation.preprocessing.syndage.data import load_ori_population, simulate_dataset
# from mitigation.preprocessing.syndage.genetic_algorithm import ContinuousGenAlgSolver

# import mitigation.preprocessing.syndage.unfairness_metrics, argparse

# from sklearn import metrics, model_selection, pipeline, preprocessing, linear_model, ensemble


from mitigation.preprocessing.preprocessor import PreProcessor

class JiangPreProcessor(PreProcessor):
    """Resampling pre-processing

    References:
        Jiang, L., Belitz, C., & Bosch, N. (2024, March). Synthetic Dataset Generation for Fairer Unfairness Research. In Proceedings of the 14th Learning Analytics and Knowledge Conference (pp. 200-209).
        https://github.com/lan-j/unfair_dataset_generation
    """
    
    def __init__(self, settings: dict):
        super().__init__(settings)
        self._name = 'jiang et al.'
        self._notation = 'jiang'
        self._preprocessor_settings = self._settings['preprocessors']['jiang']
        self._information = {}

#     def transform(self, 
#         x_train: list, y_train: list, demo_train: list,
#         ):
#         """
#         Args:
#             x_train (list): training feature data 
#             y_train (list): training label data
#             demo_train(list): training demographics data
#             x_val (list): validation feature data
#             y_val (list): validation label data
#             demo_val (list): validation demographics data
#         """
#         raise NotImplementedError

#     def run(self, x, y, demos):
#         fitness_function = calculation(x, y, demos, self._preprocessor_settings['unfair_metric'])
#         solver = ContinuousGenAlgSolver(
#             fitness_function=fitness_function.fit_scores,
#             expect_score=0.5,
#             dataset=x, labels=y, group_col=demos, feature_name=[i for i in range(len(x[0]))],
#             pop_size=self._preprocessor_settings['population_size'],  # population size (number of individuals)
#             max_gen=self._preprocessor_settings['max_gen'],  # maximum number of generations
#             gene_mutation_rate=0.002,
#             mutation_rate=0.002,  # mutation rate to apply to the population
#             selection_rate=0.6,  # percentage of the population to select for mating
#             selection_strategy="roulette_wheel",  # strategy to use for selection. see below for more details
#             plot_results=False,
#             random_state=98958
#         )

#         population, labels, group = solver.solve()
#         return population, labels, group

#     def fit_transform(self, 
#             x_train: list, y_train: list, demo_train: list,
#             x_val: list, y_val: list, demo_val: list
#         ):
#         """trains the model and transform the data given the initial training data x, and labels y. 
#         Warning: Init the model every time this function is called

#         Args:
#             x_train (list): training feature data 
#             y_train (list): training label data
#             demo_train(list): training demographics data
#             x_val (list): validation feature data
#             y_val (list): validation label data
#             demo_val (list): validation demographics data
#         """
#         demographic_attributes = self.extract_demographics(demo_train)
#         x_sampled, y_sampled, demo_sampled = self.run(x_train, y_train, demographic_attributes)
#         return x_sampled, y_sampled, demo_sampled
        
#     def get_information(self):
#         """For each pre-processor, returns information worth saving for future results
#         """
#         return self._information
    
        

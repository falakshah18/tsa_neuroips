# ADD THIS to benchmarks.py

from baselines.surrogate_gradient import SurrogateGradSNN
from baselines.ann_to_snn import ANNtoSNNConverter
from baselines.stdp import SupervisedSTDP
from baselines.eprop import EpropSNN
from baselines.temporal_coding import TTFSNetwork
from models.tst_v2 import TemporalSpikingTransformer
class BaselineComparison:
    """
    This is the core UGRP contribution:
    Systematic evaluation of ALL algorithm families
    """
    
    def __init__(self, datasets: list, save_dir: str = './baseline_comparison_results'):
        self.datasets = datasets  # ['nmnist', 'shd']
        self.save_dir = save_dir
    
    def get_all_models(self):
        return {
            # Existing algorithm families (REVIEW part)
            'Surrogate_Gradient': SurrogateGradSNN(),
            'ANN_to_SNN': ANNtoSNNConverter(),
            'Supervised_STDP': SupervisedSTDP(),
            'E_prop': EpropSNN(),
            'Temporal_Coding_TTFS': TTFSNetwork(),
            # Your novel contribution (PROPOSE part)
            'TSA_Ours': TemporalSpikingTransformer(),
        }
    
    def evaluate_on_criteria(self, model, loader):
        """
        5-criterion evaluation rubric
        This is what makes it a proper REVIEW paper
        """
        return {
            'accuracy': self.measure_accuracy(model, loader),
            'energy_uJ': self.measure_energy(model, loader),
            'biological_plausibility': self.score_bio_plausibility(model),
            'training_stability': self.measure_stability(model, loader),
            'convergence_speed': self.measure_convergence(model, loader),
        }
    
    def score_bio_plausibility(self, model) -> dict:
        """
        Your 5-criterion biological plausibility rubric
        from earlier UGRP work — plug it in here
        """
        return {
            'local_learning_rule': False,  # score per algorithm
            'spike_based_communication': True,
            'temporal_dynamics': True,
            'no_weight_transport': False,
            'online_learning': False,
            'total_score': 2,  # out of 5
        }
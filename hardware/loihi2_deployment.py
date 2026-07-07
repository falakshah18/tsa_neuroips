# hardware/loihi2_deployment.py
"""
Deploy on Intel Loihi 2 neuromorphic chip
This is GOLD for top-tier publications
"""

# Note: This requires Intel Loihi access through INRC
# If you don't have access, use simulation (see next section)
import numpy as np
import torch.nn as nn
from typing import Dict
from torch.utils.data import DataLoader
from models.tst_v2 import TemporalSpikingTransformer
from spikingjelly.activation_based import neuron, functional
try:
    import nxsdk
    from nxsdk.graph.nxboard import N2Board
    LOIHI_AVAILABLE = True
except ImportError:
    LOIHI_AVAILABLE = False
    print("[WARNING] Loihi SDK not available. Using simulation mode.")


class LoihiDeployer:
    """
    Deploy TST model on Loihi 2 chip
    """
    
    def __init__(self, model: TemporalSpikingTransformer):
        if not LOIHI_AVAILABLE:
            raise RuntimeError("Loihi SDK not installed")
        
        self.model = model
        self.board = N2Board()
        
    def convert_to_loihi(self):
        """
        Convert PyTorch model to Loihi network
        """
        # This is complex - simplified version
        # Full implementation requires detailed NxSDK knowledge
        
        net = self.board.createNetwork('TST_Network')
        
        # Convert each layer
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                # Create compartment group for neurons
                num_neurons = module.out_features
                neurons = net.createCompartmentGroup(size=num_neurons)
                
                # Set LIF dynamics
                neurons.vThMant = 100  # Threshold
                neurons.decayV = int(1/2.0 * 4096)  # Voltage decay (tau=2.0)
                
                # Create synapses
                weights = module.weight.detach().cpu().numpy()
                # Quantize weights to int8
                scale = 127 / weights.abs().max()
                weights_quantized = (weights * scale).astype(np.int8)
                
                # Add synaptic connections
                # This is simplified - actual implementation more complex
        
        return net
    
    def run_inference(self, input_spikes: np.ndarray) -> Dict:
        """
        Run inference on Loihi chip
        """
        # Convert network
        net = self.convert_to_loihi()
        
        # Configure input
        input_layer = net.inputLayer
        input_layer.send(input_spikes)
        
        # Run
        self.board.run(timesteps=input_spikes.shape[0])
        
        # Get output spikes
        output_spikes = net.outputLayer.probe.data
        
        # Measure energy
        energy = self.board.energyProbe.data
        
        return {
            'output_spikes': output_spikes,
            'energy_uJ': energy,
            'latency_ms': input_spikes.shape[0] * 0.001,  # 1ms per timestep
        }


class LoihiSimulator:
    """
    Simulate Loihi execution when chip not available
    Based on published Loihi 2 specifications
    """
    
    def __init__(self):
        # Loihi 2 specs from Intel's paper
        self.neuron_latency = 1e-6  # 1 microsecond per timestep
        self.E_neuron = 23.6e-12  # J
        self.E_spike = 0.1e-12  # J
        self.E_synapse = 3.5e-12  # J
        
    def simulate_execution(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        n_samples: int = 100
    ) -> Dict:
        """
        Simulate model execution on Loihi
        """
        model.eval()
        device = next(model.parameters()).device
        
        total_energy = 0
        total_latency = 0
        total_accuracy = 0
        total_samples = 0
        n_batches = 0
        
        with torch.no_grad():
            for data, target in dataloader:
                if total_samples >= n_samples:
                    break
                
                data = data.to(device)
                target = target.to(device)
                
                # Transpose from DataLoader's [B, T, ...] to model's [T, B, ...]
                if data.dim() >= 3:
                    data = data.transpose(0, 1).contiguous()
                T = data.shape[0]
                B = data.shape[1]
                
                # Count operations
                neuron_ops = 0
                spike_ops = 0
                synapse_ops = 0
                
                # Forward pass (model returns (logits, metrics) tuple)
                logits, _ = model(data)
                output = logits
                
                # Count from recorded activity
                for module in model.modules():
                    if isinstance(module, (neuron.LIFNode, neuron.ParametricLIFNode)):
                        if hasattr(module, 'v'):
                            neuron_ops += module.v.numel()
                        if hasattr(module, 'spike'):
                            spike_ops += module.spike.sum().item()
                    
                    if isinstance(module, nn.Linear):
                        synapse_ops += module.weight.numel()
                
                # Calculate energy
                energy = (
                    neuron_ops * self.E_neuron +
                    spike_ops * self.E_spike +
                    synapse_ops * self.E_synapse
                )
                
                # Calculate latency (based on timesteps)
                latency = T * self.neuron_latency
                
                # Accuracy
                pred = output.argmax(dim=1)
                acc = (pred == target).float().mean()
                
                total_energy += energy
                total_latency += latency
                total_accuracy += acc.item()
                total_samples += B
                n_batches += 1
                
                functional.reset_net(model)
        
        return {
            'avg_energy_per_sample_uJ': (total_energy / max(total_samples, 1)) * 1e6,
            'avg_latency_per_sample_ms': (total_latency / max(total_samples, 1)) * 1e3,
            'accuracy': total_accuracy / max(n_batches, 1),
            'total_samples': total_samples,
        }


def hardware_validation_report(
    model: TemporalSpikingTransformer,
    dataloader: DataLoader,
    use_real_hardware: bool = False
) -> Dict:
    """
    Complete hardware validation
    """
    print("\n" + "="*60)
    print("HARDWARE VALIDATION")
    print("="*60)
    
    if use_real_hardware and LOIHI_AVAILABLE:
        print("🔥 Running on Intel Loihi 2 chip...")
        deployer = LoihiDeployer(model)
        # Extract a sample batch and convert to numpy for the deployer
        sample_data, _ = next(iter(dataloader))
        if sample_data.dim() >= 3:
            sample_data = sample_data.transpose(0, 1).contiguous()
        raw = deployer.run_inference(sample_data.cpu().numpy())
        # Normalize dict keys to match simulate_execution output
        results = {
            'avg_energy_per_sample_uJ': raw['energy_uJ'] / sample_data.shape[1],
            'avg_latency_per_sample_ms': raw['latency_ms'],
            'accuracy': 0.0,  # Real hardware path doesn't compute accuracy
        }
        hardware_type = "Loihi 2 (Real Hardware)"
    else:
        print("💻 Running simulation based on Loihi 2 specs...")
        simulator = LoihiSimulator()
        results = simulator.simulate_execution(model, dataloader)
        hardware_type = "Loihi 2 (Simulated)"
    
    print(f"\nHardware: {hardware_type}")
    print(f"Energy: {results['avg_energy_per_sample_uJ']:.2f} μJ")
    print(f"Latency: {results['avg_latency_per_sample_ms']:.2f} ms")
    print(f"Accuracy: {results['accuracy']:.4f}")
    
    # Compare to standard GPU execution
    print("\n Comparison with GPU:")
    # Measure GPU energy (approximate)
    import time
    start = time.time()
    with torch.no_grad():
        for data, _ in dataloader:
            if data.dim() >= 3:
                data = data.transpose(0, 1).contiguous()
            logits, _ = model(data)
            functional.reset_net(model)
            break
    gpu_time = time.time() - start
    
    # Typical GPU power: ~300W for inference
    gpu_energy_per_sample = 300 * gpu_time / data.shape[1]  # Watts * seconds
    
    print(f"GPU Energy: ~{gpu_energy_per_sample*1000:.2f} mJ")
    print(f"Reduction: {gpu_energy_per_sample*1e6 / results['avg_energy_per_sample_uJ']:.1f}×")
    
    return results
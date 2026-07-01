# training/trainer_v2.py
"""
Production training with all bells and whistles
"""
from spikingjelly.activation_based import functional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import wandb
from tqdm import tqdm
import numpy as np
from typing import Dict, Optional, Tuple
import json
from pathlib import Path
import torch.nn.functional as F

class AdvancedTrainer:
    """
    Professional training framework
    - Mixed precision training
    - Gradient accumulation
    - Learning rate warmup + cosine decay
    - Early stopping
    - Model checkpointing
    - Comprehensive logging
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        config: dict,
        device: str = 'cuda',
        use_wandb: bool = True,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        self.device = device
        self.use_wandb = use_wandb

        # Loss function
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=config.get('label_smoothing', 0.1))

        # Spike regularization weight (plain float — avoids silent "untrained parameter" bug)
        self.spike_reg_weight = config.get('spike_reg', 0.001)

        # Optimizer
        self.optimizer = self._create_optimizer()
        
        # Learning rate scheduler
        self.scheduler = self._create_scheduler()
        
        # Mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if config.get('mixed_precision', True) else None
        
        # Gradient accumulation
        self.accum_steps = config.get('gradient_accumulation_steps', 1)
        
        # Logging
        if use_wandb:
            wandb.init(
                project=config['project_name'],
                config=config,
                name=config.get('run_name', 'tst_experiment')
            )
        self.writer = SummaryWriter(log_dir=config['log_dir'])
        
        # Checkpointing
        self.checkpoint_dir = Path(config['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Early stopping
        self.best_val_acc = 0.0
        self.patience = config.get('patience', 20)
        self.patience_counter = 0
        
        # Metrics tracking
        self.train_metrics = []
        self.val_metrics = []
        
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """
        Advanced optimizer with parameter groups
        """
        # Separate learning rates for different components
        param_groups = [
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if 'neuron' not in n],
                'lr': self.config['lr'],
                'weight_decay': self.config.get('weight_decay', 0.05)
            },
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if 'neuron' in n],
                'lr': self.config['lr'] * 0.1,  # Lower LR for neuron params
                'weight_decay': 0.0  # No decay for neuron parameters
            }
        ]
        
        optimizer = torch.optim.AdamW(
            param_groups,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        return optimizer
    
    def _create_scheduler(self):
        """
        Warmup + Cosine annealing
        """
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
        
        warmup_epochs = self.config.get('warmup_epochs', 10)
        total_epochs = self.config['epochs']
        
        warmup = LinearLR(
            self.optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=warmup_epochs
        )
        
        cosine = CosineAnnealingLR(
            self.optimizer,
            T_max=total_epochs - warmup_epochs,
            eta_min=self.config.get('min_lr', 1e-6)
        )
        
        scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_epochs]
        )
        
        return scheduler
    
    # In artifacts/training/trainer_v2.py (or wherever it is)
# Replace the compute_loss method with this cleaned version:
def compute_loss(self, outputs: torch.Tensor, targets: torch.Tensor, metrics: dict) -> Tuple[torch.Tensor, dict]:
    """Multi-objective loss with proper differentiable spike reg."""
    ce_loss = self.ce_loss(outputs, targets)
    
    # Differentiable spike regularization via learnable thresholds (lower threshold = more spikes)
    spike_reg = torch.tensor(0.0, device=self.device)
    if hasattr(self.model, 'blocks'):
        for blk in self.model.blocks:
            if hasattr(blk.attn, 'attn_threshold'):
                spike_reg += F.softplus(blk.attn.attn_threshold).mean()
    
    loss = ce_loss + self.spike_reg_weight * spike_reg
    
    loss_dict = {
        'total_loss': loss.item(),
        'ce_loss': ce_loss.item(),
        'spike_reg': spike_reg.item(),
        'spike_reg_weight': self.spike_reg_weight,
    }
    return loss, loss_dict
    
    def train_epoch(self, epoch: int) -> dict:
        """
        Single training epoch with all optimizations
        """
        self.model.train()
        
        total_loss = 0
        total_acc = 0
        total_samples = 0
        all_spike_counts = []
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        
        for batch_idx, (data, target) in enumerate(pbar):
            data = data.to(self.device)
            target = target.to(self.device)
            # DataLoader yields [B, T, C, H, W]; model expects [T, B, C, H, W]
            if data.dim() == 5:
                data = data.permute(1, 0, 2, 3, 4).contiguous()
            elif data.dim() == 3:
                data = data.permute(1, 0, 2).contiguous()
            
            # Mixed precision forward pass
            with torch.cuda.amp.autocast(enabled=(self.scaler is not None)):
                outputs, metrics = self.model(data)
                loss, loss_dict = self.compute_loss(outputs, target, metrics)
                
                # Gradient accumulation
                loss = loss / self.accum_steps
            
            # Backward pass
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Optimizer step (every accum_steps)
            if (batch_idx + 1) % self.accum_steps == 0:
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
            
            # Reset neuron states
            functional.reset_net(self.model)
            
            # Metrics
            pred = outputs.argmax(dim=1)
            acc = (pred == target).float().mean()
            
            batch_size = data.size(1)  # B dimension (DataLoader batches on dim 0)
            total_loss += loss.item() * self.accum_steps * batch_size
            total_acc += acc.item() * batch_size
            total_samples += batch_size
            
            # Track spike counts
            total_spikes = sum(
                block['attention']['total_spikes']
                for block in metrics['blocks']
            )
            all_spike_counts.append(total_spikes)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item() * self.accum_steps,
                'acc': acc.item(),
                'lr': self.scheduler.get_last_lr()[0]
            })
            
            # Log to wandb
            if self.use_wandb and batch_idx % 10 == 0:
                step = epoch * len(self.train_loader) + batch_idx
                wandb.log({
                    'train/loss': loss.item() * self.accum_steps,
                    'train/acc': acc.item(),
                    'train/lr': self.scheduler.get_last_lr()[0],
                    'train/spike_count': total_spikes,
                    **{f'train/{k}': v for k, v in loss_dict.items()}
                }, step=step)
        
        # Epoch metrics
        epoch_metrics = {
            'loss': total_loss / total_samples,
            'acc': total_acc / total_samples,
            'avg_spikes': np.mean(all_spike_counts),
            'std_spikes': np.std(all_spike_counts),
        }
        
        return epoch_metrics
    
    @torch.no_grad()
    def evaluate(self, loader: DataLoader, split: str = 'val') -> dict:
        """
        Evaluation with comprehensive metrics
        """
        self.model.eval()
        
        total_loss = 0
        total_acc = 0
        total_samples = 0
        all_predictions = []
        all_targets = []
        all_spike_counts = []
        all_energies = []
        
        for data, target in tqdm(loader, desc=f'Evaluating {split}'):
            data = data.to(self.device)
            target = target.to(self.device)
            if data.dim() == 5:
                data = data.permute(1, 0, 2, 3, 4).contiguous()
            elif data.dim() == 3:
                data = data.permute(1, 0, 2).contiguous()
            
            # Forward pass
            outputs, metrics = self.model(data)
            loss, _ = self.compute_loss(outputs, target, metrics)
            
            # Energy analysis
            energy_dict = self.model.get_energy_breakdown(data)
            all_energies.append(energy_dict['energy_per_sample_uJ'])
            
            # Metrics
            pred = outputs.argmax(dim=1)
            acc = (pred == target).float().mean()
            
            batch_size = data.size(1)
            total_loss += loss.item() * batch_size
            total_acc += acc.item() * batch_size
            total_samples += batch_size
            
            all_predictions.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            
            total_spikes = sum(
                block['attention']['total_spikes']
                for block in metrics['blocks']
            )
            all_spike_counts.append(total_spikes)
            
            functional.reset_net(self.model)
        
        # Compute additional metrics
        from sklearn.metrics import confusion_matrix, classification_report
        
        cm = confusion_matrix(all_targets, all_predictions)
        report = classification_report(
            all_targets, all_predictions, 
            output_dict=True, zero_division=0
        )
        
        eval_metrics = {
            'loss': total_loss / total_samples,
            'acc': total_acc / total_samples,
            'avg_spikes': np.mean(all_spike_counts),
            'std_spikes': np.std(all_spike_counts),
            'avg_energy_uJ': np.mean(all_energies),
            'std_energy_uJ': np.std(all_energies),
            'confusion_matrix': cm,
            'classification_report': report,
        }
        
        return eval_metrics
    
    def save_checkpoint(self, epoch: int, metrics: dict, is_best: bool = False):
        """
        Save model checkpoint
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config,
        }
        
        # Save regular checkpoint
        path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"💾 Saved best model with acc={metrics['acc']:.4f}")
    
    def train(self) -> dict:
        """
        Complete training loop
        """
        print("🚀 Starting training...")
        print(f"📊 Config: {json.dumps(self.config, indent=2)}")
        
        for epoch in range(self.config['epochs']):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{self.config['epochs']}")
            print(f"{'='*60}")
            
            # Train
            # Train
            train_metrics = self.train_epoch(epoch)
              # ← ADD THIS LINE
            print(f"Train: Loss={train_metrics['loss']:.4f}, "
                  f"Acc={train_metrics['acc']:.4f}, "
                  f"Spikes={train_metrics['avg_spikes']:.0f}")
            
            # Validate
            # Validate
            val_metrics = self.evaluate(self.val_loader, 'val')
               # ← ADD THIS LINE
            print(f"Val:   Loss={val_metrics['loss']:.4f}, "
                  f"Acc={val_metrics['acc']:.4f}, "
                  f"Energy={val_metrics['avg_energy_uJ']:.2f}μJ")
            
            # Log to wandb
            if self.use_wandb:
                _non_serializable = {'confusion_matrix', 'classification_report'}
                wandb.log({
                    'epoch': epoch,
                    **{f'train/{k}': v for k, v in train_metrics.items()},
                    **{f'val/{k}': v for k, v in val_metrics.items()
                       if k not in _non_serializable},
                })
            
            # Learning rate step
            self.scheduler.step()
            
            # Save checkpoint
            is_best = val_metrics['acc'] > self.best_val_acc
            if is_best:
                self.best_val_acc = val_metrics['acc']
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            self.save_checkpoint(epoch, val_metrics, is_best)
            
            # Early stopping
            if self.patience_counter >= self.patience:
                print(f"\n⚠️ Early stopping triggered after {epoch+1} epochs")
                break
        
        # Final test evaluation
        print(f"\n{'='*60}")
        print("📊 Final Test Evaluation")
        print(f"{'='*60}")
        
        # Load best model
        best_checkpoint = torch.load(self.checkpoint_dir / 'best_model.pth')
        self.model.load_state_dict(best_checkpoint['model_state_dict'])
        
        test_metrics = self.evaluate(self.test_loader, 'test')
        print(f"Test: Acc={test_metrics['acc']:.4f}, "
              f"Energy={test_metrics['avg_energy_uJ']:.2f}μJ")
        
        # Log final metrics
        if self.use_wandb:
            wandb.log({
                'final/test_acc': test_metrics['acc'],
                'final/test_energy': test_metrics['avg_energy_uJ'],
            })
        
        return {
            'best_val_acc': self.best_val_acc,
            'test_metrics': test_metrics,
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics,
        }
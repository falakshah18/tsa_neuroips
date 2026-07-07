# training/trainer_v2.py
"""
VERSION 2: Production-ready training framework for TSA / TST models.

Features:
- AdamW optimizer with weight decay
- Linear warmup + cosine annealing LR schedule
- Mixed precision (AMP) training
- Gradient accumulation
- Spike-count based energy regularization
- Early stopping on validation accuracy (with patience)
- Checkpointing (best + last)
- Optional Weights & Biases logging
- TensorBoard logging (always on, cheap and dependency-light)
"""

import math
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from spikingjelly.activation_based import functional

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:  # pragma: no cover - optional dependency
    _HAS_TQDM = False

    def tqdm(iterable, **kwargs):  # no-op fallback
        return iterable

try:
    from torch.utils.tensorboard import SummaryWriter
    _HAS_TENSORBOARD = True
except ImportError:  # pragma: no cover - optional dependency
    SummaryWriter = None
    _HAS_TENSORBOARD = False

try:
    import wandb as _wandb
    _HAS_WANDB = True
except ImportError:  # pragma: no cover - optional dependency
    _wandb = None
    _HAS_WANDB = False


DEFAULT_CONFIG = {
    'epochs': 100,
    'lr': 1e-3,
    'min_lr': 1e-6,
    'weight_decay': 0.05,
    'warmup_epochs': 10,
    'patience': 30,
    'mixed_precision': True,
    'gradient_accumulation_steps': 1,
    'spike_reg': 0.001,
    'grad_clip': 1.0,
    'label_smoothing': 0.1,
    'log_dir': './logs/run',
    'checkpoint_dir': './checkpoints/run',
    'project_name': 'TSA_NEUROIPS',
    'run_name': 'run',
    'save_every': 10,
    'use_compile': False,
}


class AdvancedTrainer:
    """
    General-purpose trainer for spiking-neural-network classification models.

    Expected model interface:
        logits, metrics = model(x)   # x: [T, B, C, H, W] or [T, B, D]
        metrics = {'blocks': [{'attention': {'total_spikes': int, ...}}, ...]}
        (metrics format is best-effort; trainer degrades gracefully if absent)
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        config: Optional[Dict] = None,
        device: Optional[str] = None,
        use_wandb: bool = False,
    ):
        """
        Args:
            model: A model following the (logits, metrics) forward
                interface described in the class docstring.
            train_loader, val_loader, test_loader: Standard PyTorch
                DataLoaders. Batches are expected as (x, y) with x shaped
                [B, T, ...] (batch-first, as DataLoader's default_collate
                produces) -- `_prepare_batch` transposes to [T, B, ...]
                before the model sees it.
            config: Overrides merged onto DEFAULT_CONFIG (see module level
                constant above for every available key and its default).
            device: 'cuda' or 'cpu'. Defaults to cuda if available.
            use_wandb: Enable Weights & Biases logging. Silently falls back
                to disabled if wandb isn't installed.
        """
        self.config = {**DEFAULT_CONFIG, **(config or {})}

        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)

        use_compile = self.config.get('use_compile', False)
        if use_compile and hasattr(torch, 'compile'):
            try:
                self.model = torch.compile(self.model)
                print(f"[AdvancedTrainer] Model compiled with torch.compile")
            except Exception as e:
                print(f"[AdvancedTrainer] torch.compile failed ({e}); continuing without it")

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.use_wandb = use_wandb and _HAS_WANDB
        if use_wandb and not _HAS_WANDB:
            print("[AdvancedTrainer] wandb requested but not installed; continuing without it.")

        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=self.config.get('label_smoothing', 0.0)
        )

        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()

        self.use_amp = bool(self.config['mixed_precision']) and self.device.startswith('cuda')
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)

        self.log_dir = Path(self.config['log_dir'])
        self.checkpoint_dir = Path(self.config['checkpoint_dir'])
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.writer = SummaryWriter(log_dir=str(self.log_dir)) if _HAS_TENSORBOARD else None

        if self.use_wandb:
            _wandb.init(
                project=self.config.get('project_name', 'TSA_NEUROIPS'),
                name=self.config.get('run_name', 'run'),
                config=self.config,
                reinit=True,
            )

        self.best_val_acc = 0.0
        self.epochs_without_improvement = 0
        self.history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """AdamW over all model parameters, using config['lr']/['weight_decay']."""
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config['lr'],
            weight_decay=self.config['weight_decay'],
        )

    def _build_scheduler(self):
        """
        Linear warmup for config['warmup_epochs'], then cosine decay from
        config['lr'] down to config['min_lr'] over the remaining epochs.
        Implemented as a single LambdaLR so both phases share one scheduler
        object (stepped once per epoch in `train`).
        """
        epochs = max(int(self.config['epochs']), 1)
        warmup_epochs = min(int(self.config.get('warmup_epochs', 0)), epochs)
        base_lr = self.config['lr']
        min_lr = self.config.get('min_lr', 0.0)

        def lr_lambda(epoch: int) -> float:
            if warmup_epochs > 0 and epoch < warmup_epochs:
                return (epoch + 1) / max(warmup_epochs, 1)
            progress = (epoch - warmup_epochs) / max(epochs - warmup_epochs, 1)
            progress = min(max(progress, 0.0), 1.0)
            cosine = 0.5 * (1 + math.cos(math.pi * progress))
            min_ratio = min_lr / base_lr if base_lr > 0 else 0.0
            return min_ratio + (1 - min_ratio) * cosine

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    # ------------------------------------------------------------------
    # Batch utilities
    # ------------------------------------------------------------------

    def _prepare_batch(self, x: torch.Tensor, y: torch.Tensor):
        """Move batch to device and ensure [T, B, ...] layout expected by SNN models."""
        x = x.to(self.device, non_blocking=True)
        y = y.to(self.device, non_blocking=True)

        # tonic's ToFrame (used for NMNIST/SHD) yields integer (e.g. int16/Short)
        # event-count frames. Conv/Linear layers have float32 weights, so this
        # must be cast to float before it reaches any layer.
        x = x.float()

        # DataLoaders typically yield [B, T, ...]; models expect [T, B, ...].
        if x.dim() >= 3:
            x = x.transpose(0, 1).contiguous()

        return x, y

    def _count_spikes(self, metrics: Dict) -> float:
        """Best-effort extraction of total spike count from a model's metrics dict."""
        if not isinstance(metrics, dict):
            return 0.0

        total = 0.0
        blocks = metrics.get('blocks')
        if blocks:
            for block in blocks:
                attn = block.get('attention', {}) if isinstance(block, dict) else {}
                total += float(attn.get('total_spikes', 0.0))
            return total

        # Fallback: metrics might itself look like {'total_spikes': ...}
        if 'total_spikes' in metrics:
            return float(metrics['total_spikes'])

        return total

    # ------------------------------------------------------------------
    # Core loops
    # ------------------------------------------------------------------

    def train_epoch(self, epoch: int) -> float:
        """
        Run one full pass over train_loader.

        Handles gradient accumulation (accumulates `accum_steps` batches
        before each optimizer step), AMP (mixed precision, only active on
        CUDA), gradient clipping, and the spike-count energy regularizer
        (added to the loss, weighted by config['spike_reg']). Resets all
        neuron membrane state after every batch via
        `functional.reset_net`, since each batch is an independent sample
        sequence.

        Args:
            epoch: Current epoch index (0-based), used only for the
                progress bar label.

        Returns:
            Mean training loss over the epoch.
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        accum_steps = max(int(self.config.get('gradient_accumulation_steps', 1)), 1)
        spike_reg_weight = self.config.get('spike_reg', 0.0)
        grad_clip = self.config.get('grad_clip', None)

        self.optimizer.zero_grad()

        pbar = tqdm(
            enumerate(self.train_loader),
            total=len(self.train_loader),
            desc=f"  epoch {epoch + 1}",
            leave=False,
            disable=not _HAS_TQDM,
        )
        for step, (x, y) in pbar:
            x, y = self._prepare_batch(x, y)

            with torch.amp.autocast('cuda', enabled=self.use_amp):
                logits, metrics = self.model(x)
                loss = self.criterion(logits, y)

                if spike_reg_weight:
                    total_spikes = self._count_spikes(metrics)
                    # Normalize by batch size so the regularizer is scale-stable.
                    batch_size = y.size(0)
                    loss = loss + spike_reg_weight * (total_spikes / max(batch_size, 1))

                loss = loss / accum_steps

            self.scaler.scale(loss).backward()

            if (step + 1) % accum_steps == 0:
                if grad_clip:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            functional.reset_net(self.model)

            total_loss += loss.item() * accum_steps
            n_batches += 1

            if _HAS_TQDM:
                pbar.set_postfix(loss=f"{total_loss / n_batches:.4f}")

        # Flush any remaining accumulated gradients.
        if n_batches % accum_steps != 0:
            if grad_clip:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Dict:
        """
        Run inference over `loader` with gradients disabled.

        Used for both validation (during `train`) and the final test
        evaluation, since the two only differ in which loader is passed.

        Args:
            loader: Any of train_loader/val_loader/test_loader, or an
                external DataLoader with the same (x, y) batch format.

        Returns:
            Dict with 'loss' (mean cross-entropy), 'acc' (accuracy in
            [0, 1]), and 'avg_energy_uJ' (estimated energy per sample, at
            0.1 pJ/spike -- see models.tst_v2.get_energy_breakdown for the
            same constant used elsewhere).
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        total_spikes = 0.0
        n_batches = 0

        for x, y in loader:
            x, y = self._prepare_batch(x, y)

            with torch.amp.autocast('cuda', enabled=self.use_amp):
                logits, metrics = self.model(x)
                loss = self.criterion(logits, y)

            total_loss += loss.item()
            preds = logits.argmax(dim=-1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            total_spikes += self._count_spikes(metrics)
            n_batches += 1

            functional.reset_net(self.model)

        acc = correct / max(total, 1)
        avg_loss = total_loss / max(n_batches, 1)
        # Rough energy estimate: 0.1 pJ per spike (matches models/tst_v2.get_energy_breakdown),
        # converted to microjoules and averaged per sample.
        avg_energy_uj = (total_spikes * 0.1e-6) / max(total, 1)

        return {
            'loss': avg_loss,
            'acc': acc,
            'avg_energy_uJ': avg_energy_uj,
        }

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """
        Save model/optimizer/scheduler state to checkpoint_dir/last.pth,
        and additionally to checkpoint_dir/best.pth if is_best is True.

        Args:
            epoch: Current epoch index, stored in the checkpoint for
                resuming.
            is_best: Whether this epoch achieved the best validation
                accuracy so far.
        """
        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_acc': self.best_val_acc,
            'config': self.config,
        }
        torch.save(state, self.checkpoint_dir / 'last.pth')
        if is_best:
            torch.save(state, self.checkpoint_dir / 'best.pth')

    def load_checkpoint(self, path: str):
        """
        Restore model/optimizer/scheduler state from a checkpoint saved by
        `save_checkpoint`. Also restores `best_val_acc` so early stopping
        and best-model tracking stay consistent across a resumed run.

        Args:
            path: Path to a .pth file saved by save_checkpoint.

        Returns:
            The epoch index stored in the checkpoint (0 if absent).
        """
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        if 'optimizer_state_dict' in ckpt:
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scheduler_state_dict' in ckpt:
            self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        self.best_val_acc = ckpt.get('best_val_acc', 0.0)
        return ckpt.get('epoch', 0)

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self) -> Dict:
        """
        Full training loop: for each epoch, train one pass, validate,
        step the LR scheduler, checkpoint, log, and check early stopping.
        After the loop ends (either all epochs completed or early
        stopping triggered), reloads the best checkpoint (by validation
        accuracy) if one was saved, then runs a final evaluation on
        test_loader.

        Returns:
            Dict with:
                best_val_acc: Best validation accuracy seen during training.
                test_metrics: Output of `evaluate(test_loader)` using the
                    best checkpoint's weights.
                history: Dict of per-epoch lists ('train_loss', 'val_loss',
                    'val_acc') for plotting/analysis.
        """
        epochs = int(self.config['epochs'])
        patience = int(self.config.get('patience', epochs))

        start_time = time.time()

        for epoch in range(epochs):
            epoch_start = time.time()

            train_loss = self.train_epoch(epoch)
            val_metrics = self.evaluate(self.val_loader)
            self.scheduler.step()

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['acc'])

            is_best = val_metrics['acc'] > self.best_val_acc
            if is_best:
                self.best_val_acc = val_metrics['acc']
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1

            if (epoch + 1) % max(int(self.config.get('save_every', 1)), 1) == 0 or is_best:
                self.save_checkpoint(epoch, is_best=is_best)

            elapsed = time.time() - epoch_start
            current_lr = self.optimizer.param_groups[0]['lr']
            print(
                f"  Epoch {epoch + 1}/{epochs} | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_metrics['loss']:.4f} | "
                f"val_acc={val_metrics['acc']:.4f} | "
                f"lr={current_lr:.2e} | "
                f"time={elapsed:.1f}s"
            )

            if self.writer is not None:
                self.writer.add_scalar('train/loss', train_loss, epoch)
                self.writer.add_scalar('val/loss', val_metrics['loss'], epoch)
                self.writer.add_scalar('val/acc', val_metrics['acc'], epoch)
                self.writer.add_scalar('lr', current_lr, epoch)

            if self.use_wandb:
                _wandb.log({
                    'train/loss': train_loss,
                    'val/loss': val_metrics['loss'],
                    'val/acc': val_metrics['acc'],
                    'lr': current_lr,
                    'epoch': epoch,
                })

            if self.epochs_without_improvement >= patience:
                print(f"  Early stopping at epoch {epoch + 1} (patience={patience})")
                break

        # Load best checkpoint (if one was saved) before final test evaluation.
        best_ckpt_path = self.checkpoint_dir / 'best.pth'
        if best_ckpt_path.exists():
            self.load_checkpoint(str(best_ckpt_path))

        test_metrics = self.evaluate(self.test_loader)

        total_time = time.time() - start_time
        print(
            f"  Training complete in {total_time:.1f}s | "
            f"best_val_acc={self.best_val_acc:.4f} | "
            f"test_acc={test_metrics['acc']:.4f}"
        )

        if self.writer is not None:
            self.writer.close()
        if self.use_wandb:
            _wandb.finish()

        return {
            'best_val_acc': self.best_val_acc,
            'test_metrics': test_metrics,
            'history': self.history,
        }

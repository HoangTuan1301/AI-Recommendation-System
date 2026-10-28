"""Implicit Neural Collaborative Filtering (NeuMF) Recommender.

Implements the NeuMF deep learning architecture (He et al., WWW 2017)
combining Generalized Matrix Factorization (GMF) and Multi-Layer Perceptron (MLP)
trained on implicit song feedback with BCEWithLogitsLoss and negative sampling.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union
import logging
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import joblib

from src.music.models.base_music_recommender import BaseMusicRecommender

logger = logging.getLogger(__name__)


class ImplicitInteractionsDataset(Dataset):
    """PyTorch Dataset for implicit user-song pairs with negative sampling."""

    def __init__(self, user_indices: np.ndarray, song_indices: np.ndarray, labels: np.ndarray) -> None:
        self.users = torch.tensor(user_indices, dtype=torch.long)
        self.songs = torch.tensor(song_indices, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.users[idx], self.songs[idx], self.labels[idx]


class NeuMFNet(nn.Module):
    """Dual-branch Neural Matrix Factorization Network (GMF + MLP)."""

    def __init__(
        self,
        num_users: int,
        num_songs: int,
        gmf_dim: int = 32,
        mlp_dim: int = 32,
        mlp_layers: Sequence[int] = (64, 32, 16),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        # GMF Branch
        self.user_embed_gmf = nn.Embedding(num_users, gmf_dim)
        self.song_embed_gmf = nn.Embedding(num_songs, gmf_dim)

        # MLP Branch
        self.user_embed_mlp = nn.Embedding(num_users, mlp_dim)
        self.song_embed_mlp = nn.Embedding(num_songs, mlp_dim)

        mlp_modules: List[nn.Module] = []
        in_features = mlp_dim * 2
        for out_features in mlp_layers:
            mlp_modules.append(nn.Linear(in_features, out_features))
            mlp_modules.append(nn.BatchNorm1d(out_features))
            mlp_modules.append(nn.ReLU())
            mlp_modules.append(nn.Dropout(p=dropout))
            in_features = out_features
        self.mlp_stack = nn.Sequential(*mlp_modules)

        # Prediction Layer
        final_in = gmf_dim + mlp_layers[-1]
        self.prediction_layer = nn.Linear(final_in, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize embedding and linear layer weights with Xavier."""
        nn.init.xavier_uniform_(self.user_embed_gmf.weight)
        nn.init.xavier_uniform_(self.song_embed_gmf.weight)
        nn.init.xavier_uniform_(self.user_embed_mlp.weight)
        nn.init.xavier_uniform_(self.song_embed_mlp.weight)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, user_indices: torch.Tensor, song_indices: torch.Tensor) -> torch.Tensor:
        # GMF Branch
        u_gmf = self.user_embed_gmf(user_indices)
        s_gmf = self.song_embed_gmf(song_indices)
        phi_gmf = u_gmf * s_gmf

        # MLP Branch
        u_mlp = self.user_embed_mlp(user_indices)
        s_mlp = self.song_embed_mlp(song_indices)
        phi_mlp = torch.cat([u_mlp, s_mlp], dim=-1)
        phi_mlp = self.mlp_stack(phi_mlp)

        # Fusion
        fusion = torch.cat([phi_gmf, phi_mlp], dim=-1)
        logits = self.prediction_layer(fusion)
        return logits.squeeze(-1)


class ImplicitNeuralCFRecommender(BaseMusicRecommender):
    """PyTorch-based NeuMF model for implicit song recommendation."""

    def __init__(
        self,
        name: str = "Implicit Neural CF",
        gmf_dim: int = 32,
        mlp_dim: int = 32,
        mlp_layers: Sequence[int] = (64, 32, 16),
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5,
        num_negatives: int = 4,
        batch_size: int = 1024,
        max_epochs: int = 15,
        early_stopping_patience: int = 3,
        random_seed: int = 42,
        device: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name=name,
            gmf_dim=gmf_dim,
            mlp_dim=mlp_dim,
            mlp_layers=list(mlp_layers),
            dropout=dropout,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            num_negatives=num_negatives,
            batch_size=batch_size,
            max_epochs=max_epochs,
            early_stopping_patience=early_stopping_patience,
            random_seed=random_seed,
            **kwargs,
        )
        self.gmf_dim = gmf_dim
        self.mlp_dim = mlp_dim
        self.mlp_layers = list(mlp_layers)
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_negatives = num_negatives
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.early_stopping_patience = early_stopping_patience
        self.random_seed = random_seed
        self.device_str = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.net: Optional[NeuMFNet] = None
        self.user_to_idx: Dict[str, int] = {}
        self.idx_to_user: Dict[int, str] = {}
        self.song_to_idx: Dict[str, int] = {}
        self.idx_to_song: Dict[int, str] = {}
        self.training_losses: List[float] = []

    def _generate_training_data(
        self,
        u_arr: np.ndarray,
        i_arr: np.ndarray,
        user_positives: Dict[int, Set[int]],
        num_songs: int,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate balanced positive and negative pairs for one epoch."""
        n_pos = len(u_arr)
        total_samples = n_pos * (1 + self.num_negatives)

        users_out = np.zeros(total_samples, dtype=np.int32)
        songs_out = np.zeros(total_samples, dtype=np.int32)
        labels_out = np.zeros(total_samples, dtype=np.float32)

        # Positive instances
        users_out[:n_pos] = u_arr
        songs_out[:n_pos] = i_arr
        labels_out[:n_pos] = 1.0

        # Negative instances
        offset = n_pos
        for u, _ in zip(u_arr, i_arr):
            pos_set = user_positives[u]
            for _ in range(self.num_negatives):
                neg_song = rng.integers(0, num_songs)
                while neg_song in pos_set:
                    neg_song = rng.integers(0, num_songs)
                users_out[offset] = u
                songs_out[offset] = neg_song
                labels_out[offset] = 0.0
                offset += 1

        # Shuffle
        perm = rng.permutation(total_samples)
        return users_out[perm], songs_out[perm], labels_out[perm]

    def fit(self, train_df: pd.DataFrame, val_evaluator: Optional[Any] = None) -> "ImplicitNeuralCFRecommender":
        """Train NeuMF network with mini-batch SGD and negative sampling."""
        self._init_train_state(train_df)
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
        start_time = time.time()

        unique_users = sorted(train_df["user_id"].unique())
        unique_songs = self.all_train_items

        self.user_to_idx = {u: i for i, u in enumerate(unique_users)}
        self.idx_to_user = {i: u for i, u in enumerate(unique_users)}
        self.song_to_idx = {s: i for i, s in enumerate(unique_songs)}
        self.idx_to_song = {i: s for i, s in enumerate(unique_songs)}

        num_users = len(unique_users)
        num_songs = len(unique_songs)

        u_arr = np.array([self.user_to_idx[u] for u in train_df["user_id"]], dtype=np.int32)
        i_arr = np.array([self.song_to_idx[s] for s in train_df["song_id"]], dtype=np.int32)

        user_positives: Dict[int, Set[int]] = {u: set() for u in range(num_users)}
        for u, i in zip(u_arr, i_arr):
            user_positives[u].add(i)

        device = torch.device(self.device_str)
        self.net = NeuMFNet(
            num_users=num_users,
            num_songs=num_songs,
            gmf_dim=self.gmf_dim,
            mlp_dim=self.mlp_dim,
            mlp_layers=self.mlp_layers,
            dropout=self.dropout,
        ).to(device)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        logger.info(
            f"Starting NeuMF training on {self.device_str}: {num_users:,} users, {num_songs:,} songs, "
            f"epochs={self.max_epochs}, batch_size={self.batch_size}, num_negatives={self.num_negatives}..."
        )

        rng = np.random.default_rng(self.random_seed)
        best_val_ndcg = -1.0
        best_state = None
        patience_counter = 0

        for epoch in range(1, self.max_epochs + 1):
            epoch_start = time.time()
            self.net.train()

            # Generate fresh negative samples per epoch
            u_data, s_data, y_data = self._generate_training_data(u_arr, i_arr, user_positives, num_songs, rng)
            dataset = ImplicitInteractionsDataset(u_data, s_data, y_data)
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)

            total_loss = 0.0
            for batch_u, batch_s, batch_y in dataloader:
                batch_u = batch_u.to(device)
                batch_s = batch_s.to(device)
                batch_y = batch_y.to(device)

                optimizer.zero_grad()
                logits = self.net(batch_u, batch_s)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * len(batch_y)

            avg_loss = total_loss / len(dataset)
            self.training_losses.append(avg_loss)
            epoch_time = time.time() - epoch_start

            # Validation checkpoint tracking if evaluator provided
            val_info = ""
            if val_evaluator is not None and epoch % 2 == 0:
                self.is_fitted = True
                val_metrics = val_evaluator.evaluate(self)
                val_ndcg = val_metrics.get("NDCG@10", 0.0)
                val_info = f" | Val NDCG@10: {val_ndcg:.4f}"

                if val_ndcg > best_val_ndcg:
                    best_val_ndcg = val_ndcg
                    best_state = {k: v.cpu().clone() for k, v in self.net.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.early_stopping_patience:
                        logger.info(f"Early stopping triggered at epoch {epoch}.")
                        break

            logger.info(f"NeuMF Epoch {epoch:02d}/{self.max_epochs} - BCE Loss: {avg_loss:.4f}{val_info} ({epoch_time:.2f}s)")

        # Restore best weights if saved
        if best_state is not None:
            self.net.load_state_dict(best_state)
            logger.info(f"Restored best NeuMF checkpoint with Val NDCG@10: {best_val_ndcg:.4f}")

        elapsed = time.time() - start_time
        self.is_fitted = True
        logger.info(f"Fitted {self.name} in {elapsed:.2f}s. Final training loss: {self.training_losses[-1]:.4f}")
        return self

    def score(self, user_id: str, song_ids: Sequence[str]) -> np.ndarray:
        """Compute NeuMF predicted probabilities for candidate songs."""
        if not self.is_fitted or self.net is None:
            raise RuntimeError(f"Model '{self.name}' must be fitted before calling score().")

        if user_id not in self.user_to_idx:
            return np.zeros(len(song_ids), dtype=np.float32)

        u_idx = self.user_to_idx[user_id]
        valid_indices = []
        valid_pos = []
        for idx, song_id in enumerate(song_ids):
            if song_id in self.song_to_idx:
                valid_indices.append(self.song_to_idx[song_id])
                valid_pos.append(idx)

        scores = np.zeros(len(song_ids), dtype=np.float32)
        if not valid_indices:
            return scores

        device = torch.device(self.device_str)
        self.net.eval()
        with torch.no_grad():
            u_tensor = torch.full((len(valid_indices),), u_idx, dtype=torch.long, device=device)
            s_tensor = torch.tensor(valid_indices, dtype=torch.long, device=device)
            logits = self.net(u_tensor, s_tensor)
            preds = torch.sigmoid(logits).cpu().numpy()

        for pos, p in zip(valid_pos, preds):
            scores[pos] = float(p)

        return scores

    def save(self, path: Union[str, Path]) -> None:
        """Serialize NeuMF model instance to disk."""
        target_path = Path(path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        # Ensure model is on CPU before saving for cross-device portability
        if self.net is not None:
            self.net.to(torch.device("cpu"))
            self.device_str = "cpu"
        joblib.dump(self, target_path)
        logger.info(f"Saved NeuMF model checkpoint to: {target_path}")

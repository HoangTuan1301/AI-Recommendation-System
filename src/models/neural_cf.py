"""Neural Collaborative Filtering (NeuMF) Recommender.

This module implements the Neural Matrix Factorization (NeuMF) architecture proposed by
Xiangnan He et al. (WWW 2017: "Neural Collaborative Filtering").

Architecture:
    1. Generalized Matrix Factorization (GMF):
       \\phi^{GMF} = \\mathbf{p}_u^G \\odot \\mathbf{q}_i^G
    2. Multi-Layer Perceptron (MLP):
       \\phi^{MLP} = a_L(\\mathbf{W}_L^T (\\dots a_1(\\mathbf{W}_1^T [\\mathbf{p}_u^M, \\mathbf{q}_i^M] + \\mathbf{b}_1) \\dots) + \\mathbf{b}_L)
    3. NeuMF Fusion Layer:
       \\hat{y}_{ui} = \\sigma(\\mathbf{h}^T [\\phi^{GMF}, \\phi^{MLP}])

Training Protocol:
    - Positive interactions: Training interactions with rating >= relevance_threshold (default 4.0).
    - Negative sampling: Uniform sampling from unobserved items (I_all \\setminus I_u^{train}) for each user.
      Negative items are never drawn from observed training interactions or test interactions.
    - Loss: Binary Cross-Entropy with Logits (BCEWithLogitsLoss).
"""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from typing import Dict, List, Optional, Sequence, Union
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from src.models.base_recommender import BaseRecommender

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class NeuMFNet(nn.Module):
    """PyTorch implementation of the Neural Matrix Factorization (NeuMF) network."""

    def __init__(
        self,
        n_users: int,
        n_items: int,
        n_factors_gmf: int = 16,
        n_factors_mlp: int = 32,
        layers: Sequence[int] = (64, 32, 16),
        dropout: float = 0.1,
    ) -> None:
        """Initialize NeuMF layers.

        Args:
            n_users: Number of unique users.
            n_items: Number of unique items.
            n_factors_gmf: Embedding dimension for GMF branch (default 16).
            n_factors_mlp: Embedding dimension for MLP branch (default 32).
            layers: List of hidden layer dimensions for MLP (default [64, 32, 16]).
            dropout: Dropout probability between MLP layers (default 0.1).
        """
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors_gmf = n_factors_gmf
        self.n_factors_mlp = n_factors_mlp

        # 1. GMF Embeddings
        self.user_embed_gmf = nn.Embedding(n_users, n_factors_gmf)
        self.item_embed_gmf = nn.Embedding(n_items, n_factors_gmf)

        # 2. MLP Embeddings
        self.user_embed_mlp = nn.Embedding(n_users, n_factors_mlp)
        self.item_embed_mlp = nn.Embedding(n_items, n_factors_mlp)

        # 3. MLP Hidden Layers
        mlp_modules = []
        in_dim = n_factors_mlp * 2
        for out_dim in layers:
            mlp_modules.append(nn.Linear(in_dim, out_dim))
            mlp_modules.append(nn.ReLU())
            if dropout > 0:
                mlp_modules.append(nn.Dropout(p=dropout))
            in_dim = out_dim
        self.mlp = nn.Sequential(*mlp_modules)

        # 4. NeuMF Fusion Projection Layer
        final_in_dim = n_factors_gmf + (layers[-1] if len(layers) > 0 else in_dim)
        self.affine_output = nn.Linear(final_in_dim, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize embedding and linear layer weights with normal distribution."""
        nn.init.normal_(self.user_embed_gmf.weight, std=0.01)
        nn.init.normal_(self.item_embed_gmf.weight, std=0.01)
        nn.init.normal_(self.user_embed_mlp.weight, std=0.01)
        nn.init.normal_(self.item_embed_mlp.weight, std=0.01)

        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

        nn.init.kaiming_uniform_(self.affine_output.weight, a=1, nonlinearity="sigmoid")
        nn.init.zeros_(self.affine_output.bias)

    def forward(self, user_indices: torch.Tensor, item_indices: torch.Tensor) -> torch.Tensor:
        """Forward pass to compute unnormalized interaction logits.

        Args:
            user_indices: Tensor of user index IDs (batch_size,).
            item_indices: Tensor of item index IDs (batch_size,).

        Returns:
            torch.Tensor: Interaction logits of shape (batch_size,).
        """
        # GMF branch
        p_gmf = self.user_embed_gmf(user_indices)
        q_gmf = self.item_embed_gmf(item_indices)
        phi_gmf = p_gmf * q_gmf

        # MLP branch
        p_mlp = self.user_embed_mlp(user_indices)
        q_mlp = self.item_embed_mlp(item_indices)
        phi_mlp = self.mlp(torch.cat([p_mlp, q_mlp], dim=-1))

        # Fusion concatenation and projection
        fusion = torch.cat([phi_gmf, phi_mlp], dim=-1)
        logits = self.affine_output(fusion)
        return logits.squeeze(-1)


class NeuralCFRecommender(BaseRecommender):
    """Neural Matrix Factorization Recommender complying with BaseRecommender."""

    def __init__(
        self,
        n_factors_gmf: int = 16,
        n_factors_mlp: int = 32,
        layers: Sequence[int] = (64, 32, 16),
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
        neg_ratio: int = 4,
        n_epochs: int = 15,
        batch_size: int = 512,
        dropout: float = 0.1,
        relevance_threshold: float = 4.0,
        random_state: int = 42,
        name: str = "Neural CF",
    ) -> None:
        """Initialize the Neural Collaborative Filtering Recommender.

        Args:
            n_factors_gmf: GMF latent dimension (default 16).
            n_factors_mlp: MLP embedding dimension (default 32).
            layers: MLP hidden layer dimensions (default (64, 32, 16)).
            learning_rate: Adam optimizer learning rate (default 0.001).
            weight_decay: L2 regularization penalty (default 1e-4).
            neg_ratio: Number of negative samples per positive interaction (default 4).
            n_epochs: Maximum training epochs (default 15).
            batch_size: Mini-batch size for SGD (default 512).
            dropout: Dropout probability in MLP layers (default 0.1).
            relevance_threshold: Rating threshold to define positive interactions (default 4.0).
            random_state: Random seed for deterministic reproducibility (default 42).
            name: Model identifier name.
        """
        super().__init__(name=name)
        self.n_factors_gmf = int(n_factors_gmf)
        self.n_factors_mlp = int(n_factors_mlp)
        self.layers = list(layers)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.neg_ratio = int(neg_ratio)
        self.n_epochs = int(n_epochs)
        self.batch_size = int(batch_size)
        self.dropout = float(dropout)
        self.relevance_threshold = float(relevance_threshold)
        self.random_state = int(random_state)

        # Internal index mappings and model handle
        self.user_to_idx: Dict[int, int] = {}
        self.idx_to_user: Dict[int, int] = {}
        self.item_to_idx: Dict[int, int] = {}
        self.idx_to_item: Dict[int, int] = {}
        self.net: Optional[NeuMFNet] = None

        # Training history logs
        self.train_loss_history: List[float] = []

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None,
    ) -> "NeuralCFRecommender":
        """Fit NeuMF strictly using training data and negative sampling.

        Args:
            train_df: DataFrame containing ['user_id', 'movie_id', 'rating'].
            val_df: Optional validation DataFrame for monitoring loss.

        Returns:
            self: Fitted recommender instance.
        """
        self._store_train_metadata(train_df)
        self._set_seeds(self.random_state)
        rng = np.random.RandomState(self.random_state)

        unique_users = sorted(train_df["user_id"].unique())
        unique_items = sorted(train_df["movie_id"].unique())
        self.user_to_idx = {u: idx for idx, u in enumerate(unique_users)}
        self.idx_to_user = {idx: u for idx, u in enumerate(unique_users)}
        self.item_to_idx = {i: idx for idx, i in enumerate(unique_items)}
        self.idx_to_item = {idx: i for idx, i in enumerate(unique_items)}

        n_users = len(unique_users)
        n_items = len(unique_items)
        all_item_indices = np.arange(n_items)

        # Map user train history to internal item indices
        user_seen_indices: Dict[int, set] = {}
        for u in unique_users:
            seen_mids = self.user_train_history.get(u, set())
            user_seen_indices[self.user_to_idx[u]] = {
                self.item_to_idx[m] for m in seen_mids if m in self.item_to_idx
            }

        # Filter positive training interactions
        pos_df = train_df[train_df["rating"] >= self.relevance_threshold]
        if len(pos_df) == 0:
            logger.warning("No interactions meet threshold %.1f; using all train ratings as positive.", self.relevance_threshold)
            pos_df = train_df

        pos_u = pos_df["user_id"].map(self.user_to_idx).values.astype(np.int64)
        pos_i = pos_df["movie_id"].map(self.item_to_idx).values.astype(np.int64)
        n_pos = len(pos_u)

        # Pre-generate unobserved candidate pools per user for fast negative sampling
        user_unseen_pools: Dict[int, np.ndarray] = {}
        for u_idx in range(n_users):
            seen_set = user_seen_indices[u_idx]
            unseen = np.array([i for i in all_item_indices if i not in seen_set], dtype=np.int64)
            user_unseen_pools[u_idx] = unseen if len(unseen) > 0 else all_item_indices

        logger.info(
            "Fitting %s (GMF=%d, MLP=%d, layers=%s, lr=%.4f, neg_ratio=%d, epochs=%d, seed=%d) on %d positives...",
            self.name,
            self.n_factors_gmf,
            self.n_factors_mlp,
            self.layers,
            self.learning_rate,
            self.neg_ratio,
            self.n_epochs,
            self.random_state,
            n_pos,
        )

        # Initialize network
        self.net = NeuMFNet(
            n_users=n_users,
            n_items=n_items,
            n_factors_gmf=self.n_factors_gmf,
            n_factors_mlp=self.n_factors_mlp,
            layers=self.layers,
            dropout=self.dropout,
        )
        self.net.train()

        optimizer = optim.Adam(
            self.net.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        criterion = nn.BCEWithLogitsLoss()

        self.train_loss_history = []

        for epoch in range(1, self.n_epochs + 1):
            # Negative sampling: neg_ratio samples per positive
            neg_items = []
            for u_idx in pos_u:
                pool = user_unseen_pools[u_idx]
                sampled = rng.choice(pool, size=self.neg_ratio, replace=False if len(pool) >= self.neg_ratio else True)
                neg_items.extend(sampled)

            neg_items_arr = np.array(neg_items, dtype=np.int64)
            neg_users_arr = np.repeat(pos_u, self.neg_ratio)

            epoch_users = np.concatenate([pos_u, neg_users_arr])
            epoch_items = np.concatenate([pos_i, neg_items_arr])
            epoch_labels = np.concatenate([np.ones(n_pos, dtype=np.float32), np.zeros(len(neg_users_arr), dtype=np.float32)])

            # Shuffle epoch dataset
            shuffle_perm = rng.permutation(len(epoch_users))
            epoch_users = epoch_users[shuffle_perm]
            epoch_items = epoch_items[shuffle_perm]
            epoch_labels = epoch_labels[shuffle_perm]

            u_tensor = torch.from_numpy(epoch_users)
            i_tensor = torch.from_numpy(epoch_items)
            y_tensor = torch.from_numpy(epoch_labels)

            total_loss = 0.0
            n_total = len(epoch_users)

            for start in range(0, n_total, self.batch_size):
                end = min(start + self.batch_size, n_total)
                b_u = u_tensor[start:end]
                b_i = i_tensor[start:end]
                b_y = y_tensor[start:end]

                optimizer.zero_grad()
                logits = self.net(b_u, b_i)
                loss = criterion(logits, b_y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * (end - start)

            epoch_loss = total_loss / n_total
            self.train_loss_history.append(epoch_loss)

            if epoch % 5 == 0 or epoch == self.n_epochs:
                logger.info("Epoch %2d/%2d - Training Loss: %.4f", epoch, self.n_epochs, epoch_loss)

        self.net.eval()
        self.is_fitted = True
        logger.info("Fitted %s successfully across %d users and %d items.", self.name, n_users, n_items)
        return self

    def _set_seeds(self, seed: int) -> None:
        """Set deterministic seeds across PyTorch and NumPy."""
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

    def predict(self, user_id: int, movie_id: int) -> float:
        """Predict the interaction likelihood for a user-item pair.

        IMPORTANT:
            Unlike rating prediction models (which output scores on a 1.0 to 5.0 scale),
            NeuralCF is an implicit ranking model trained with BCEWithLogitsLoss.
            This method returns the predicted interaction probability:
                \\sigma(\\hat{y}_{logits}) \\in [0.0, 1.0]
            where higher values denote stronger preference likelihood.

        Args:
            user_id: Target user ID.
            movie_id: Target movie ID.

        Returns:
            float: Predicted interaction probability in [0.0, 1.0].
        """
        if not self.is_fitted or self.net is None:
            raise RuntimeError(f"{self.name} is not fitted yet.")

        u_idx = self.user_to_idx.get(user_id)
        i_idx = self.item_to_idx.get(movie_id)

        if u_idx is None or i_idx is None:
            return 0.5  # Neutral prior for cold user/item

        u_t = torch.tensor([u_idx], dtype=torch.long)
        i_t = torch.tensor([i_idx], dtype=torch.long)

        with torch.no_grad():
            logit = self.net(u_t, i_t).item()
            prob = float(torch.sigmoid(torch.tensor(logit)).item())
        return prob

    def recommend(
        self,
        user_id: int,
        top_n: int = 10,
        candidate_items: Optional[Sequence[int]] = None,
    ) -> List[int]:
        """Generate top_n movie recommendations using NeuMF interaction scores.

        Args:
            user_id: Target user ID.
            top_n: Number of recommendations (default 10).
            candidate_items: Candidates to rank. If None, all unseen catalog items.

        Returns:
            List[int]: Top-N recommended movie IDs.
        """
        if not self.is_fitted or self.net is None:
            raise RuntimeError(f"{self.name} is not fitted yet.")

        if candidate_items is None:
            candidate_items = self._get_default_candidates(user_id)

        candidate_list = list(candidate_items)
        if not candidate_list:
            return []

        u_idx = self.user_to_idx.get(user_id)
        if u_idx is None:
            # Fallback for completely cold user: preserve candidate order
            return [int(mid) for mid in candidate_list[:top_n]]

        known_indices = []
        known_cand_ids = []
        unknown_cand_ids = []

        for cid in candidate_list:
            idx = self.item_to_idx.get(cid)
            if idx is not None:
                known_indices.append(idx)
                known_cand_ids.append(cid)
            else:
                unknown_cand_ids.append(cid)

        scores_map: Dict[int, float] = {}

        if known_indices:
            u_tensor = torch.full((len(known_indices),), u_idx, dtype=torch.long)
            i_tensor = torch.tensor(known_indices, dtype=torch.long)

            with torch.no_grad():
                logits = self.net(u_tensor, i_tensor).numpy()

            for cid, sc in zip(known_cand_ids, logits):
                scores_map[cid] = float(sc)

        # Unknown items receive negative infinity to rank below valid candidates
        for cid in unknown_cand_ids:
            scores_map[cid] = -1e9

        ranked = sorted(candidate_list, key=lambda mid: (scores_map[mid], -int(mid)), reverse=True)
        return [int(mid) for mid in ranked[:top_n]]

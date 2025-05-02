"""TabSyn module for tabular data synthesis."""

import os
import numpy as np
import torch
import pickle

from .model import MLPDiffusion, Model
from .vae.model import Model_VAE, Encoder_model, Decoder_model
from .data_utils import TabularDataset, preprocess
from .latent_utils import get_input_generate, split_num_cat_target, recover_data


class TabSyn:
    """TabSyn Synthesizer.
    
    Combines VAE and diffusion model to generate high-quality tabular data samples.
    
    Args:
        embedding_dim (int): 
            Dimension of the latent space. Defaults to 4.
        vae_layers (int): 
            Number of layers in the VAE model. Defaults to 2.
        vae_factor (int): 
            Expansion factor for hidden dimensions in VAE. Defaults to 32.
        vae_lr (float): 
            Learning rate for VAE training. Defaults to 1e-3.
        max_beta (float): 
            Maximum beta value for KL annealing. Defaults to 1e-2.
        min_beta (float): 
            Minimum beta value for KL annealing. Defaults to 1e-5.
        beta_decay (float): 
            Decay factor for beta annealing. Defaults to 0.7.
        diffusion_lr (float): 
            Learning rate for diffusion model. Defaults to 1e-4.
        vae_epochs (int): 
            Number of epochs to train VAE model. Defaults to 10.
        diffusion_epochs (int): 
            Number of epochs to train diffusion model. Defaults to 10.
        batch_size (int): 
            Batch size for training. Defaults to 4096.
        steps (int): 
            Number of function evaluations for sampling. Defaults to 50.
        device (str): 
            Device to use for training ('cpu', 'cuda'). 
            If None, will use 'cuda' if available. Defaults to None.
        verbose (bool): 
            Whether to display verbose output. Defaults to False.
    """

    def __init__(
        self,
        embedding_dim=4,
        vae_layers=2, 
        vae_factor=32,
        vae_lr=1e-3,
        max_beta=1e-2,
        min_beta=1e-5,
        beta_decay=0.7,
        diffusion_lr=1e-4,
        vae_epochs=10,
        diffusion_epochs=10,
        batch_size=4096,
        steps=50,
        device=None,
        verbose=False,
    ):
        self.embedding_dim = embedding_dim
        self.vae_layers = vae_layers
        self.vae_factor = vae_factor
        self.vae_lr = vae_lr
        self.max_beta = max_beta
        self.min_beta = min_beta
        self.beta_decay = beta_decay
        self.diffusion_lr = diffusion_lr
        self.vae_epochs = vae_epochs
        self.diffusion_epochs = diffusion_epochs
        self.batch_size = batch_size
        self.steps = steps
        self.verbose = verbose
        
        # Set device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        # Initialize models
        self.vae_model = None
        self.diffusion_model = None
        self.encoder = None
        self.decoder = None
        
        # Store dataset info
        self.info = None
        self.categories = None
        self.d_numerical = None
        self.num_inverse = None
        self.cat_inverse = None
        
        # Paths
        self.model_dir = None
    def _train_vae(self, train_data, info, ckpt_dir):
        """Train the VAE component of TabSyn."""
        if self.verbose:
            print("Training VAE model...")
        
        X_num, X_cat, self.categories, self.d_numerical = preprocess(
            train_data, task_type=info['task_type']
        )
        
        X_train_num, _ = X_num
        X_train_cat, _ = X_cat
        
        # Keep data on CPU for the dataset
        X_train_num = torch.tensor(X_train_num).float()
        X_train_cat = torch.tensor(X_train_cat)
        
        train_data = TabularDataset(X_train_num, X_train_cat)
        
        # Reduce number of workers if CUDA issues occur
        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,  # Set to 0 to avoid CUDA issues with multiprocessing
            pin_memory=True,  # Enable pin_memory for faster data transfer to GPU
        )   
        
        # Initialize VAE model
        self.vae_model = Model_VAE(
            self.vae_layers, 
            self.d_numerical, 
            self.categories, 
            self.embedding_dim, 
            n_head=1, 
            factor=self.vae_factor
        ).to(self.device)
        
        # Initialize encoder and decoder
        self.encoder = Encoder_model(
            self.vae_layers, 
            self.d_numerical, 
            self.categories, 
            self.embedding_dim, 
            n_head=1, 
            factor=self.vae_factor
        ).to(self.device)
        
        self.decoder = Decoder_model(
            self.vae_layers, 
            self.d_numerical, 
            self.categories, 
            self.embedding_dim, 
            n_head=1, 
            factor=self.vae_factor
        ).to(self.device)
        
        # Set up optimizer
        optimizer = torch.optim.Adam(
            self.vae_model.parameters(), 
            lr=self.vae_lr
        )
        
        # Training loop
        beta = self.max_beta
        for epoch in range(self.vae_epochs):
            self.vae_model.train()
            
            if self.verbose:
                print(f"VAE Epoch {epoch+1}/{self.vae_epochs}")
            
            curr_loss_multi = 0.0
            curr_loss_gauss = 0.0
            curr_loss_kl = 0.0
            curr_count = 0
                
            for batch_num, batch_cat in train_loader:
                optimizer.zero_grad()
                

                batch_num = batch_num.to(self.device)
                batch_cat = batch_cat.to(self.device)
                
                Recon_X_num, Recon_X_cat, mu_z, std_z = self.vae_model(batch_num, batch_cat)
                
                # Compute loss
                mse_loss, ce_loss, kl_loss, _ = self._compute_vae_loss(
                    batch_num, batch_cat, Recon_X_num, Recon_X_cat, mu_z, std_z
                )
                
                # Total loss
                loss = mse_loss + ce_loss + beta * kl_loss
                loss.backward()
                optimizer.step()
                
                # Update loss statistics
                batch_length = batch_num.shape[0]
                curr_count += batch_length
                curr_loss_multi += ce_loss.item() * batch_length
                curr_loss_gauss += mse_loss.item() * batch_length
                curr_loss_kl += kl_loss.item() * batch_length

            
            # Calculate average losses for the epoch
            num_loss = curr_loss_gauss / curr_count
            cat_loss = curr_loss_multi / curr_count
            kl_loss_val = curr_loss_kl / curr_count
            total_loss = num_loss + cat_loss + beta * kl_loss_val
            
            if self.verbose:
                print(f"epoch: {epoch}, beta = {beta:.6f}, Train MSE: {num_loss:.6f}, Train CE:{cat_loss:.6f}, Train KL:{kl_loss_val:.6f}, Train Loss: {total_loss:.6f}")
            
            # Decay beta every 5 epochs if it's above the minimum
            if epoch % 5 == 0 and epoch > 0 and beta > self.min_beta:
                beta = beta * self.beta_decay

        # Save encoder and decoder
        self.encoder.load_weights(self.vae_model)
        self.decoder.load_weights(self.vae_model)
        
        torch.save(self.encoder.state_dict(), os.path.join(ckpt_dir, 'encoder.pt'))
        torch.save(self.decoder.state_dict(), os.path.join(ckpt_dir, 'decoder.pt'))

        # Free CUDA memory before generating embeddings
        torch.cuda.empty_cache()
        
        # Generate and save embeddings
        with torch.no_grad():
            # Move data to the same device as the encoder
            X_train_num_device = X_train_num.to(self.device)
            X_train_cat_device = X_train_cat.to(self.device)
            
            # Process in batches to avoid OOM errors
            batch_size = 1024
            all_embeddings = []
            
            for i in range(0, len(X_train_num), batch_size):
                end_idx = min(i + batch_size, len(X_train_num))
                batch_num = X_train_num_device[i:end_idx]
                batch_cat = X_train_cat_device[i:end_idx]
                
                # Get embeddings for this batch
                batch_z = self.encoder(batch_num, batch_cat).detach().cpu()
                all_embeddings.append(batch_z)
                
                # Free up GPU memory
                torch.cuda.empty_cache()
            
            # Concatenate all batch embeddings
            train_z = torch.cat(all_embeddings, dim=0).numpy()
        
        np.save(os.path.join(ckpt_dir, 'train_z.npy'), train_z)
        
        if self.verbose:
            print("VAE training complete.")
    
    def _compute_vae_loss(self, X_num, X_cat, Recon_X_num, Recon_X_cat, mu_z, logvar_z):
        """Compute VAE loss components."""
        ce_loss_fn = torch.nn.CrossEntropyLoss()
        mse_loss = (X_num - Recon_X_num).pow(2).mean()
        ce_loss = 0
        acc = 0
        total_num = 0
        
        for idx, x_cat in enumerate(Recon_X_cat):
            if x_cat is not None:
                ce_loss += ce_loss_fn(x_cat, X_cat[:, idx])
                x_hat = x_cat.argmax(dim=-1)
                acc += (x_hat == X_cat[:, idx]).float().sum()
                total_num += x_hat.shape[0]
        
        ce_loss /= (idx + 1)
        acc /= total_num
        
        # KL divergence
        temp = 1 + logvar_z - mu_z.pow(2) - logvar_z.exp()
        loss_kld = -0.5 * torch.mean(temp.mean(-1).mean())
        
        return mse_loss, ce_loss, loss_kld, acc
    
    def _train_diffusion(self, train_data_dir, ckpt_dir):
        """Train the diffusion component of TabSyn."""
        if self.verbose:
            print("Training diffusion model...")
        
        # Load embeddings
        train_z = torch.tensor(np.load(os.path.join(ckpt_dir, 'train_z.npy'))).float()
        
        # Process embeddings
        train_z = train_z[:, 1:, :]
        B, num_tokens, token_dim = train_z.size()
        in_dim = num_tokens * token_dim
        train_z = train_z.view(B, in_dim)
        
        # Center the data
        mean = train_z.mean(0)
        train_z = (train_z - mean) / 2
        
        # Create data loader
        train_loader = torch.utils.data.DataLoader(
            train_z,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
        )
        
        # Initialize diffusion model
        denoise_fn = MLPDiffusion(in_dim, 1024).to(self.device)
        self.diffusion_model = Model(
            denoise_fn=denoise_fn, 
            hid_dim=train_z.shape[1]
        ).to(self.device)
        
        # Set up optimizer
        optimizer = torch.optim.Adam(
            self.diffusion_model.parameters(), 
            lr=self.diffusion_lr
        )
        
        # Training loop
        best_loss = float('inf')
        patience = 0
        
        for epoch in range(self.diffusion_epochs):
            self.diffusion_model.train()
            
            if self.verbose:
                print(f"Diffusion Epoch {epoch+1}/{self.diffusion_epochs}")
            
            batch_loss = 0.0
            len_input = 0
            
            for batch in train_loader:
                optimizer.zero_grad()
                
                inputs = batch.float().to(self.device)
                loss = self.diffusion_model(inputs).mean()
                
                batch_loss += loss.item() * len(inputs)
                len_input += len(inputs)
                
                loss.backward()
                optimizer.step()
            
            curr_loss = batch_loss / len_input

            if self.verbose:
                print(f"Epoch {epoch+1}/{self.diffusion_epochs}, Loss: {curr_loss:.6f}")
            
            if curr_loss < best_loss:
                best_loss = curr_loss
                patience = 0
                torch.save(self.diffusion_model.state_dict(), os.path.join(ckpt_dir, 'model.pt'))
            else:
                patience += 1
                if patience == 500:
                    if self.verbose:
                        print('Early stopping')
                    break
        
        if self.verbose:
            print("Diffusion model training complete.")
    

    def fit(self, data_path, task_type='regression'):
        """
        Fit the TabSyn model to the provided dataset.
        
        Args:
            data_path (str): 
                Path to the directory containing the dataset.
            task_type (str): 
                Type of task, 'regression' or 'classification'. 
                Defaults to 'regression'.
        """
        # Prepare data directory and create checkpoint directory
        if os.path.isdir(data_path):
            train_data_dir = data_path
        else:
            # If a direct file is provided, we need to preprocess it
            raise ValueError("Please provide a directory with preprocessed data")
        
        # Create checkpoint directory within the model directory
        model_dir = os.path.dirname(data_path)
        ckpt_dir = os.path.join(model_dir, 'ckpt')
        os.makedirs(ckpt_dir, exist_ok=True)
        
        # Store paths
        self.model_dir = model_dir
        
        # Load and store info
        info_path = os.path.join(train_data_dir, 'info.json')
        if os.path.exists(info_path):
            import json
            with open(info_path, 'r') as f:
                self.info = json.load(f)
        else:
            self.info = {
                'task_type': task_type,
                'name': os.path.basename(data_path)
            }
        
        # Train VAE
        self._train_vae(train_data_dir, self.info, ckpt_dir)
        
        # Train diffusion model
        self._train_diffusion(train_data_dir, ckpt_dir)
        
        # Save model info
        self._save_model_info(ckpt_dir)
        
        if self.verbose:
            print("TabSyn model training complete.")

    def _save_model_info(self, ckpt_dir):
        """Save model parameters and info."""
        config = {
            'embedding_dim': self.embedding_dim,
            'vae_layers': self.vae_layers,
            'vae_factor': self.vae_factor,
            'categories': self.categories,
            'd_numerical': self.d_numerical,
            'info': self.info
        }
        
        with open(os.path.join(ckpt_dir, 'config.pkl'), 'wb') as f:
            pickle.dump(config, f)
    
    def sample(self, n_samples=None, steps=None):
        """
        Generate synthetic samples using the trained TabSyn model.
        
        Args:
            n_samples (int, optional):
                Number of samples to generate. If None, 
                matches the training data size. Defaults to None.
            steps (int, optional): 
                Number of function evaluations for sampling.
                If None, uses the value from initialization. Defaults to None.
                
        Returns:
            pandas.DataFrame: The generated synthetic data.
        """
        if self.model_dir is None:
            raise ValueError("Model has not been trained or loaded yet")
        
        if steps is None:
            steps = self.steps
            
        # Load required components
        train_z, _, _, _, info, num_inverse, cat_inverse = get_input_generate(self.model_dir)
        
        if n_samples is None:
            n_samples = train_z.shape[0]
            
        in_dim = train_z.shape[1]
        mean = train_z.mean(0)
        
        # Load diffusion model if needed
        diffusion_model_path = os.path.join(self.model_dir, 'ckpt', 'model.pt')
        if self.diffusion_model is None:
            denoise_fn = MLPDiffusion(in_dim, 1024).to(self.device)
            self.diffusion_model = Model(
                denoise_fn=denoise_fn, 
                hid_dim=train_z.shape[1]
            ).to(self.device)
            self.diffusion_model.load_state_dict(
                torch.load(diffusion_model_path, map_location=self.device)
            )
        
        # Generate samples
        if self.verbose:
            print(f"Generating {n_samples} samples...")
            
        # Generate in chunks to avoid OOM
        chunk_size = 4096
        chunks = []
        
        for start in range(0, n_samples, chunk_size):
            cur = min(chunk_size, n_samples - start)
            with torch.no_grad():
                # Use the correct torch.amp.autocast syntax
                if self.device.startswith('cuda'):
                    with torch.amp.autocast('cuda', enabled=False):
                        x = self._sample_from_diffusion(
                            self.diffusion_model.denoise_fn_D, cur, in_dim, steps
                        )
                        x = x * 2 + mean.to(self.device)
                else:
                    x = self._sample_from_diffusion(
                        self.diffusion_model.denoise_fn_D, cur, in_dim, steps
                    )
                    x = x * 2 + mean.to(self.device)
                        
            chunks.append(x.cpu())
            torch.cuda.empty_cache()
            
        syn_data = torch.cat(chunks).numpy()
        
        # Convert to original data format
        syn_num, syn_cat, syn_target = split_num_cat_target(
            syn_data, info, num_inverse, cat_inverse, self.device
        )
        
        syn_df = recover_data(syn_num, syn_cat, syn_target, info)
        
        # Rename columns
        idx_name_mapping = info['idx_name_mapping']
        idx_name_mapping = {int(key): value for key, value in idx_name_mapping.items()}
        
        syn_df.rename(columns=idx_name_mapping, inplace=True)
        
        if self.verbose:
            print("Sample generation complete.")
            
        return syn_df
    
    def _sample_from_diffusion(self, model, num_samples, dim, num_steps=50):
        """Sample from the diffusion model."""
        import torch
        from .diffusion_utils import sample
        
        return sample(model, num_samples, dim, num_steps, device=self.device)
    
    def save(self, path):
        """
        Save the trained TabSyn model.
        
        Args:
            path (str): Path where to save the model.
        """
        if self.model_dir is None:
            raise ValueError("Model has not been trained or loaded yet")
            
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Create a dict with paths to model components
        model_info = {
            'model_dir': self.model_dir,
            'embedding_dim': self.embedding_dim,
            'vae_layers': self.vae_layers,
            'vae_factor': self.vae_factor,
            'categories': self.categories,
            'd_numerical': self.d_numerical,
            'info': self.info,
            'steps': self.steps,
            'device': self.device,
            'verbose': self.verbose
        }
        
        # Save the model info
        with open(path, 'wb') as f:
            pickle.dump(model_info, f)
            
        if self.verbose:
            print(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path):
        """
        Load a trained TabSyn model.
        
        Args:
            path (str): Path to the saved model.
            
        Returns:
            TabSyn: The loaded TabSyn model.
        """
        # Load model info
        with open(path, 'rb') as f:
            model_info = pickle.load(f)
            
        # Create a new TabSyn instance
        model = cls(
            embedding_dim=model_info['embedding_dim'],
            vae_layers=model_info['vae_layers'],
            vae_factor=model_info['vae_factor'],
            steps=model_info['steps'],
            device=model_info['device'],
            verbose=model_info['verbose']
        )
        
        # Set model attributes
        model.model_dir = model_info['model_dir']
        model.categories = model_info['categories']
        model.d_numerical = model_info['d_numerical']
        model.info = model_info['info']
        
        return model
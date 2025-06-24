"""
Patch for ACTIVA to enable CPU training
This modifies the model to skip data_parallel when running on CPU
"""

import torch
from ACTIVA.networks.model import ACTIVA

# Save original methods
original_encode = ACTIVA.encode
original_decode = ACTIVA.decode

def patched_encode(self, x):
    """Patched encode method that handles CPU mode"""
    if torch.cuda.is_available() and x.is_cuda:
        # Use original data_parallel for GPU
        return original_encode(self, x)
    else:
        # Direct call for CPU
        mu, variance = self.encoder(x)
        return mu, variance

def patched_decode(self, z):
    """Patched decode method that handles CPU mode"""
    if torch.cuda.is_available() and z.is_cuda:
        # Use original data_parallel for GPU
        return original_decode(self, z)
    else:
        # Direct call for CPU
        x_r = self.decoder(z)
        return x_r

# Apply patches
ACTIVA.encode = patched_encode
ACTIVA.decode = patched_decode

print("CPU patch applied successfully!")
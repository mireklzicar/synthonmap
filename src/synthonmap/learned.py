"""Optional reaction-aware coordinate model extracted from the map experiment.

UMAP is the teacher; this small network is a supervised coordinate approximation.
Once fitted, synthons are encoded once and product inference uses cache lookup.
"""
import torch
from torch import nn


class CachedReactionMap(nn.Module):
    def __init__(self, n_reactions, *, fingerprint_bits=2048, code_dim=64, width=256):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(fingerprint_bits, 128), nn.SiLU(),
                                     nn.Dropout(.05), nn.Linear(128, code_dim))
        self.reaction = nn.Embedding(n_reactions, 16)
        self.head = nn.Sequential(nn.Linear(4 * code_dim + 20, width), nn.SiLU(), nn.Dropout(.05),
                                  nn.Linear(width, width // 2), nn.SiLU(), nn.Linear(width // 2, 2))

    def from_cache(self, codes, slots, reactions):
        mask = slots.ne(0).to(codes.dtype)
        features = codes[slots] * mask[..., None]
        return self.head(torch.cat([features.flatten(1), self.reaction(reactions), mask], dim=1))

    def forward(self, fingerprints, slots, reactions):
        # Encode only the synthons needed for this training minibatch.
        ids, inverse = torch.unique(slots, return_inverse=True)
        codes = self.encoder(fingerprints[ids])
        features = codes[inverse.reshape(slots.shape)] * slots.ne(0)[..., None]
        return self.head(torch.cat([features.flatten(1), self.reaction(reactions), slots.ne(0).float()], dim=1))

    @torch.no_grad()
    def cache(self, fingerprints, batch_size=4096):
        if self.training:
            raise ValueError("call model.eval() before caching synthon codes")
        codes = torch.cat([self.encoder(batch) for batch in fingerprints.split(batch_size)])
        codes[0] = 0
        return codes

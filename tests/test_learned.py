import pytest


def test_cached_inference_matches_direct_and_ignores_padding():
    torch = pytest.importorskip("torch")
    from synthonmap.learned import CachedReactionMap
    torch.manual_seed(7)
    model = CachedReactionMap(2, fingerprint_bits=32).eval()
    fingerprints = torch.rand(10, 32)
    slots = torch.tensor([[1, 2, 0, 0], [3, 4, 5, 0]])
    routes = torch.tensor([0, 1])
    with torch.no_grad():
        direct = model(fingerprints, slots, routes)
        cache = model.cache(fingerprints)
        torch.testing.assert_close(model.from_cache(cache, slots, routes), direct)
        cache[0] = 10000
        torch.testing.assert_close(model.from_cache(cache, slots, routes), direct)

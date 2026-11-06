"""Freedom slice: product UMAP -> cached-synthon model -> streaming histogram.

This is a small reproducible demonstration, not the production-trained weights.
Uses only installed packages and synthonor's bundled example resources.
"""
import argparse
import copy
import csv
import hashlib
from importlib import metadata, resources
import io
import json
import os
from pathlib import Path
import time

# Avoid oversubscription in the small CPU example. Respect explicit user values.
os.environ.setdefault("NUMBA_NUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem, rdFingerprintGenerator
from scipy.spatial.distance import cdist
import torch
from threadpoolctl import threadpool_limits
from synthonor import example_space_path

from synthonmap import Histogram, fit_bounds, coarsen_counts, display_ceiling, save_png
from synthonmap.learned import CachedReactionMap
from synthonmap.recipes import RecipeSpace


def load_space():
    tsv = Path(example_space_path()).read_bytes()
    schemas = resources.files("synthonor.data").joinpath("synthon_space_1M.reactions.tsv").read_bytes()
    rows = list(csv.DictReader(io.StringIO(tsv.decode()), delimiter="\t"))
    definitions = {r["reaction_id"]: r for r in csv.DictReader(io.StringIO(schemas.decode()), delimiter="\t")}
    groups = {}
    for uid, row in enumerate(rows, 1):
        groups.setdefault(row["reaction_id"], {}).setdefault(int(row["position"]), []).append(uid)
    names = sorted(groups)
    routes, reactions = [], []
    with rdBase.BlockLogs():
        for name in names:
            schema = definitions[name]
            positions = sorted(groups[name])
            if positions != list(range(1, int(schema["components"]) + 1)):
                raise ValueError(f"incomplete roles for {name}")
            routes.append([groups[name][slot] for slot in positions])
            reaction = AllChem.ReactionFromSmarts(schema["Reaction"])
            reaction.Initialize()
            reactions.append(reaction)
    molecules = [None] + [Chem.MolFromSmiles(row["smiles"]) for row in rows]
    if any(mol is None for mol in molecules[1:]):
        raise ValueError("invalid bundled synthon SMILES")
    fp = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=False)
    fingerprints = np.zeros((len(molecules), 2048), dtype=np.uint8)
    for i, mol in enumerate(molecules[1:], 1):
        fingerprints[i] = fp.GetFingerprintAsNumPy(mol)
    provenance = dict(resource="synthonor.data/synthon_space_1M.tsv", rows=len(rows), reactions=len(names),
                      tsv_sha256=hashlib.sha256(tsv).hexdigest(), schemas_sha256=hashlib.sha256(schemas).hexdigest())
    return RecipeSpace(routes), reactions, molecules, fingerprints, fp, provenance


def assemble(reaction, slots, molecules):
    """Keep a single sanitized product, reject unresolved markers/fragments.

    Multiple valid matches are canonicalized and the lexicographically first
    is used, deterministically. This convention is recorded in the report.
    """
    candidates = {}
    with rdBase.BlockLogs():
        for products in reaction.RunReactants(tuple(molecules[i] for i in slots if i)):
            if len(products) != 1:
                continue
            product = products[0]
            try:
                Chem.SanitizeMol(product)
            except (ValueError, RuntimeError):
                continue
            if any(a.GetAtomicNum() in {0, *range(92, 100)} for a in product.GetAtoms()):
                continue
            if len(Chem.GetMolFrags(product)) != 1:
                continue
            candidates[Chem.MolToSmiles(product, canonical=True, isomericSmiles=True)] = product
    if not candidates:
        return None
    smiles = min(candidates)
    return smiles, candidates[smiles]


def training_products(space, reactions, molecules, fp, count, seed):
    if count > space.total:
        raise ValueError("requested more distinct products than raw recipe combinations")
    rng = np.random.default_rng(seed)
    known, attempted = set(), set()
    ids_out, r_out, s_out, smiles_out, fp_out = [], [], [], [], []
    rejected = duplicates = draws = 0
    while len(ids_out) < count:
        ids, rs, slots = space.sample(rng, 1024)
        draws += len(ids)
        for recipe_id, reaction, row in zip(ids, rs, slots):
            if int(recipe_id) in attempted:
                continue
            attempted.add(int(recipe_id))
            result = assemble(reactions[reaction], row, molecules)
            if result is None:
                rejected += 1
            elif result[0] in known:
                duplicates += 1
            else:
                smiles, mol = result
                known.add(smiles)
                ids_out.append(recipe_id); r_out.append(reaction); s_out.append(row)
                smiles_out.append(smiles); fp_out.append(fp.GetFingerprintAsNumPy(mol))
            if len(ids_out) == count:
                break
        if draws >= count * 100 or len(attempted) == space.total:
            if len(ids_out) < count:
                raise ValueError("not enough distinct valid products; reduce training/audit sizes")
    return dict(ids=np.array(ids_out), reactions=np.array(r_out), slots=np.array(s_out),
                smiles=np.array(smiles_out), fingerprints=np.array(fp_out)), dict(
                    attempted=len(attempted), invalid=rejected, duplicate_structures=duplicates)


def recall(query_fp, reference_fp, query_xy, reference_xy):
    # Exact full-product binary Tanimoto against this bounded reference bank.
    q, r = query_fp.astype(np.float32), reference_fp.astype(np.float32)
    intersection = q @ r.T
    union = q.sum(1)[:, None] + r.sum(1)[None, :] - intersection
    similarity = np.divide(intersection, union, out=np.ones_like(intersection), where=union > 0)
    truth = np.argsort(-similarity, axis=1, kind="stable")[:, :10]
    neighbors = np.argsort(cdist(query_xy, reference_xy), axis=1, kind="stable")[:, :min(100, len(r))]
    return dict(recall10=float((neighbors[:, :10, None] == truth[:, None, :]).any(1).mean()),
                true10_in_100=float((neighbors[:, :, None] == truth[:, None, :]).any(1).mean()),
                queries=len(q), references=len(r),
                scope="Canonical-product-disjoint test queries vs training reference products; not recall over the rendered draws.",
                ties="Stable reference-order top ten; equal-similarity ties can lower strict identity recall.")


def fit(args, loaded, device):
    import umap

    if (args.out / "model.pt").exists():
        raise FileExistsError("model.pt already exists; use --stage render or a new --out directory")
    space, reactions, molecules, sfp, fp, provenance = loaded
    started = time.perf_counter()
    data, assembly = training_products(space, reactions, molecules, fp, args.train + args.validation + args.test, args.seed)
    np.savez_compressed(args.out / "training-products.npz", **data)
    print(f"Assembled {len(data['ids'])} distinct products; fitting UMAP to {args.train}", flush=True)
    teacher = umap.UMAP(n_neighbors=20, min_dist=.2, repulsion_strength=3,
                        negative_sample_rate=40, n_epochs=args.umap_epochs, metric="jaccard",
                        random_state=args.seed, transform_seed=args.seed, n_jobs=1)
    teacher_xy = teacher.fit_transform(data["fingerprints"][:args.train].astype(bool))
    val_end = args.train + args.validation
    validation_xy = teacher.transform(data["fingerprints"][args.train:val_end].astype(bool))
    np.save(args.out / "teacher-training.npy", teacher_xy)
    mean, scale = teacher_xy.mean(0), max(float(teacher_xy.std(0).mean()), 1e-6)
    targets = torch.as_tensor((np.concatenate([teacher_xy, validation_xy]) - mean) / scale, device=device)
    fingerprints = torch.as_tensor(sfp, dtype=torch.float32, device=device)
    slots = torch.as_tensor(data["slots"], device=device)
    routes = torch.as_tensor(data["reactions"], device=device)
    torch.manual_seed(args.seed)
    model = CachedReactionMap(len(space.routes)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    best, best_step, best_state, history = float("inf"), 0, None, []
    for step in range(args.steps + 1):
        if step % 100 == 0 or step == args.steps:
            model.eval()
            with torch.no_grad():
                codes = model.cache(fingerprints)
                pred = model.from_cache(codes, slots[args.train:val_end], routes[args.train:val_end])
                mse = float(torch.nn.functional.mse_loss(pred, targets[args.train:val_end]))
            history.append(dict(step=step, validation_mse=mse))
            if mse < best:
                best, best_step = mse, step
                best_state = copy.deepcopy(model.state_dict())
            print(f"Step {step}/{args.steps}, validation MSE {mse:.4f}", flush=True)
        if step == args.steps:
            break
        model.train()
        optimizer.zero_grad(set_to_none=True)
        batch = torch.randint(args.train, (256,), device=device)
        loss = torch.nn.functional.mse_loss(model(fingerprints, slots[batch], routes[batch]), targets[batch])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        codes = model.cache(fingerprints)
        xy = model.from_cache(codes, slots, routes).cpu().numpy() * scale + mean
    quality = recall(data["fingerprints"][val_end:], data["fingerprints"][:args.train], xy[val_end:], xy[:args.train])
    np.save(args.out / "student-positions.npy", xy)
    torch.save(dict(state_dict={k: v.cpu() for k, v in model.state_dict().items()}, codes=codes.cpu(),
                    mean=mean.tolist(), scale=scale, n_reactions=len(space.routes), source=provenance), args.out / "model.pt")
    report = dict(source=provenance, recipe_combinations=space.total, seed=args.seed,
                  train=args.train, validation=args.validation, test=args.test,
                  assembly=assembly, assembly_policy="Single sanitized, connected product without attachment atoms; lexical canonical first match; structure deduplication before splitting.",
                  fingerprint="Morgan radius 2, 2048 bits, no chirality; full assembled products for teacher/audit, attachment atoms retained in synthon inputs.",
                  umap=dict(neighbors=20, min_dist=.2, repulsion=3, negative_samples=40, epochs=args.umap_epochs, implementation="umap-learn CPU"),
                  student=dict(selection="Lowest validation coordinate MSE, no test-based selection", best_step=best_step, validation_mse=best, history=history),
                  quality=quality, fit_seconds=time.perf_counter() - started,
                  versions={p: metadata.version(p) for p in ["synthonor", "rdkit", "umap-learn", "torch", "numpy"]})
    (args.out / "fit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(quality), flush=True)


def render(args, loaded, device):
    from synthonmap.torch_histogram import TorchHistogram

    space, _, _, _, _, provenance = loaded
    saved = torch.load(args.out / "model.pt", map_location=device, weights_only=True)
    if saved["source"] != provenance:
        raise ValueError("example source changed; use a fresh fit")
    model = CachedReactionMap(saved["n_reactions"]).to(device).eval()
    model.load_state_dict(saved["state_dict"])
    codes = saved["codes"].to(device)
    mean = torch.tensor(saved["mean"], device=device)

    def batches():
        # Separate seed from training; reset for identical bounds/count passes.
        rng = np.random.default_rng(args.seed + 1)
        with torch.inference_mode():
            for start in range(0, args.draws, args.batch_size):
                _, rs, slots = space.sample(rng, min(args.batch_size, args.draws - start))
                yield model.from_cache(codes, torch.as_tensor(slots, device=device), torch.as_tensor(rs, device=device)) * saved["scale"] + mean

    started = time.perf_counter()
    bounds = fit_bounds(xy.cpu().numpy() for xy in batches())
    bounds_seconds = time.perf_counter() - started
    histogram = TorchHistogram(bounds, (args.size, args.size), device=device)
    for xy in batches():
        histogram.add(xy)
    counts = histogram.numpy()
    count_seconds = time.perf_counter() - started - bounds_seconds
    receipt = histogram.receipt()
    assert receipt["accepted"] == args.draws == int(counts.sum(dtype=np.uint64))
    np.save(args.out / "counts.npy", counts)
    ceiling = display_ceiling(counts)
    for factor in [1, 8, 64]:
        level = coarsen_counts(counts, factor)
        np.save(args.out / f"counts-{factor}x.npy", level.counts)
        np.save(args.out / f"areas-{factor}x.npy", level.areas)
        save_png(args.out / f"density-{factor}x.png", level.counts, areas=level.areas, ceiling=ceiling)
    receipt.update(source=provenance, recipe_combinations=space.total, seed=args.seed + 1,
                   batch_size=args.batch_size, display_ceiling=ceiling, palette="viridis",
                   color="log1p(count / original_bin_area) / log1p(display_ceiling)",
                   bounds_pass_seconds=bounds_seconds, count_pass_seconds=count_seconds,
                   model_sha256=hashlib.sha256((args.out / "model.pt").read_bytes()).hexdigest(),
                   interpretation="Uniform raw recipes with replacement mapped by the cached-synthon student. No assembly, validity filtering or molecular deduplication in the raster pass. Density is in learned coordinate space, not a count of unique molecules.")
    (args.out / "render.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results/freedom"))
    parser.add_argument("--stage", choices=["all", "fit", "render"], default="all")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--train", type=int, default=3000)
    parser.add_argument("--validation", type=int, default=400)
    parser.add_argument("--test", type=int, default=400)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--umap-epochs", type=int, default=500)
    parser.add_argument("--draws", type=int, default=1_000_000)
    parser.add_argument("--size", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=47)
    args = parser.parse_args()
    if args.train < 21 or min(args.validation, args.test, args.steps, args.umap_epochs, args.draws, args.size, args.batch_size) < 1:
        parser.error("train must be >=21; all counts, sizes and epochs must be positive")
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    device = torch.device(args.device)
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable; use --device cpu")
    with threadpool_limits(limits=2):
        loaded = load_space()
        print(f"Source: {loaded[-1]}; {loaded[0].total:,} raw combinations", flush=True)
        if args.stage in {"all", "fit"}:
            fit(args, loaded, device)
        if args.stage in {"all", "render"}:
            render(args, loaded, device)


if __name__ == "__main__":
    main()

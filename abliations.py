"""
Ablation runner. Takes the full novelty model as reference and toggles one
component at a time; also runs a small grid over the most impactful knobs.
"""
from typing import Dict, List

from experiments import run_sequence_experiment
from novelty import NOVELTY_PRESETS, resolve_novelty, ConfigurableMultiModal


# Presets that isolate a single component of the full model.
ABLATION_PRESETS: Dict[str, str] = {
    "full":              "full",
    "no_bidirectional":  "no_bidirectional",
    "no_tabular":        "no_tabular",
    "tab_concat":        "tab_concat",
    "tab_gate":          "tab_gate",
    "no_cross_attn":     "no_cross_attn",
    "cross_bidir":       "cross_bidir",
    "cross_t2a":         "cross_t2a",
    "no_second_pass":    "no_second_pass",
    "second_transformer":"second_transformer",
    "second_both":       "second_both",
    "no_temporal_gate":  "no_temporal_gate",
    "no_modality_gate":  "no_modality_gate",
    "fusion_early":      "fusion_early",
    "fusion_late":       "fusion_late",
    "hybrid_2t":         "hybrid_2t",
    "hybrid_4t":         "hybrid_4t",
    "pool_attention":    "pool_attention",
    "pool_max":          "pool_max",
    "regularised":       "regularised",
    "contrastive":       "contrastive",
}


def run_ablation_study(df, folds, cfg, egemaps_df, audio_emb, text_emb,
                       presets=None):
    if presets is None:
        presets = ABLATION_PRESETS

    egemaps_arr = egemaps_df.values.astype(np.float32)
    egemaps_index = {s: i for i, s in enumerate(egemaps_df.index)}

    results = {}
    for label, preset_name in presets.items():
        print(f"\n{'=' * 70}\nABLATION: {label}  (preset='{preset_name}')\n"
              f"{'=' * 70}")

        def builder(d_a, d_t, d_z, c, pn=preset_name):
            nc = resolve_novelty(pn, None)
            return ConfigurableMultiModal(
                d_a, d_t, d_z, c.n_outputs, nc,
                d_model=c.fusion_d_model)

        try:
            fm, sm = run_sequence_experiment(
                f"ablation_{label}", builder, df, folds, cfg,
                audio_emb, text_emb, egemaps_arr, egemaps_index)
            results[label] = {
                "preset": preset_name,
                "folds": fm,
                "summary": sm.to_dict("records"),
            }
        except Exception as e:
            import traceback
            print(f"[ERROR] ablation {label} failed: {e}")
            traceback.print_exc()
    return results


def run_ablation_grid(df, folds, cfg, egemaps_df, audio_emb, text_emb):
    """
    A small 2×3×3 grid over the three most impactful knobs.
    Runs 18 configurations (2 × 3 × 3).
    """
    import itertools
    egemaps_arr = egemaps_df.values.astype(np.float32)
    egemaps_index = {s: i for i, s in enumerate(egemaps_df.index)}

    results = {}
    for bi, ca, sp in itertools.product(
            [True, False],
            ["a_to_t", "bidir", "none"],
            ["mamba", "transformer", "none"]):
        label = f"grid_b{int(bi)}_c{ca}_s{sp}"
        print(f"\n{'=' * 70}\nGRID: {label}\n{'=' * 70}")

        overrides = (f"bidirectional={'true' if bi else 'false'},"
                     f"cross_attention={ca},"
                     f"second_pass={sp}")

        def builder(d_a, d_t, d_z, c, ov=overrides):
            nc = resolve_novelty("full", ov)
            return ConfigurableMultiModal(
                d_a, d_t, d_z, c.n_outputs, nc,
                d_model=c.fusion_d_model)

        try:
            fm, sm = run_sequence_experiment(
                label, builder, df, folds, cfg,
                audio_emb, text_emb, egemaps_arr, egemaps_index)
            results[label] = {
                "overrides": overrides,
                "folds": fm,
                "summary": sm.to_dict("records"),
            }
        except Exception as e:
            print(f"[ERROR] grid {label} failed: {e}")
    return results
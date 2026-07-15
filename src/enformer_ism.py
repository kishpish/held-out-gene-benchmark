"""Per-variant Enformer ISM effects as a transferable feature (METHODOLOGY §6.3).

Computes, for each cis-variant, Enformer's in-silico-mutagenesis effect on the GM12878 CAGE track at the
TSS bin: CAGE(reference window with the single alt allele) − CAGE(reference window). This is the model's
own estimate of the variant's regulatory effect. Crucially it depends only on the reference window + the
variant (NOT on any individual), so it is computed once per variant — O(genes × variants), not
O(genes × individuals) — making it tractable to feed the gene-agnostic transfer head (Phase 6) as a
richer, learned alternative to hand-picked motif deltas.

Forwards are batched (bf16, H100); the reference-window CAGE is computed once per gene.
"""
from __future__ import annotations
import numpy as np

from genome import revcomp
from sequence import one_hot
import backbone


def variant_cage_deltas(ref, chrom, tss1, strand, positions, alt_alleles,
                        track=backbone.LCL_CAGE, length=196_608, batch_size=8,
                        device="cuda") -> tuple[np.ndarray, float]:
    """Return (delta per variant [n], reference CAGE at TSS bin).

    positions: 1-based POS array; alt_alleles: matching alt bases. Variant must lie in-window.
    """
    start0 = tss1 - 1 - length // 2
    ref_window = ref.get_seq(chrom, start0, start0 + length)      # + strand
    # reference pass
    ref_seq = revcomp(ref_window) if strand == "-" else ref_window
    cage_ref = float(backbone.predict_human_batch(
        one_hot(ref_seq)[None].astype(np.float32), device=device)[0, backbone.CENTER_BIN, track])

    # build one alt sequence per variant
    seqs, keep = [], []
    for pos1, alt in zip(positions, alt_alleles):
        off = int(pos1 - 1 - start0)
        if off < 0 or off >= length:
            keep.append(False); continue
        alt_win = ref_window[:off] + str(alt) + ref_window[off + 1:]
        seqs.append(revcomp(alt_win) if strand == "-" else alt_win)
        keep.append(True)
    keep = np.array(keep)

    deltas = np.zeros(len(positions), dtype=np.float32)
    vals = np.empty(len(seqs), dtype=np.float32)
    for i in range(0, len(seqs), batch_size):
        chunk = seqs[i:i + batch_size]
        oh = np.stack([one_hot(s) for s in chunk]).astype(np.float32)
        pred = backbone.predict_human_batch(oh, device=device)
        vals[i:i + len(chunk)] = pred[:, backbone.CENTER_BIN, track]
    deltas[keep] = vals - cage_ref
    return deltas, cage_ref

# Use Cases

```{image} _static/cellpin_usecases.png
:align: right
:width: 220px
:class: no-bg
```

## When should you use cellpin?

Cellpin is worth reaching for whenever you have **single-cell resolved spatial transcriptomics data** and intend to do any kind of downstream analysis with it.

Matched reference data, a scRNA-seq sample from the same tissue block, is great, but it is entirely optional. Cellpin works well with out-of-batch public atlases, so you do not need to generate your own reference to benefit from it.

Everything below applies to both panel-based platforms (e.g. Xenium) and full-transcriptome platforms (e.g. Atera).

## 1. Biologically meaningful embeddings

Spatial transcriptomics data is noisy and full of technical artefacts. That makes it hard to produce meaningful cell embeddings, and close to impossible to resolve rare cell states or fine-grained subtypes using embeddings derived from linear methods.

Cellpin embeddings are clean while retaining, and often recovering, the biological signal that downstream analysis depends on. Importantly, cellpin does not impose an explicit cross-dataset alignment, so it will not force cells into a reference position where they do not belong, which is a failure mode of methods that make stronger mapping assumptions.

**We recommend embedding your spatial data with cellpin before any downstream task**, even if you do not need imputation.

## 2. Imputation and denoising

Cellpin imputes genes that are missing from your panel and denoises the genes you did measure. Both capabilities are benchmarked in the [preprint](https://doi.org/10.64898/2026.06.02.729566).

This is useful when you want to:

- **Detect cell types whose markers are absent from your panel.** Panels are finite; the cell types in your tissue are not.
- **Test hypotheses that depend on unmeasured genes**, without designing and running a new panel.
- **Run analyses that noise would otherwise corrupt**, such as differential expression, cell–cell communication and neighbourhood statistics, without missegmentation and transcript-diffusion artefacts driving the result.

The [Xenium denoising tutorial](notebooks/xenium_denoising.ipynb) works through the second point in detail, showing spurious B-cell markers being removed from neighbouring epithelial cells.

## 3. Label transfer

Transfer annotations from a reference dataset to annotate your spatial data automatically. See the [label transfer tutorial](notebooks/label_transfer.ipynb).

## One model, one forward pass

These are not three separate workflows. A single trained model and a single forward pass give you a high-quality embedding, imputed and denoised expression profiles, and transferred cell-state annotations at once, a solid starting point for essentially any spatial analysis.

## Next steps

- [Best Practices](best_practices.md): practical recommendations before you train
- [Basic usage tutorial](notebooks/cellpin_tutorial.ipynb): the core imputation workflow
- [API reference](api.md): all tunable parameters

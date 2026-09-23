# Use Cases

```{image} _static/cellpin_usecases.png
:align: right
:width: 220px
:class: no-bg
```

## When should you use cellpin?

cellpin is particularly useful when you have **single-cell-resolved spatial transcriptomics data** and want to obtain a richer and more robust representation of cellular state for downstream analysis.

Matched reference data, a scRNA-seq sample from the same tissue block, is great, but it is entirely optional. Cellpin works well with out-of-batch public atlases, so you do not need to generate your own reference to benefit from it.

Everything below applies to both panel-based platforms (e.g. Xenium) and full-transcriptome platforms (e.g. Atera).

## 1. Biologically meaningful embeddings

Spatial transcriptomics data is noisy and full of technical artefacts. That makes it hard to produce meaningful cell embeddings, and close to impossible to resolve rare cell states or fine-grained subtypes using embeddings derived from linear methods.

Cellpin embeddings are clean while retaining, and often recovering, the biological signal that downstream analysis depends on. Importantly, cellpin does not impose an explicit cross-dataset alignment, so it will not force cells into a reference position where they do not belong, which is a failure mode of methods that make stronger mapping assumptions.

**We recommend embedding your spatial data with cellpin before any downstream task**, even if you do not need imputation.

## 2. Imputation and denoising

Cellpin imputes genes that are missing from your panel and denoises the genes you did measure. Both capabilities are benchmarked in the [preprint](https://doi.org/10.64898/2026.06.02.729566).

Reconstructed expression can be useful for:

* **exploring cell types or states whose canonical markers are absent from the measured panel**;
* **generating hypotheses involving genes that were not included in the original experiment**;
* **visualizing expected expression patterns for unmeasured genes**;
* **reducing technical noise** in measured genes before exploratory downstream analysis;
* prioritizing genes, pathways, or cell populations for subsequent experimental validation.

Reconstructed expression should nevertheless be interpreted as model-derived information rather than experimentally measured ground truth. For analyses that depend strongly on gene-level measurements—particularly formal differential-expression testing, cell–cell communication inference, or conclusions based primarily on imputed genes—we recommend using reconstructed values as complementary evidence and validating important findings against measured genes or independent data where possible.

The [Xenium denoising tutorial](notebooks/xenium_denoising.ipynb) works through the second point in detail, showing spurious B-cell markers being removed from neighbouring epithelial cells.

## 3. Label transfer

cellpin can also transfer cell-type or cell-state annotations from the reference dataset to the spatial data.

Because annotation is derived from the same learned representation, label transfer naturally accompanies the embedding and reconstruction rather than requiring a separate integration workflow.

As with any reference-based annotation method, transferred labels should be checked against independent biological evidence such as measured marker genes, tissue morphology, or spatial context. See the [label transfer tutorial](notebooks/label_transfer.ipynb).

## One model, one forward pass

These are not three separate workflows. A single trained model and a single forward pass give you a high-quality embedding, imputed and denoised expression profiles, and transferred cell-state annotations at once, a solid starting point for essentially any spatial analysis.

## Next steps

- [Best Practices](best_practices.md): practical recommendations before you train
- [Basic usage tutorial](notebooks/cellpin_tutorial.ipynb): the core imputation workflow
- [API reference](api.md): all tunable parameters

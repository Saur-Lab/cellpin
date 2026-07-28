# Best Practices

```{image} _static/cellpin_bestpractices.png
:align: right
:width: 220px
:class: no-bg
```

A few recommendations to get the most out of cellpin before you dive in.

## Reference data

We recommend using at least **20,000 reference cells** for reliable imputation. In our experience, cellpin performs well with a wide range of scRNA-seq reference datasets, but **10X Chromium Flex data** tends to yield the best results, likely due to its high sensitivity and high capture efficiency across diverse cell types. As a general rule: the better the reference, the better the spatial reconstruction.

## Batch size for large atlases

When mapping against large-scale cell atlases, consider increasing the batch size to **512 or 1024**. The default batch size is optimized for typical reference sizes, but larger batches give the SNN-loss more within-batch diversity to work with, which improves training stability at scale.

## Default parameters

Default parameters have been shown to work robustly across a broad range of tissue types and spatial platforms. That said, they are a starting point, and if results are not as expected it is worth exploring the learning rate, number of epochs, loss weights, or `distillation_mode`. The [API reference](api.md) documents all tunable parameters.

## Integer counts for denoised expression

We strongly recommend setting `return_int=True` when retrieving reconstructed expression profiles. This rounds the output to integer counts, making the denoised matrix directly compatible with tools that expect raw-count-like input, including differential expression methods, trajectory inference, and most standard Scanpy/Seurat workflows.

## Accounting for batch effects

If your spatial data or reference contains strong batch effects, make use of the `batch_key` parameter during setup. Passing a batch annotation allows cellpin to account for technical variation during training and typically leads to cleaner cell-type assignments and more accurate expression reconstruction.

## Inspect label transfer results

Label transfer results are only as good as the reference and the biological similarity between reference and query. As with any label-transfer method, accuracy can vary by dataset, tissue, and cell-type composition. We recommend always inspecting the assigned labels (e.g. against known marker genes or spatial context) before using them in downstream analyses.

## Reading the loss curves

`model.pl.losses()` is a quick sanity check after training. Two things help you read it correctly.

**It shows Stage 2.** Training runs in two stages: Stage 1 pretrains the full-gene VAE on the reference, Stage 2 distills the panel encoder against it. After `fit()`, `pl.losses()` reads the Stage 2 log. To inspect Stage 1, pass its log directory via `log_path=`.

**Flat or shaky curves are usually fine.** When the panel already carries most of the reference signal, Stage 1 has done the job and Stage 2 only fine-tunes, so the curve flattens. Jagged curves are often just a narrow y-axis, so check the value range first.

Use the plot to rule out real problems (divergence, `NaN`s, a steadily climbing loss), then judge the model on what matters: imputation correlation on panel genes, known markers behaving as expected, and whether the embedding separates the cell types you expect.

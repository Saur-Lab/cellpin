# Best Practices

```{image} _static/cellpin_bestpractices.png
:align: right
:width: 220px
:class: no-bg
```

A few recommendations to get the most out of cellpin before you dive in.

## Reference data

We recommend using at least **20,000 reference cells** for reliable imputation. In our experience, cellpin performs well with a wide range of scRNA-seq reference datasets, but **10X Chromium Flex data** tends to yield the best results — likely due to its high sensitivity and high capture efficiency across diverse cell types. As a general rule: the better the reference, the better the spatial reconstruction.

## Batch size for large atlases

When mapping against large-scale cell atlases, consider increasing the batch size to **512 or 1024**. The default batch size is optimized for typical reference sizes, but larger batches give the SNN-loss more within-batch diversity to work with, which improves training stability at scale.

## Default parameters

Default parameters have been shown to work robustly across a broad range of tissue types and spatial platforms. That said, they are a starting point — if results are not as expected, it is worth exploring the learning rate, number of epochs, loss weights, or `distillation_mode`. The [API reference](api.md) documents all tunable parameters.

## Integer counts for denoised expression

We strongly recommend setting `return_int=True` when retrieving reconstructed expression profiles. This rounds the output to integer counts, making the denoised matrix directly compatible with tools that expect raw-count-like input — including differential expression methods, trajectory inference, and most standard Scanpy/Seurat workflows.

## Accounting for batch effects

If your spatial data or reference contains strong batch effects, make use of the `batch_key` parameter during setup. Passing a batch annotation allows cellpin to account for technical variation during training and typically leads to cleaner cell-type assignments and more accurate expression reconstruction.

## Inspect label transfer results

Label transfer results are only as good as the reference and the biological similarity between reference and query — as with any label-transfer method, accuracy can vary by dataset, tissue, and cell-type composition. We recommend always inspecting the assigned labels manually (e.g. against known marker genes or spatial context) before using them in downstream analyses, rather than taking them at face value.

## Reading the loss curves

`model.pl.losses()` is a good sanity check after training, but it is easy to over-interpret. Two things are worth knowing before you read too much into it.

**It only shows Stage 2.** Training runs in two stages: Stage 1 pretrains the full-gene VAE on the reference, and Stage 2 distills the panel encoder against it. After `fit()`, `pl.losses()` reads the Stage 2 log, so what you see is the distillation phase, not the whole run. If you want to inspect Stage 1, pass its log directory explicitly via `log_path=`.

**Flat or shaky curves are usually fine.** On "easy" samples — where the panel already carries most of the information present in the reference — Stage 1 has effectively done the job and Stage 2 is only making small adjustments. The curve then looks flat, or wobbly at a very small scale, simply because there is little left to correct. Equally, a jagged-looking curve is often just a narrow y-axis: check the actual value range before worrying about it.

So use the plot to rule out the things that genuinely go wrong — divergence, `NaN`s, a loss climbing steadily — and then judge the model on what actually matters: imputation correlation on panel genes, known marker genes behaving as expected, and whether the embedding separates the cell types you expect.

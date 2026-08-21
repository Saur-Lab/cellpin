# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/en/1.0.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html

## [0.3.0] - 2026-08-21

- Added `kl_free_bits`, an opt-in per-latent-dimension floor on the KL term
  (default `0.0`, unchanged behavior). Counters within-cell-type variance
  collapse — cells of the same type reconstructing near-identically despite
  real sub-state heterogeneity (e.g. activated vs. exhausted). See
  [Best Practices](docs/best_practices.md) for guidance; not a guaranteed
  fix, but worth sweeping if reconstructions look over-smoothed.

## [0.2.0] - 2026-07-29
 
- Significant (>10x) speed up in impute function, specifically
  parallelisation and GPU support for normalisation


## [0.1.1] - 2026-06-10

- Changed to sparse outputs
- Removed dead code

### Added

- Initial release with basic tool, preprocessing and plotting functions

## [0.0.1] - 2026-05-29

### Added

- Initial release with basic tool, preprocessing and plotting functions

# Changelog

## [0.2.0](https://github.com/SentioLabs/comment-linter/compare/v0.1.0...v0.2.0) (2026-03-03)


### Features

* add config system with TOML loading and default values ([d93d8e3](https://github.com/SentioLabs/comment-linter/commit/d93d8e32d53e59aea3714861e12513b6935ddfd3))
* add core data types, traits, and module structure ([0868211](https://github.com/SentioLabs/comment-linter/commit/08682114301d97adebbe42dd624117f765613e99))
* add cross-reference and staleness feature extraction ([4421d56](https://github.com/SentioLabs/comment-linter/commit/4421d56f913ff6883f4664374e7fad349bd44921))
* add lexical feature extraction with tokenization and similarity ([3da7143](https://github.com/SentioLabs/comment-linter/commit/3da71437adf22d2434b929dd51c01eb24c71e9f4))
* add semantic feature extraction with why-indicators and patterns ([ceb6258](https://github.com/SentioLabs/comment-linter/commit/ceb62585a9831c262939c2fb66b024a54b5be4c0))
* **ci:** add CI/CD setup with GitHub Actions and Makefile ([3d4d90a](https://github.com/SentioLabs/comment-linter/commit/3d4d90acf8f9b2bf2cfcee55f4af614c84230782))
* **cli:** add --scorer and --model-path flags for ML scorer selection ([ecef924](https://github.com/SentioLabs/comment-linter/commit/ecef9245a116ae830c785f855aa49e22eb13e120))
* **cli:** wire up clap CLI with pipeline and output formatting ([94eea42](https://github.com/SentioLabs/comment-linter/commit/94eea4201751eec6e1c34254b4c82f4de2e306f5))
* **export:** add --export-features JSONL output for ML training data ([0bd5b6b](https://github.com/SentioLabs/comment-linter/commit/0bd5b6bd02293726ab9a75212f12e65cd84dab7a))
* **go:** implement Go language comment extraction ([1745a89](https://github.com/SentioLabs/comment-linter/commit/1745a89c8b30cfb8ea96894066832dd1f23cf13a))
* **javascript:** implement JavaScript language comment extraction ([c807f7f](https://github.com/SentioLabs/comment-linter/commit/c807f7f640e72a99547fc300d76caf252e2c40ee))
* **ml:** add EnsembleScorer combining heuristic and ML scoring ([b6bb93c](https://github.com/SentioLabs/comment-linter/commit/b6bb93caafeeb3c7641c026dd4479cb4f16b5090))
* **ml:** implement MLScorer with ONNX Runtime inference ([d223015](https://github.com/SentioLabs/comment-linter/commit/d223015a3e5053ac01f5959ff317741086125aac))
* **ml:** scaffold comment-lint-ml crate with feature-to-tensor conversion ([014b1ef](https://github.com/SentioLabs/comment-linter/commit/014b1ef9f421599863a56d527bcd7a15a7e5c058))
* **output:** implement text, JSON, and GitHub Actions output formatters ([4f07814](https://github.com/SentioLabs/comment-linter/commit/4f0781412a59078969e1b7f519b9ab87f6a01661))
* **pipeline:** add pipeline orchestration with streaming and caching ([c9d67f8](https://github.com/SentioLabs/comment-linter/commit/c9d67f88f5018ceeb85bc9a7ee79c9e91432ae31))
* **python:** implement Python language comment extraction ([2f628c6](https://github.com/SentioLabs/comment-linter/commit/2f628c61932bf67913434ba6533663b8757e541a))
* **rust:** implement Rust language comment extraction ([df8a334](https://github.com/SentioLabs/comment-linter/commit/df8a33409ddf394e445afad54852b79f12e15777))
* scaffold Cargo workspace with comment-lint and comment-lint-core crates ([883b9a8](https://github.com/SentioLabs/comment-linter/commit/883b9a8b44f657cce455248ea51eba2fa305b60f))
* **scoring:** implement heuristic weighted-sum scorer with confidence ([a5bd418](https://github.com/SentioLabs/comment-linter/commit/a5bd418686114cf5cd5158a996a7eb92f39099b9))
* **training:** add model training pipeline and ONNX export ([5327b3d](https://github.com/SentioLabs/comment-linter/commit/5327b3d82445edfad367c4da2b0a6b52fa60372c))
* **training:** add training data pipeline with labeling and splitting ([ebcbbed](https://github.com/SentioLabs/comment-linter/commit/ebcbbed973dc1deb6e7739c44578be8b76ad7e44))
* **typescript:** implement TypeScript language comment extraction ([62aea83](https://github.com/SentioLabs/comment-linter/commit/62aea839a66ecdcf7218dc0bb2ec1d97da6a9ecb))


### Bug Fixes

* **ci:** fix formatting, clippy warnings, and release-please config ([f3395c3](https://github.com/SentioLabs/comment-linter/commit/f3395c31b6569a554d62ea28a04bfba7a3e02853))


### Refactoring

* **training:** restructure scripts into src/clt/ package ([f006a60](https://github.com/SentioLabs/comment-linter/commit/f006a60d05ac5d46087e4047917cab76e09b1f1a))

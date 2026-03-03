# comment-linter Makefile
# ─────────────────────────────────────────────────────────────────────────────

VERSION  := $(shell grep '^version' crates/comment-lint/Cargo.toml | head -1 | sed 's/.*"\(.*\)"/\1/')
COMMIT   := $(shell git rev-parse --short HEAD 2>/dev/null || echo "unknown")
HOST_TARGET := $(shell rustc -vV | grep '^host:' | cut -d' ' -f2)
BINARY   := comment-lint
PROJECT  := comment-linter

# ── Build ────────────────────────────────────────────────────────────────────

.PHONY: build
build: ## Build debug binary
	cargo build -p $(BINARY)

.PHONY: build-release
build-release: ## Build release binary
	cargo build --release -p $(BINARY)

.PHONY: install
install: ## Install binary to ~/.cargo/bin
	cargo install --path crates/comment-lint

.PHONY: clean
clean: ## Remove build artifacts
	cargo clean

# ── Test ─────────────────────────────────────────────────────────────────────

.PHONY: test
test: ## Run all tests
	cargo test

.PHONY: test-single
test-single: ## Run a single test (T=test_name)
	cargo test $(T)

.PHONY: test-file
test-file: ## Run tests in a specific file (F=file_path)
	cargo test --test $(F)

# ── Lint ─────────────────────────────────────────────────────────────────────

.PHONY: fmt
fmt: ## Format code
	cargo fmt --all

.PHONY: fmt-check
fmt-check: ## Check formatting
	cargo fmt --all --check

.PHONY: clippy
clippy: ## Run clippy lints
	cargo clippy --all-targets -- -D warnings

.PHONY: lint
lint: fmt-check clippy ## Run all lints (fmt-check + clippy)

.PHONY: check
check: lint test ## Full check (lint + test)

# ── Review ───────────────────────────────────────────────────────────────────

.PHONY: review
review: ## Full status report (format, clippy, tests verbose)
	@echo "══════════════════════════════════════════════════════"
	@echo " $(PROJECT) v$(VERSION) ($(COMMIT)) review"
	@echo "══════════════════════════════════════════════════════"
	@echo ""
	@echo "── Format ──────────────────────────────────────────"
	@cargo fmt --all --check && echo "OK" || echo "FAIL"
	@echo ""
	@echo "── Clippy ──────────────────────────────────────────"
	@cargo clippy --all-targets -- -D warnings 2>&1
	@echo ""
	@echo "── Tests ───────────────────────────────────────────"
	@cargo test -- --nocapture 2>&1
	@echo ""
	@echo "══════════════════════════════════════════════════════"
	@echo " Review complete"
	@echo "══════════════════════════════════════════════════════"

# ── Release ──────────────────────────────────────────────────────────────────

.PHONY: release-build
release-build: ## Build release binary for host target
	cargo build --release -p $(BINARY) --target $(HOST_TARGET)
	@echo "Binary: target/$(HOST_TARGET)/release/$(BINARY)"

.PHONY: release-local
release-local: release-build ## Package a local release tarball
	@mkdir -p dist
	@ARCHIVE="$(PROJECT)_$(VERSION)_$(HOST_TARGET)"; \
	mkdir -p "dist/$$ARCHIVE"; \
	cp "target/$(HOST_TARGET)/release/$(BINARY)" "dist/$$ARCHIVE/"; \
	cp README.md LICENSE "dist/$$ARCHIVE/"; \
	cd dist && tar czf "$$ARCHIVE.tar.gz" "$$ARCHIVE" && rm -rf "$$ARCHIVE"; \
	echo "Created dist/$$ARCHIVE.tar.gz"

.PHONY: release-list
release-list: ## List GitHub releases
	gh release list

.PHONY: release-show
release-show: ## Show latest GitHub release
	gh release view

.PHONY: release-download
release-download: ## Download latest release assets
	@mkdir -p dist
	gh release download --dir dist

.PHONY: release-delete
release-delete: ## Delete a GitHub release (V=tag)
	gh release delete $(V) --yes

.PHONY: release-pr
release-pr: ## Show the current release-please PR
	gh pr list --label "autorelease: pending"

# ── Help ─────────────────────────────────────────────────────────────────────

.PHONY: help
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help

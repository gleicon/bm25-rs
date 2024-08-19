# BM25 sparse in Rust

This repository contains a Rust implementation of the BM25S algorithm, optimized for speed and memory efficiency.

This is an experimental work based on [BM25 sparse](https://github.com/xhluca/bm25s) and [BMX](https://www.mixedbread.ai/blog/intro-bmx)




## Features

- Efficient indexing using sparse matrices
- Query processing and top-k retrieval
- Easy saving and loading of indexed models

## Installation

Clone the repository and use Cargo to build the project:

```bash
git clone https://github.com/yourusername/bm25s-rs.git
cd bm25s-rs
cargo build


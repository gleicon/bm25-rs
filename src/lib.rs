extern crate nalgebra as na;
use std::collections::HashMap;

pub struct BM25S {
    pub sparse_matrix: HashMap<usize, Vec<(usize, f64)>>, // term_index -> Vec<(doc_index, score)>
    doc_lengths: Vec<f64>,  // Stores the length of each document
    avg_doc_length: f64,    // Average document length
    doc_count: usize,       // Total number of documents
    k1: f64,                // BM25 parameter
    b: f64,                 // BM25 parameter
}

impl BM25S {
    pub fn new(doc_count: usize, avg_doc_length: f64, k1: f64, b: f64) -> Self {
        BM25S {
            sparse_matrix: HashMap::new(),
            doc_lengths: vec![0.0; doc_count],
            avg_doc_length,
            doc_count,
            k1,
            b,
        }
    }

    pub fn index(&mut self, term_freqs: Vec<Vec<(usize, f64)>>, idf: Vec<f64>) {
        for (doc_id, terms) in term_freqs.iter().enumerate() {
            for &(term_id, tf) in terms {
                let len_d = self.doc_lengths[doc_id];
                let score = (tf * idf[term_id]) / (tf + self.k1 * (1.0 - self.b + self.b * (len_d / self.avg_doc_length)));
                
                self.sparse_matrix
                    .entry(term_id)
                    .or_insert_with(Vec::new)
                    .push((doc_id, score));
            }
        }
    }

    pub fn query(&self, query_terms: Vec<usize>) -> Vec<f64> {
        let mut doc_scores = vec![0.0; self.doc_count];
        for &term_id in &query_terms {
            if let Some(entries) = self.sparse_matrix.get(&term_id) {
                for &(doc_id, score) in entries {
                    doc_scores[doc_id] += score;
                }
            }
        }
        doc_scores
    }

    pub fn top_k(&self, scores: Vec<f64>, k: usize) -> Vec<(usize, f64)> {
        let mut indexed_scores: Vec<(usize, f64)> = scores.into_iter().enumerate().collect();
        indexed_scores.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        indexed_scores.into_iter().take(k).collect()
    }

    pub fn save(&self, path: &str) {
        // Placeholder for saving the model to disk
        // You would typically serialize the sparse_matrix and other relevant fields
    }

    pub fn load(path: &str) -> Self {
        // Placeholder for loading the model from disk
        // You would deserialize the data into a BM25S object
        BM25S::new(4, 8.0, 1.5, 0.75) // Example placeholder return value
    }
}

pub fn tokenize_corpus(corpus: &[&str]) -> Vec<Vec<(usize, f64)>> {
    // Simplified tokenizer: splits by whitespace and assigns a dummy tf value
    corpus
        .iter()
        .map(|doc| {
            doc.split_whitespace()
                .enumerate()
                .map(|(i, _word)| (i, 1.0)) // Assigning 1.0 as a dummy term frequency
                .collect()
        })
        .collect()
}

pub fn tokenize_query(query: &str) -> Vec<usize> {
    // Simplified tokenizer: splits by whitespace and converts to term ids
    query
        .split_whitespace()
        .enumerate()
        .map(|(i, _word)| i)
        .collect()
}


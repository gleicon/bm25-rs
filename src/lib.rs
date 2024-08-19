extern crate nalgebra as na;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::{Read, Write};

#[derive(Serialize, Deserialize)]
pub struct BM25S {
    pub sparse_matrix: HashMap<usize, Vec<(usize, f64)>>, // term_index -> Vec<(doc_index, score)>
    doc_lengths: Vec<f64>,                                // Stores the length of each document
    avg_doc_length: f64,                                  // Average document length
    doc_count: usize,                                     // Total number of documents
    k1: f64,                                              // BM25 parameter
    b: f64,                                               // BM25 parameter
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
                if term_id < idf.len() {
                    let len_d = self.doc_lengths[doc_id];
                    let score = (tf * idf[term_id])
                        / (tf + self.k1 * (1.0 - self.b + self.b * (len_d / self.avg_doc_length)));

                    self.sparse_matrix
                        .entry(term_id)
                        .or_insert_with(Vec::new)
                        .push((doc_id, score));
                } else {
                    println!(
                        "Warning: term_id {} is out of bounds for idf vector.",
                        term_id
                    );
                }
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
        let mut file = std::fs::File::create(path).unwrap();
        let encoded: Vec<u8> = bincode::serialize(self).unwrap();
        file.write_all(&encoded).unwrap();
    }

    pub fn load(path: &str) -> Self {
        let mut file = std::fs::File::open(path).unwrap();
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer).unwrap();
        bincode::deserialize(&buffer).unwrap()
    }
}

// Function to calculate IDF values
pub fn calculate_idf(corpus_tokens: &Vec<Vec<(usize, f64)>>, vocab_size: usize) -> Vec<f64> {
    let mut df = vec![0; vocab_size];
    let num_docs = corpus_tokens.len() as f64;

    for doc in corpus_tokens {
        let mut seen_terms = vec![false; vocab_size];
        for &(term_id, _) in doc {
            if !seen_terms[term_id] {
                df[term_id] += 1;
                seen_terms[term_id] = true;
            }
        }
    }

    df.iter()
        .map(|&n| ((num_docs - n as f64 + 0.5) / (n as f64 + 0.5)).ln() + 1.0)
        .collect()
}

pub fn tokenize_corpus(corpus: &[&str]) -> (Vec<Vec<(usize, f64)>>, HashMap<String, usize>) {
    let mut term_to_id = HashMap::new();
    let mut id_counter = 0;

    let corpus_tokens = corpus
        .iter()
        .map(|doc| {
            doc.split_whitespace()
                .map(|word| {
                    let term_id = *term_to_id.entry(word.to_string()).or_insert_with(|| {
                        let id = id_counter;
                        id_counter += 1;
                        id
                    });
                    (term_id, 1.0) // Simplified term frequency as 1.0
                })
                .collect::<Vec<(usize, f64)>>()
        })
        .collect();

    (corpus_tokens, term_to_id)
}

pub fn tokenize_query(query: &str, term_to_id: &HashMap<String, usize>) -> Vec<usize> {
    query
        .split_whitespace()
        .filter_map(|word| term_to_id.get(word).cloned())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bm25_with_corpus() {
        let doc_count = 4;
        let avg_doc_length = 8.0;
        let k1 = 1.5;
        let b = 0.75;

        let mut bm25s = BM25S::new(doc_count, avg_doc_length, k1, b);

        let corpus = vec![
            "a cat is a feline and likes to purr",
            "a dog is the human's best friend and loves to play",
            "a bird is a beautiful animal that can fly",
            "a fish is a creature that lives in water and swims",
        ];

        let corpus_tokens = tokenize_corpus(&corpus);

        let idf = vec![1.2, 1.0, 0.8, 1.3, 1.0];

        bm25s.index(corpus_tokens.clone(), idf);

        let query = "does the fish purr like a cat?";
        let query_tokens = tokenize_query(query);

        let scores = bm25s.query(query_tokens);

        let top_k_results = bm25s.top_k(scores, 2);

        for (i, (doc_id, score)) in top_k_results.iter().enumerate() {
            println!(
                "Rank {} (score: {:.2}): {}",
                i + 1,
                score,
                corpus[*doc_id]
            );
        }

        bm25s.save("animal_index_bm25");
        let reloaded_bm25s = BM25S::load("animal_index_bm25");

        let reloaded_scores = reloaded_bm25s.query(tokenize_query(query));
        let reloaded_top_k_results = reloaded_bm25s.top_k(reloaded_scores, 2);

        assert_eq!(top_k_results, reloaded_top_k_results);
    }
}


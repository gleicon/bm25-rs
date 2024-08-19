use bm25s_rs::*; // Import everything from your library

#[test]
fn test_bm25_with_corpus() {
    let corpus = vec![
        "a cat is a feline and likes to purr",
        "a dog is the human's best friend and loves to play",
        "a bird is a beautiful animal that can fly",
        "a fish is a creature that lives in water and swims",
    ];

    // Tokenize the corpus and get the term_to_id mapping
    let (corpus_tokens, term_to_id) = tokenize_corpus(&corpus);

    // Calculate IDF
    let idf = calculate_idf(&corpus_tokens, term_to_id.len());

    // Create BM25 model
    let doc_count = corpus.len();
    let avg_doc_length = corpus
        .iter()
        .map(|doc| doc.split_whitespace().count() as f64)
        .sum::<f64>()
        / doc_count as f64;
    let k1 = 1.5;
    let b = 0.75;
    let mut bm25s = BM25S::new(doc_count, avg_doc_length, k1, b);

    // Index the corpus
    bm25s.index(corpus_tokens.clone(), idf);

    // Tokenize the query using the same term_to_id mapping
    let query = "does the fish purr like a cat?";
    let query_tokens = tokenize_query(query, &term_to_id);

    // Get scores
    let scores = bm25s.query(query_tokens);

    // Get top-k results
    let top_k_results = bm25s.top_k(scores, 2);

    for (i, (doc_id, score)) in top_k_results.iter().enumerate() {
        println!("Rank {} (score: {:.2}): {}", i + 1, score, corpus[*doc_id]);
    }

    bm25s.save("animal_index_bm25");
    let reloaded_bm25s = BM25S::load("animal_index_bm25");

    let reloaded_scores = reloaded_bm25s.query(tokenize_query(query, &term_to_id));
    let reloaded_top_k_results = reloaded_bm25s.top_k(reloaded_scores, 2);

    assert_eq!(top_k_results, reloaded_top_k_results);
}

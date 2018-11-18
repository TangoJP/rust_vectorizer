extern crate regex;
extern crate ndarray;
extern crate indexmap;

use std::vec::Vec;
use std::collections::HashMap;
use ndarray::{Array1, Array2};
// use indexmap::IndexMap;

mod tokenizer;  // implement trait related to tokenization

pub struct CountVectorizer<'a> {
    pub vocabulary_ : HashMap<&'a str, u32>,
}

impl<'a> CountVectorizer<'a> {
    pub fn new() -> CountVectorizer<'a> {
        let map: HashMap<&'a str, u32> = HashMap::new();

        // Return a new instance
        CountVectorizer {
            vocabulary_: map,
        }
    }

    fn _sort_vocabulary_count(&self, vec_of_map: Vec<HashMap<u32, u32>>) -> Array2<u32>{
        let num_rows = vec_of_map.len();
        let num_columns = self.vocabulary_.len();
        let mut sorted_vec = Array2::<u32>::zeros((num_rows, num_columns));

        for i in 0..num_rows {
            for key in vec_of_map[i].keys() {
                sorted_vec[[i, (*key as usize)]] = *vec_of_map[i].get(key).unwrap();
            }
        }
        sorted_vec
    }

    pub fn reverse_vocabulary_hashmap(&self) -> HashMap<u32, &'a str> {
        // Utility method that returns a HashMap for vocab where k and v are swapped
        let mut vocabulary_inverted: HashMap<u32, &str> = HashMap::new();
        for (k, v) in self.vocabulary_.iter() {
        vocabulary_inverted.insert(*v, k);
        }
        vocabulary_inverted
    }

    pub fn fit_transform(&mut self, docs: Vec<&'a str>) -> Array2<u32> { //Vec<HashMap<i32, i32>> {
        // tokenize the document collection
        let _tokenized_docs = tokenizer::Tokenizer::_tokenize_multiple_docs(docs);

        // Vec to store vocab. count HashMap. Variable to return.
        let mut vec_of_map: Vec<HashMap<u32, u32>> = Vec::new();

        // Collect vocabulary
        let mut vocab_indexer: u32 = 0;             // indexer for unique words

        for _doc in _tokenized_docs {
            // HashMap to store vocab. counts for a doc
            let mut _vocab_counts: HashMap<u32, u32> = HashMap::new();

            for _token in _doc {
                // if _token is a new word, add to vocabulary_ and vocabulary_counts_
                if !self.vocabulary_.contains_key(_token) {
                    self.vocabulary_.insert(_token, vocab_indexer.clone());
                    _vocab_counts.insert(vocab_indexer.clone(), 1);
                    vocab_indexer = vocab_indexer + 1;
                } else {        // Otherwise add vocab counts
                    let vocab_ind = self.vocabulary_[_token];
                    *(_vocab_counts).entry(vocab_ind).or_insert(0) += 1;
                };
            }
            vec_of_map.push(_vocab_counts);
        }
        let sorted_vec = self._sort_vocabulary_count(vec_of_map);
        sorted_vec   // Return the Vec of count HashMaps
    }

}

fn _normalize_matrix() {}

pub fn _get_term_frequency(countvector: Array2<u32>, method: &str) -> Array2<f64>{
    // First pass. For now, if method == "ln", takes natural log of each element
    // Otherwise, just take the original counts.

    let term_frequency = countvector.mapv(|element| element as f64);
    if method == "ln" {
        term_frequency.mapv(f64::ln)
    } else { 
        term_frequency
    }
}

pub fn _get_document_frequency(countvector: Array2<u32>) -> Array1<f64>{
    // First pass. Refactor to make it more efficient

    let (num_rows, num_columns) = countvector.dim();
    let mut document_frequency = Array1::<f64>::zeros(num_columns);
    for index_row in 0..num_rows {
        for index_col in 0..num_columns {
            if countvector[[index_row, index_col]] != 0 {
                document_frequency[index_col] += 1.;
            }
        }
    }
    document_frequency = document_frequency/(num_rows as f64);
    document_frequency
}



// fn tfidi_transform(countvector: Array2<u32>) -> Array2<f64> {
//     let mut sorted_vec = Array2::<f64>::zeros((2, 2));
//     sorted_vec
// }

// pub struct TfidfVectorizer<'a> {
//     // pub vocabulary_ : HashMap<&'a str, i32>,
// }

// impl<'a> TfidfVectorizer<'a> {}
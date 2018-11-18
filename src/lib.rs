extern crate regex;
extern crate ndarray;
extern crate indexmap;

use std::vec::Vec;
use std::collections::HashMap;
use ndarray::Array2;
// use indexmap::IndexMap;

mod tokenizer;  // implement trait related to tokenization

pub struct CountVectorizer<'a> {
    pub vocabulary_ : HashMap<&'a str, i32>,
}

impl<'a> CountVectorizer<'a> {
    pub fn new() -> CountVectorizer<'a> {
        let map: HashMap<&'a str, i32> = HashMap::new();

        // Return a new instance
        CountVectorizer {
            vocabulary_: map,
        }
    }

    fn _sort_vocabulary_count(&self, vec_of_map: Vec<HashMap<i32, i32>>) -> Array2<i32>{
        let num_rows = vec_of_map.len();
        let num_columns = self.vocabulary_.len();
        let mut sorted_vec = Array2::<i32>::zeros((num_rows, num_columns));

        for i in 0..num_rows {
            for key in vec_of_map[i].keys() {
                sorted_vec[[i, (*key as usize)]] = *vec_of_map[i].get(key).unwrap();
            }
        }
        sorted_vec
    }

    pub fn reverse_vocabulary_hashmap(&self) -> HashMap<i32, &'a str> {
        // Utility method that returns a HashMap for vocab where k and v are swapped
        let mut vocabulary_inverted: HashMap<i32, &str> = HashMap::new();
        for (k, v) in self.vocabulary_.iter() {
        vocabulary_inverted.insert(*v, k);
        }
        vocabulary_inverted
    }

    pub fn fit_transform(&mut self, docs: Vec<&'a str>) -> Array2<i32> { //Vec<HashMap<i32, i32>> {
        // tokenize the document collection
        let _tokenized_docs = tokenizer::_tokenize_multiple_docs(docs);

        // Vec to store vocab. count HashMap. Variable to return.
        let mut vec_of_map: Vec<HashMap<i32, i32>> = Vec::new();

        // Collect vocabulary
        let mut vocab_indexer: i32 = 0;             // indexer for unique words

        for _doc in _tokenized_docs {
            // HashMap to store vocab. counts for a doc
            let mut _vocab_counts: HashMap<i32, i32> = HashMap::new();

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

    // Also, split up fit_transform into generalizable functions that can be
    // sent out as trait methods. That way, so functions can be shared with
    // TfidfVectorizer to com.
}


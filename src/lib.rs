extern crate regex;
extern crate ndarray;
extern crate indexmap;

use std::vec::Vec;
use std::collections::HashMap;
use regex::Regex;
use ndarray::Array2;
use indexmap::IndexMap;

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

    fn _tokenize_single_doc(doc: &str) -> Vec<&str> {
        let token_pattern=r"(?u)\b\w\w+\b";
        let _re = Regex::new(token_pattern).unwrap();
        let _tokens: Vec<&str> = _re.find_iter(doc)
            .map(|f| f.as_str())
            .collect();
        _tokens
    }

    fn _tokenize_multiple_docs(docs: Vec<&str>) -> Vec<Vec<&str>> {
        let mut _tokenized_docs: Vec<Vec<&str>> = Vec::new();
        for doc in docs {
            let mut _tokens: Vec<&str> ;
            _tokens = CountVectorizer::_tokenize_single_doc(doc);
            _tokenized_docs.push(_tokens)
        };
        _tokenized_docs
    }

    fn _sort_vocabulary_count(&self, X: Vec<HashMap<i32, i32>>) -> Array2<i32>{
        let num_rows = X.len();
        let num_columns = self.vocabulary_.len();
        let mut sorted_X = Array2::<i32>::zeros((num_rows, num_columns));

        for i in 0..num_rows {
            for key in X[i].keys() {
                sorted_X[[i, (*key as usize)]] = *X[i].get(key).unwrap();
            }
        }
        sorted_X
    }

    pub fn fit_transform(&mut self, docs: Vec<&'a str>) -> Array2<i32> { //Vec<HashMap<i32, i32>> {
        // tokenize the document collection
        let _tokenized_docs = CountVectorizer::_tokenize_multiple_docs(docs);

        // Vec to store vocab. count HashMap. Variable to return.
        let mut X: Vec<HashMap<i32, i32>> = Vec::new();

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
            X.push(_vocab_counts);
        }
        let sortedX = self._sort_vocabulary_count(X);
        sortedX   // Return the Vec of count HashMaps
    }
}


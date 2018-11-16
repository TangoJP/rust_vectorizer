extern crate regex;

use std::vec::Vec;
use std::collections::HashMap;
use regex::Regex;

pub struct CountVectorizer<'a> {
    pub vocabulary_ : HashMap<&'a str, i32>,
    pub vocabulary_counts_ : HashMap<i32, i32>,
}

impl<'a> CountVectorizer<'a> {
    pub fn new() -> CountVectorizer<'a> {
        let map: HashMap<&'a str, i32> = HashMap::new();
        let vocab_counts: HashMap<i32, i32> = HashMap::new();
        CountVectorizer {
            vocabulary_: map,
            vocabulary_counts_: vocab_counts,
        }
    }

    pub fn fit(&mut self, doc: &'a str) {
        // Tokenize the doc
        let token_pattern=r"(?u)\b\w\w+\b";
        let _re = Regex::new(token_pattern).unwrap();
        let _tokens: Vec<&str> = _re.find_iter(doc)
            .map(|f| f.as_str())
            .collect();
        
        // Collect vocabulary
        let mut counter: i32 = 1;
        for _token in _tokens {
            if !self.vocabulary_.contains_key(_token) {
                self.vocabulary_.insert(_token, counter.clone());
                counter = counter + 1;
            }
        }
    }

    pub fn fit_transform(&mut self, doc: &'a str) {
        // Tokenize the doc
        let token_pattern=r"(?u)\b\w\w+\b";
        let _re = Regex::new(token_pattern).unwrap();
        let _tokens: Vec<&str> = _re.find_iter(doc)
            .map(|f| f.as_str())
            .collect();
        
        // Collect vocabulary
        let mut vocab_indexer: i32 = 1;
        for _token in _tokens {
            // if _token is a new word, add to vocabulary_ and vocabulary_counts_
            if !self.vocabulary_.contains_key(_token) {
                self.vocabulary_.insert(_token, vocab_indexer.clone());
                self.vocabulary_counts_.insert(vocab_indexer, 1);

                vocab_indexer = vocab_indexer + 1;
            } else { // Otherwise add vocab counts
                let vocab_ind = self.vocabulary_[_token];
                *(self.vocabulary_counts_).get_mut(&vocab_ind).unwrap() += 1;
            };
        }
    }

}
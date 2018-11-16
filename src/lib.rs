extern crate regex;

use std::vec::Vec;
use std::collections::HashMap;
use regex::Regex;

pub struct CountVectorizer<'a> {
    pub vocabulary : HashMap<&'a str, i32>,
    // vocabulary_counts : HashMap<i32, i32>,
}

impl<'a> CountVectorizer<'a> {
    pub fn new() -> CountVectorizer<'a> {
        let map: HashMap<&'a str, i32> = HashMap::new();
        CountVectorizer {
            vocabulary: map,
            // vocabulary_counts: vocab_counts,
        }
    }

    pub fn fit(&mut self, doc: &'a str) {
        // Tokenize the doc
        let token_pattern=r"(?u)\b\w\w+\b";
        let _re = Regex::new(token_pattern).unwrap();
        let _tokens: Vec<&str> = _re.find_iter(doc)
            .map(|f| f.as_str())
            .collect();
        
        // let mut vocabulary = HashMap::new();
        let mut counter: i32 = 1;
        for _token in _tokens {
            if !self.vocabulary.contains_key(_token) {
                self.vocabulary.insert(_token, counter.clone());
                counter = counter + 1;
            }
        }
        // self.vocabulary = vocabulary;
    }

}
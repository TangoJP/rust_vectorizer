use std::vec::Vec;
use regex::Regex;

// enum NGram{
//     Unigram(u8),
//     Bigram(u8),
//     Trigram(u8),
// }

pub struct Tokenizer {
    // n_gram: n_gram,
}

impl Tokenizer{
    pub fn new() -> Tokenizer {
        Tokenizer {}
    }

    pub fn _tokenize_single_doc(doc: &str) -> Vec<&str> {
        let token_pattern=r"(?u)\b\w\w+\b";
        let _re = Regex::new(token_pattern).unwrap();
        let _tokens: Vec<&str> = _re.find_iter(doc)
            .map(|f| f.as_str())
            .collect();
        _tokens
    }

    pub fn _tokenize_multiple_docs(docs: Vec<&str>) -> Vec<Vec<&str>> {
        let mut _tokenized_docs: Vec<Vec<&str>> = Vec::new();
        for doc in docs {
            let mut _tokens: Vec<&str> ;
            _tokens = _tokenize_single_doc(doc);
            _tokenized_docs.push(_tokens)
        };
        _tokenized_docs
    }
}

pub fn _tokenize_single_doc(doc: &str) -> Vec<&str> {
    let token_pattern=r"(?u)\b\w\w+\b";
    let _re = Regex::new(token_pattern).unwrap();
    let _tokens: Vec<&str> = _re.find_iter(doc)
        .map(|f| f.as_str())
        .collect();
    _tokens
}

pub fn _tokenize_multiple_docs(docs: Vec<&str>) -> Vec<Vec<&str>> {
    let mut _tokenized_docs: Vec<Vec<&str>> = Vec::new();
    for doc in docs {
        let mut _tokens: Vec<&str> ;
        _tokens = _tokenize_single_doc(doc);
        _tokenized_docs.push(_tokens)
    };
    _tokenized_docs
}
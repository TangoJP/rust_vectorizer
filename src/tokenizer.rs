use std::vec::Vec;
use regex::Regex;


pub struct Tokenizer {
    n_gram: u32,
}

impl Tokenizer{
    pub fn new(n_gram: u32) -> Tokenizer {
        Tokenizer {
            n_gram: n_gram,
        }
    }

    pub fn _pick_pattern<'a>(&'a self) -> &'a str {
        let pattern_choice = self.n_gram;
        match pattern_choice {
            1_u32 => r"(?u)\b\w\w+\b",
            2_u32 => r"(?u)\b\w\w+\b",
            3_u32 => r"(?u)\b\w\w+\b",
            _ => "None"
        }
    }

    pub fn _tokenize_single_doc<'a>(&self, doc: &'a str) -> Vec<&'a str> {
        let token_pattern=r"(?u)\b\w\w+\b";
        let _re = Regex::new(token_pattern).unwrap();
        let _tokens: Vec<&str> = _re.find_iter(doc)
            .map(|f| f.as_str())
            .collect();
        _tokens
    }

    pub fn _tokenize_multiple_docs<'a>(&self, docs: Vec<&'a str>) -> Vec<Vec<&'a str>> {
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
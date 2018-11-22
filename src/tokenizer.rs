use std::vec::Vec;
use std::cmp;
use std::ops::Range;
use regex::Regex;

/// A struct used to tokenize a collection of documents (i.e. Vector of
/// string slices). It has one field, n_gram, to specify the n-gram option
/// for tokenization.
/// 
pub struct Tokenizer {
    ngram_range: (u32, u32),
}

impl Tokenizer{
    /// Create a new instance of Tokenizer with n_gram (u32) input.
    /// 
    pub fn new(ngram_range: (u32, u32)) -> Tokenizer {
        Tokenizer {
            ngram_range: ngram_range,
        }
    }
    
    // Private function for picking a pattern based on n_gram choice.
    fn _pick_pattern<'a>(&'a self) -> &'a str {
        let pattern_choice = self.n_gram;
        match pattern_choice { // this can be replaced by generalized &str creation function.
            1_u32 => r"(?u)\b\w\w+\b",
            2_u32 => r"(?u)(\b\w\w+\b){2}",
            3_u32 => r"(?u)(\b\w\w+\b){3}",
            _ => "None" // <= implement error handling here! or generalize this to n-grams
        }
    }

    fn _word_ngrams(ngram_range: (u32, u32), tokens: Vec<&str>) {
        let (min_n, max_n) = ngram_range;
        let original_tokens = tokens.clone();
        let num_original_tokens = original_tokens.len() as u32;
        let n_iter_max = cmp::min(max_n+1, num_original_tokens + 1);
        for n in min_n..n_iter_max {
            for i in 0..(num_original_tokens - n + 1) {
                let range = Range {start: i as usize, end: (i + n) as usize};
                println!("{:?}", &original_tokens[range].join(" "));
            }
    
        }

    }

    fn _tokenize_single_doc<'a>(&self, doc: &'a str) -> Vec<&'a str> {
        let token_pattern=r"(?u)\b\w\w+\b";//self._pick_pattern();
        let _re = Regex::new(token_pattern).unwrap();
        let _tokens: Vec<&str> = _re.find_iter(doc)
            .map(|f| f.as_str())
            .collect();
        _tokens
    }

    fn _tokenize_multiple_docs<'a>(&self, docs: Vec<&'a str>) -> Vec<Vec<&'a str>> {
        let mut _tokenized_docs: Vec<Vec<&str>> = Vec::new();
        for doc in docs {
            let mut _tokens: Vec<&str> ;
            _tokens = _tokenize_single_doc(doc);
            _tokenized_docs.push(_tokens)
        };
        _tokenized_docs
    }

    /// Tokenize all documents in a collection
    pub fn tokenize<'a>(&self, docs: Vec<&'a str>) -> Vec<Vec<&'a str>> {
        let _tokenized_doc = self._tokenize_multiple_docs(docs);
        _tokenized_doc
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


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pick_pattern(){
        let tk1 = Tokenizer::new(1);
        let tk2 = Tokenizer::new(2);
        let tk3 = Tokenizer::new(3);
        let tk4 = Tokenizer::new(4);
        let pattern1 = tk1._pick_pattern();
        let pattern2 = tk2._pick_pattern();
        let pattern3 = tk3._pick_pattern();
        let pattern4 = tk4._pick_pattern();
        
        assert_eq!(r"(?u)\b\w\w+\b", pattern1);
        assert_eq!(r"(?u)(\b\w\w+\b){2}", pattern2);
        assert_eq!(r"(?u)(\b\w\w+\b){3}", pattern3);
        assert_eq!("None", pattern4);
        
        println!("Unigram: {:?}", pattern1);
        println!("Bigram: {:?}", pattern2);
        println!("Trigram: {:?}", pattern3);
        println!("Invalid: {:?}", pattern4);
    }

    #[test]
    fn test_tokenize(){
        let corpus = vec![
            "This is the first document.",
            "This is the second second document.",
            "And the third one.",
            "Is this the first document?",
        ];

        let tk1 = Tokenizer::new(1);
        let tk2 = Tokenizer::new(2);
        let tk3 = Tokenizer::new(3);
        // let tk4 = Tokenizer::new(4);

        let tokens1 = tk1.tokenize(corpus.clone());
        let tokens2 = tk2.tokenize(corpus.clone());
        let tokens3 = tk3.tokenize(corpus);
        // let tokens4 = tk1.tokenize(corpus);

        println!("Unigram: {:?}\n", tokens1);
        println!("Bigram: {:?}\n", tokens2);
        println!("Trigram: {:?}\n", tokens3);
        // println!("Invalid: {:?}", pattern4);
    }

    #[test]
    fn test_word_ngrams(){
        let doc = "This is the second second document.";
        let tokens = _tokenize_single_doc(doc);
        Tokenizer::_word_ngrams((2, 2), tokens);
    }
}
use std::vec::Vec;
use std::cmp;
use std::ops::Range;
use regex::Regex;
use std::string::String;
// use std::mem;

// pub fn string_Vec_to_string_slice_Vec<'a>(mut string_vec: Vec<String>) -> Vec<&'a str> {
//     let slice_vec: Vec<_> = string_vec.drain(..).map(|e| e.as_ref()).collect();
//     slice_vec
// }

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
    
    fn _word_ngrams<'a>(&self, tokens: Vec<&'a str>) -> Vec<String> {
        let (min_n, max_n) = self.ngram_range;
        let num_tokens = tokens.len() as u32;
        let n_iter_max = cmp::min(max_n+1, num_tokens + 1);

        let n_s = min_n..n_iter_max;

        let mut final_tokens = <Vec<String>>::new();
        for n in n_s {
            let i_s = 0..(num_tokens - n + 1);
            let mut sub_tokens: Vec<_> = i_s.map(|i| {
            // for i in 0..(num_tokens - n + 1) {
                let n = n as usize;
                let i = i as usize;
                let range = Range {start: i , end: (i + n) };
                tokens[range].join(" ").to_owned()
            }).collect();
            println!("{:?}", sub_tokens);
            final_tokens.append(&mut sub_tokens);
        }
        final_tokens
    }

    fn _tokenize_single_doc<'a>(&self, doc: &'a str) -> Vec<&'a str> {
        let token_pattern=r"(?u)\b\w\w+\b";//self._pick_pattern();
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
            _tokens = self._tokenize_single_doc(doc);
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



#[cfg(test)]
mod tests {
    use super::*;

    // #[test]
    // fn test_tokenize(){
    //     let corpus = vec![
    //         "This is the first document.",
    //         "This is the second second document.",
    //         "And the third one.",
    //         "Is this the first document?",
    //     ];

    //     let tk1 = Tokenizer::new(1);
    //     let tk2 = Tokenizer::new(2);
    //     let tk3 = Tokenizer::new(3);
    //     // let tk4 = Tokenizer::new(4);

    //     let tokens1 = tk1.tokenize(corpus.clone());
    //     let tokens2 = tk2.tokenize(corpus.clone());
    //     let tokens3 = tk3.tokenize(corpus);
    //     // let tokens4 = tk1.tokenize(corpus);

    //     println!("Unigram: {:?}\n", tokens1);
    //     println!("Bigram: {:?}\n", tokens2);
    //     println!("Trigram: {:?}\n", tokens3);
    //     // println!("Invalid: {:?}", pattern4);
    // }

    // #[test]
    // fn test_word_ngrams(){
    //     let doc = "This is the second second document.";

    //     let tk = Tokenizer{ngram_range: (1, 2)};
    //     let tokens = tk._tokenize_single_doc(doc);
    //     let res = tk._word_ngrams(tokens);
    //     println!("{:?}", res);
    // }
}
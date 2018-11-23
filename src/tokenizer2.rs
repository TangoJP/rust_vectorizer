use std::vec::Vec;
use std::cmp;
use std::ops::Range;
use std::string::String;
use regex::Regex;


// *** Vec<String> version of tokenizer

/// A struct used to tokenize a collection of documents (i.e. Vector of
/// string slices). It has one field, n_gram, to specify the n-gram option
/// for tokenization.
/// 
pub struct Tokenizer {
    ngram_range: (u32, u32),
    case: String,
}

impl Tokenizer{
    /// Create a new instance of Tokenizer with n_gram (u32) input.
    /// 
    pub fn new(ngram_range: (u32, u32)) -> Tokenizer {
        Tokenizer {
            ngram_range: ngram_range,
            case: "lower".to_string(),
        }
    }
    
    // Takes a tokenized &str (i.e. Vev<&str>) and returns N-gram toklens for
    // 'N's specified by ngram_range (e.g. if ngram_range = (1, 3), uni-, bi-
    // and tri-grams will be created.
    //
    fn _word_ngrams(&self, tokens: Vec<&str>) -> Vec<String> {
        let (min_n, max_n) = self.ngram_range;              // Range of N-grams
        let num_tokens = tokens.len() as u32;               // Number of unigram tokens
        let n_iter_max = cmp::min(max_n+1, num_tokens + 1); // Max N-gram to get
        let n_s = min_n..n_iter_max;                        // Range for N of N-gram
        let mut final_tokens = <Vec<String>>::new();        // Declare Vector to return
        for n in n_s {                                  // iterate over different N's
            let i_s = 0..(num_tokens - n + 1);          // iterate over i-th unigram position
            let mut sub_tokens: Vec<_> = i_s.map(|i| {
                let range = Range {start: i as usize, end: (i + n) as usize};
                tokens[range].join(" ")
            }).collect();
            final_tokens.append(&mut sub_tokens);
        }
        final_tokens
    }

    // tokenize a single doc (i.e. &str) by regex followed by _word_ngrams().
    // It returns Vec<String>
    //
    fn _tokenize_single_doc<'a>(&self, doc: &'a str) -> Vec<String> {
        let token_pattern=r"(?u)\b\w\w+\b";
        let _re = Regex::new(token_pattern).unwrap();
        let _tokens: Vec<&str> = _re.find_iter(doc)
            .map(|f| f.as_str())
            .collect();
        let mut _ngrams_tokens: Vec<_> = self._word_ngrams(_tokens);
        if self.case == "lower".to_string() {
            _ngrams_tokens = _ngrams_tokens.iter()
                .map(|s| s.to_lowercase())
                .collect();
        }
        _ngrams_tokens
    }

    /// It takes a collection of documents (i.e. Vec<&str>), tokenize each 
    /// doc with specified ngram_range, and collect returned Vec<String> 
    /// into a Vec (Vec<Vec<String>> returned.
    ///
    pub fn tokenize<'a>(&self, docs: Vec<&'a str>) -> Vec<Vec<String>> {
        let mut _tokenized_docs: Vec<Vec<String>> = Vec::new();
        for doc in docs {
            let mut _tokens: Vec<String> ;
            _tokens = self._tokenize_single_doc(doc);
            _tokenized_docs.push(_tokens)
        };
        _tokenized_docs
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenize(){
        let corpus = vec![
            "This is the first document.",
            "This is the second second document.",
            "And the third one.",
            "Is this the first document?",
        ];

        let tk1 = Tokenizer::new((1, 1));
        let tk2 = Tokenizer::new((2, 2));
        let tk3 = Tokenizer::new((3, 3));
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

    // #[test]
    // fn test_word_ngrams(){
    //     let doc = "This is the second second document.";

    //     let tk = Tokenizer2{ngram_range: (2, 3)};
    //     let tokens = tk._tokenize_single_doc(doc);
    //     // let res = tk._word_ngrams(tokens);
    //     println!("{:?}", tokens);
    // }
}
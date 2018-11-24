//! 
//! Current issues:
//! - Only tokenize into words and/or n-grams of words
//! - Accents, etc are not handled

use std::vec::Vec;
use std::cmp;
use std::ops::Range;
use std::string::String;
use regex::Regex;


/// A struct used to tokenize a collection of documents (i.e. Vector of
/// string slices). It has two fields, ngram_range and case to specify the 
/// n-gram option for tokenization, and the letter case treatment, respectively.
/// 
pub struct Tokenizer {
    /// A range of n-values for n-grams to be included. For example
    /// ngram_range: (1, 3) would include uni-, bi-, and tr-grams. Lower bound
    /// should be above zero (unless -grams are to be included) and cannot
    /// be larger than the upper bound in order for valid tokens to be returned.
    pub ngram_range: (u32, u32), 

    /// The case of the resulting tokens. Default is no conversion. Options 
    /// are "upper" and "lower". Other inputs will use default. 
    pub case: String,
}

impl Tokenizer{
    /// Create a new instance of Tokenizer with ngram_range ((u32, u32)) input.
    pub fn new(ngram_range: (u32, u32), case: &str) -> Tokenizer {
        // Check the specifications
        let (min_n, max_n) = ngram_range;
        if min_n == 0 {
            println!("WARNING: Lower bound of of ngram_range set to 0. \
                0-grams will be included.");
        } else if min_n > max_n {
            println!("WARNING: Lower bound of of ngram_range \
                larger than upper bound. Empty tokens will be returned.")
        } else {}

        // Return tokenizer
        Tokenizer {
            ngram_range: ngram_range,
            case: case.to_string(),
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
    fn _tokenize_single_doc<'a>(&self, doc: &'a str) -> Vec<String> {
        // Split into words/tokens
        let token_pattern=r"(?u)\b\w\w+\b";
        let _re = Regex::new(token_pattern).unwrap();
        let _tokens: Vec<&str> = _re.find_iter(doc)
            .map(|f| f.as_str())
            .collect();

        // Case conversion
        let mut _ngrams_tokens: Vec<_> = self._word_ngrams(_tokens);
        match self.case.as_str() {  // could refactor to return conversion function
            "upper" => {
                _ngrams_tokens = _ngrams_tokens.iter()
                .map(|s| s.to_uppercase())
                .collect();},
            "lower" => {
                _ngrams_tokens = _ngrams_tokens.iter()
                .map(|s| s.to_lowercase())
                .collect();},
            _ => {
                _ngrams_tokens = _ngrams_tokens.iter()
                .map(|s| s.to_string())
                .collect();},
        }

        _ngrams_tokens
    }

    /// It takes a collection of documents (i.e. Vec<&str>), tokenize each 
    /// doc with the Tokenizer's specs, and collect returned Vec<String> 
    /// into a Vec (Vec<Vec<String>> returned.
    ///
    /// # Examples
    /// ```
    /// use vectorizer::tokenizer::Tokenizer;
    /// 
    /// // Collection of documents as Vec<&str>
    /// let corpus = vec![
    ///     "This is the first document.",
    ///     "This is the second second document.",
    ///     "And the third one.",
    ///     "Is this the first document?",
    /// ];
    ///
    /// // Set up tokenizers with different settings
    /// let tk1 = Tokenizer::new((1, 1), "none");   // Unigrams with no case setting
    /// let tk2 = Tokenizer::new((2, 2), "lower");  // Bigrams with lowercase
    /// let tk3 = Tokenizer::new((1, 3), "upper");  // Uni~Trigrams with uppercase
    /// 
    /// // Tokenize with tokenizers
    /// let tokens1 = tk1.tokenize(corpus.clone());
    /// let tokens2 = tk2.tokenize(corpus.clone());
    /// let tokens3 = tk3.tokenize(corpus);
    /// 
    /// // Print results
    /// println!("Uni-gram ({:?}): {:?}\n", tk1.case, tokens1);
    /// println!("Bi-gram ({:?}): {:?}\n", tk2.case, tokens2);
    /// println!("Uni~Tri-gram ({:?}): {:?}\n", tk3.case, tokens3);
    /// ```
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

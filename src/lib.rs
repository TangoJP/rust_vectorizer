#[macro_use]
extern crate ndarray;
extern crate regex;
extern crate num;
// extern crate indexmap;

/// tokenization of &str documents.
/// 
pub mod tokenizer;

/// Custom extension functions for ndarray
/// 
pub mod ndarray_extension;

/// Count vectorizer module
/// 
pub mod countvectorizer;

/// Td-Idf vectorizer module
/// 
pub mod tfidfvectorizer;


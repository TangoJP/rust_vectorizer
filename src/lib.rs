#[macro_use]
extern crate ndarray;
extern crate regex;
extern crate num;
// extern crate indexmap;

/// tokenization of &str documents
pub mod tokenizer;

/// custom extension codes for ndarray
pub mod ndarray_extension;

/// countvectorizer module
pub mod countvectorizer;

/// tdidfvectorizer module
pub mod tfidfvectorizer;


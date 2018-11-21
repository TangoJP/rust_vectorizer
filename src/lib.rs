#[macro_use]
extern crate ndarray;
extern crate regex;
extern crate num;
// extern crate indexmap;

pub mod tokenizer;          // implement trait related to tokenization
pub mod ndarray_extension;  // custom extension codes for ndarray
pub mod countvectorizer;    // countvectorizer module
pub mod tfidfvectorizer;    // tdidfvectorizer module



// extern crate regex;
extern crate vectorizer;

#[macro_use]
extern crate ndarray;

use ndarray::prelude::*;
// use std::vec::Vec;
// use std::collections::HashMap;
// use regex::Regex;
// use vectorizer:: CountVectorizer;
// use vectorizer:: tokenizer;

fn main() {
    let x = array![
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0]
    ];
    let y = array![
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [10, 11, 12]
    ];

    let tf1 = vectorizer::_get_term_frequency(x.clone(), "linear");
    let tf2 = vectorizer::_get_term_frequency(y.clone(), "linear");
    println!("X TF:\n{:?}", tf1);
    println!("Y TF:\n{:?}", tf2);

    let df1 = vectorizer::_get_document_frequency(x.clone());
    let df2 = vectorizer::_get_document_frequency(y.clone());
    let mat1 = vectorizer::vec2diagonal(df1);
    let mat2 = vectorizer::vec2diagonal(df2);
    println!("X Mat1:\n{:?}", mat1);
    println!("Y Mat2:\n{:?}", mat2);

    let temp_tfidf1 = vectorizer::_get_idf_matrix(x, 0);
    let temp_tfidf2 = vectorizer::_get_idf_matrix(y, 0);
    println!("X transformed:\n{:?}", temp_tfidf1);
    println!("Y transformed:\n{:?}", temp_tfidf2);
}

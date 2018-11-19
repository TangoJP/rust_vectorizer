
// extern crate regex;
extern crate vectorizer;

#[macro_use]
extern crate ndarray;

// use ndarray::prelude::*;
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
        [2, 0, 0]
    ];
    let y = array![
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [10, 11, 12]
    ];
    // let z = y.clone().mapv(|element| element as f64);
    // println!("{:?}", 1./z);

    // let temp_tfidf1 = vectorizer::_get_idf_matrix(x, 1);
    // let temp_tfidf2 = vectorizer::_get_idf_matrix(y, 0);
    // println!("X transformed:\n{:?}", temp_tfidf1);
    // println!("Y transformed:\n{:?}", temp_tfidf2);

    let tfidf1 = vectorizer::tfidfvectorizer::tfidi_transform(x, "linear", 0);
    let tfidf2 = vectorizer::tfidfvectorizer::tfidi_transform(y, "linear", 0);
    println!("X tf-idf:\n{:?}", tfidf1);
    println!("Y tf-idf:\n{:?}", tfidf2);
}

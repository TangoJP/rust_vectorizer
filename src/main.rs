
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

    let tf1 = vectorizer::_get_term_frequency(x.clone(), "ln");
    let tf2 = vectorizer::_get_term_frequency(y.clone(), "ln");
    println!("X TF{:?}", tf1);
    println!("Y TF{:?}", tf2);

    let df1 = vectorizer::_get_document_frequency(x);
    let df2 = vectorizer::_get_document_frequency(y);
    println!("X DF{:?}", df1);
    println!("Y DF{:?}", df2);
}

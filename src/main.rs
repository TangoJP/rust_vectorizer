
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
use vectorizer::ndarray_extension;

fn main() {
    let x = array![
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 2.0],
        [4.0, 0.0, 3.0]];
    let rnorms = ndarray_extension::row_norms(x.clone());
    println!("L2 Norms = {:?}", rnorms);

    let l2 = ndarray_extension::l2_normalize(x.clone());
    println!("L2 Matrix = {:?}", l2);

    let l1 = ndarray_extension::l1_normalize(x);
    println!("L1 Matrix = {:?}", l1);
}

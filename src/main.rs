//#[macro_use]
extern crate regex;
extern crate vectorizer;

// use std::vec::Vec;
// use std::collections::HashMap;
// use regex::Regex;
use vectorizer:: CountVectorizer;

fn main() {

    let fruits_str = "apple, banana, apple, banana, orange, three, \
                      apple. apple, banana, orange, orange, ONE, three";
    let numbers_str = "one, two, three, two, three, apple, three. three, four, four, ONE";
    let mut docs: Vec<&str> = Vec::new();
    docs.push(fruits_str);
    docs.push(numbers_str);
    //println!("{:?}", docs);
    
    // Testing CountVectorizer
    let mut vectorizer = CountVectorizer::new();
    let x = vectorizer.fit_transform(docs);
    
    // Print original docs
    println!("=== Example1 ===");
    println!("Doc0 :{:?}", fruits_str);
    println!("Doc1 :{:?}", numbers_str);

    // print Word_id: Word correspondence
    let vocabulary_inverted = vectorizer.reverse_vocabulary_hashmap();
    for i in 0..vectorizer.vocabulary_.len() {
        println!("(Word_id: Word) : ({:?}:{:?})", i, vocabulary_inverted[&(i as i32)]);
    }
    
    println!("CountVector :\n{:?}", x);
    println!("\n");

    println!("=== Example2 (From scikit-learn example)===");
    // all vec strings are convereted to lowercase
    let vec = vec![
        "this is the first document.",
        "this document is the second document.",
        "and this is the third one.",
        "is this the first document?"
    ];
    println!("Docs :\n{:?}", vec);
    let mut vectorizer2 = CountVectorizer::new();
    let x2 = vectorizer2.fit_transform(vec);

    // print Word_id: Word correspondence
    let vocabulary_inverted2 = vectorizer2.reverse_vocabulary_hashmap();
    for i in 0..vectorizer2.vocabulary_.len() {
        println!("(Word_id: Word) : ({:?}:{:?})", i, vocabulary_inverted2[&(i as i32)]);
    }

    println!("CountVector :\n{:?}", x2);
    println!("\n");
}

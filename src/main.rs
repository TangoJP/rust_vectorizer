//#[macro_use]
extern crate regex;
extern crate vectorizer;

// use std::vec::Vec;
// use std::collections::HashMap;
// use regex::Regex;
use vectorizer:: CountVectorizer;

fn main() {

    let fruits_str = "apple, banana, apple, banana, orange, \
                      apple. apple, banana, orange, orange";
    let numbers_str = "one, two, three, two, three, three. three, four, four, ONE";
    let mut docs: Vec<&str> = Vec::new();
    docs.push(fruits_str);
    docs.push(numbers_str);
    //println!("{:?}", docs);
    
    // Testing CountVectorizer
    let mut vectorizer = CountVectorizer::new();

    let X = vectorizer.fit_transform(docs);
    println!("(word_id: count) :{:?}", X);
    println!("(Word: word_id) : {:?}", vectorizer.vocabulary_);
}

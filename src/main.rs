//#[macro_use]
extern crate regex;
extern crate vectorizer;

// use std::vec::Vec;
use std::collections::HashMap;
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
    println!("Doc0 :{:?}", fruits_str);
    println!("Doc1 :{:?}", numbers_str);
    println!("\n");

    // print Word_id: Word correspondence
    let vocab_size = vectorizer.vocabulary_.len();
    let mut vocabulary_inverted: HashMap<i32, &str> = HashMap::new();
    for (k, v) in vectorizer.vocabulary_.iter() {
        vocabulary_inverted.insert(*v, k);
    }
    for i in 0..vocab_size {
        println!("(Word_id: Word) : ({:?}:{:?})", i, vocabulary_inverted[&(i as i32)]);
    }
    
    println!("CountVector :\n{:?}", x);
    println!("\n");

    // let array = vectorizer._sort_vocabulary_count(X);
    // println!("{:?}", array);

}

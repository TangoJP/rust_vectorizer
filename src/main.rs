//#[macro_use]
extern crate regex;

use std::vec::Vec;
//use std::collections::HashSet;
use regex::Regex;

fn main() {
    let token_pattern=r"(?u)\b\w\w+\b";
    let re = Regex::new(token_pattern).unwrap();
    let fruits_str = "apple, banana, apple, banana, orange, apple.";
    //let mut fruits = Vec::new();

    let fruits = re.captures_iter(fruits_str)
        .collect::<Vec<_>>();
    let fruits = fruits
        .iter()
        .map(|f| &f[0])
        .collect::<Vec<_>>();

    // println!("Without function\n{:?}", fruits);


    fn _tokenize1(doc: &str) {
        let token_pattern=r"(?u)\b\w\w+\b";
        let re = Regex::new(token_pattern).unwrap();
        let tokens = re.captures_iter(doc)
            .collect::<Vec<_>>();
        let tokens: Vec<&str> = tokens
            .iter()
            .map(|f| &f[0])
            .collect();
        println!("Tokenizer1: {:?}", tokens);
        
    };

    fn _tokenize2(doc: &str) -> Vec<&str> {
        let token_pattern=r"(?u)\b\w\w+\b";
        let re = Regex::new(token_pattern).unwrap();
        let _captures: Vec<&str> = re.find_iter(doc)
            .map(|f| f.as_str())
            .collect();
        _captures
    };

    _tokenize1(fruits_str);
    let captures = _tokenize2(fruits_str);
    println!("Tokenizer2: {:?}", captures)
    //println!("With _tokenize1()\n{:?}", _tokenize1(fruits_str));
    // println!("With _tokenize2()\n{:?}", _tokenize2(fruits_str));


        // tokenize a string into a list of words
    
    //fn _build_vocabulary(tokens: &Vec) -> vocabulary: HashMap<&str, i32> {};
        // create a HashSet containing vocabulary in the Vect
        // {'word': word_index}
        // if word not in Map: Map[word] = some_unused_ind
        // else: nothing

    //fn _count_vocab(tokens: &Vec, vocabulary: HashSet<&str> {};
        // count number of appearances of each word in the token

}

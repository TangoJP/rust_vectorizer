//#[macro_use]
extern crate regex;

use std::vec::Vec;
//use std::collections::HashSet;
use regex::Regex;

fn main() {
    let fruits_str = "apple, banana, apple, banana, orange, apple.";

    fn _tokenize(doc: &str) -> Vec<&str> {
        let token_pattern=r"(?u)\b\w\w+\b";
        let _re = Regex::new(token_pattern).unwrap();
        let _tokens: Vec<&str> = _re.find_iter(doc)
            .map(|f| f.as_str())
            .collect();
        _tokens
    };

    let tokens = _tokenize(fruits_str);
    println!("Tokenizer: {:?}", tokens)
    
    
    //fn _build_vocabulary(tokens: &Vec) -> vocabulary: HashMap<&str, i32> {};
        // create a HashSet containing vocabulary in the Vect
        // {'word': word_index}
        // if word not in Map: Map[word] = some_unused_ind
        // else: nothing

    //fn _count_vocab(tokens: &Vec) -> vocabulary: HashSet<&str> {};
        // count number of appearances of each word in the token

}

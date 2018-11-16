//#[macro_use]
extern crate regex;

use std::vec::Vec;
use std::collections::HashMap;
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

    fn _build_vocabulary(tokens: Vec<&str>) -> HashMap<&str, i32> {
        let mut vocabulary = HashMap::new();
        let mut counter: i32 = 1;
        for token in tokens {
            // println!("Token#{:?}: {:?}", &counter, &token);
            if !vocabulary.contains_key(token) {
                vocabulary.insert(token.clone(), counter.clone());
                counter = counter + 1;
            }
        };
        vocabulary
    }
    
    fn _build_vocabulary_and_count(tokens: Vec<&str>) {//-> HashMap<i32, i32> {
        let mut vocabulary = HashMap::new();
        let mut vocab_count = HashMap::new();
        vocabulary = _build_vocabulary(tokens.clone());

        // let mut counter: i32 = 1;
        for token in tokens {
            let token_ind = vocabulary[token];
            // println!("{:?}: {:?}", token_ind, token);
            
            if !vocab_count.contains_key(&token_ind) {
                vocab_count.insert(token_ind, 1);
            }
            else {
                *vocab_count.get_mut(&token_ind).unwrap() += 1;
            };
        };
        println!("{:?}", vocab_count);

    };

    let tokens = _tokenize(fruits_str);
    println!("Tokenizer: {:?}", tokens);
    println!("Vocabulary: {:?}", _build_vocabulary(tokens.clone()));
    _build_vocabulary_and_count(tokens);

    //fn _count_vocab(tokens: &Vec) -> vocabulary: HashSet<&str> {};
        // count number of appearances of each word in the token

}

//#[macro_use]
extern crate regex;
extern crate vectorizer;

use std::vec::Vec;
use std::collections::HashMap;
use regex::Regex;
use vectorizer:: CountVectorizer;

fn main() {

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
    
    fn _build_vocabulary_and_count(tokens: Vec<&str>) -> HashMap<i32, i32> {
        let vocabulary = _build_vocabulary(tokens.clone());

        let mut vocab_count = HashMap::new();

        for token in tokens {
            let token_ind = vocabulary[token];
            if !vocab_count.contains_key(&token_ind) {
                vocab_count.insert(token_ind, 1);
            }
            else {
                *vocab_count.get_mut(&token_ind).unwrap() += 1;
            };
        };
        // println!("{:?}", vocab_count);
        vocab_count
    };

    let fruits_str = "apple, banana, apple, banana, orange, apple.";
    let numbers_str = "one, two, three, two, three, three.";

    let tokens1 = _tokenize(fruits_str.clone());
    let tokens2 = _tokenize(numbers_str.clone());
    println!("=== parsing fruits ===");
    println!("Tokenizer: {:?}", tokens1);
    println!("Vocabulary: {:?}", _build_vocabulary(tokens1.clone()));
    println!("Vocabulary Counts: {:?}\n", _build_vocabulary_and_count(tokens1));

    println!("=== parsing numbers ===");
    println!("Tokenizer: {:?}", tokens2);
    println!("Vocabulary: {:?}", _build_vocabulary(tokens2.clone()));
    println!("Vocabulary Counts: {:?}\n", _build_vocabulary_and_count(tokens2));


    let mut vectorizer1 = CountVectorizer::new();
    let mut vectorizer2 = CountVectorizer::new();
    vectorizer1.fit(fruits_str);
    vectorizer2.fit(numbers_str);
    println!("=== fit() with Vectorizer ===");
    println!("Vectorizer Vocabulary: {:?}", vectorizer1.vocabulary_);
    println!("Vectorizer Vocabulary: {:?}\n", vectorizer2.vocabulary_);

    let mut vectorizer3 = CountVectorizer::new();
    let mut vectorizer4 = CountVectorizer::new();
    vectorizer3.fit_transform(fruits_str);
    vectorizer4.fit_transform(numbers_str);
    println!("=== fit_transform() with Vectorizer ===");
    println!("Vectorizer Vocabulary: {:?}, Counts:{:?}", 
            vectorizer3.vocabulary_, vectorizer3.vocabulary_counts_);
    println!("Vectorizer Vocabulary: {:?}, Counts:{:?}\n", 
            vectorizer4.vocabulary_, vectorizer4.vocabulary_counts_);

    //fn _count_vocab(tokens: &Vec) -> vocabulary: HashSet<&str> {};
        // count number of appearances of each word in the token

}

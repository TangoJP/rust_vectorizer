extern crate vectorizer;

use vectorizer::countvectorizer::CountVectorizer;

#[test]
fn example1() {
    let fruits_str = "apple, banana, apple, banana, orange, three, \
                      apple. apple, banana, orange, orange, one, three";
    let numbers_str = "one, two, three, two, three, apple, three. three, four, four, one";
    let mut docs1: Vec<&str> = Vec::new();
    docs1.push(fruits_str);
    docs1.push(numbers_str);

    // Check vocabulary size
    let mut vectorizer = CountVectorizer::new();
    assert_eq!(0, vectorizer.vocabulary_.len());    // Before fit

    let x = vectorizer.fit_transform(docs1.clone());
    assert_eq!(7, vectorizer.vocabulary_.len());    // After fit

    let apple_col_index = vectorizer.vocabulary_["apple"];
    assert_eq!(0, apple_col_index);
    assert_eq!(4, x[[0, apple_col_index as usize]]);
    assert_eq!(1, x[[1, apple_col_index as usize]]);

    // Print original docs
    println!("=== Example1 ===");
    println!("Doc0 :{:?}", fruits_str);
    println!("Doc1 :{:?}", numbers_str);

    // print Word_id: Word correspondence
    let vocabulary_inverted = vectorizer.reverse_vocabulary_hashmap();
    for i in 0..vectorizer.vocabulary_.len() {
        println!("(Word_id: Word) : ({:?}:{:?})", i, vocabulary_inverted[&(i as u64)]);
    }
    
    println!("CountVector :\n{:?}", x);
    println!("\n");

}

#[test]
fn example2() {
    // all vec strings were convereted to lowercase for this example
    let docs2 = vec![
        "ああ いい うう", 
        "ああ いい ええ",
        "this is the first document.",
        "this document is the second document.",
        "and this is the third one.",
        "is this the first document?"
    ];

    // Check vocabulary size
    let mut vectorizer = CountVectorizer::new();
    assert_eq!(0, vectorizer.vocabulary_.len());    // Before fit

    let x2 = vectorizer.fit_transform(docs2.clone());
    assert_eq!(13, vectorizer.vocabulary_.len());    // After fit

    println!("=== Example2 (From scikit-learn example Plus-alpha)===");

    // print Word_id: Word correspondence
    let vocabulary_inverted = vectorizer.reverse_vocabulary_hashmap();
    for i in 0..vectorizer.vocabulary_.len() {
        println!("(Word_id: Word) : ({:?}:{:?})", i, vocabulary_inverted[&(i as u64)]);
    }

    println!("CountVector :\n{:?}", x2);
    println!("\n");
}
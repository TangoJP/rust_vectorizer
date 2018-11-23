extern crate vectorizer;
// #[macro_use]
// extern crate ndarray;

// use vectorizer::tfidfvectorizer;
use vectorizer::tfidfvectorizer::TfidfVectorizer;
// use vectorizer::ndarray_extension;
// use ndarray::{Array1, Array2};


#[test]
fn test_fit_transform(){
    let fruits_str = "apple, banana, apple, banana, orange, three, \
                        apple. apple, banana, orange, orange, one, three";
    let numbers_str = "one, two, three, two, three, apple, three. three, four, four, one";
    let mut docs1: Vec<&str> = Vec::new();
    docs1.push(fruits_str);
    docs1.push(numbers_str);

    let mut vectorizer = TfidfVectorizer::new((1, 2), "lower".to_string());
    let tfidf = vectorizer.fit_transform(docs1);
    
    assert_eq!(24, vectorizer.vocabulary_.len());
    assert_eq!((2, 24), tfidf.dim());

    println!("Vocabulary: {:?}",vectorizer.vocabulary_);
    println!("Tf-Idf Matrix:\n{:?}", tfidf);

}
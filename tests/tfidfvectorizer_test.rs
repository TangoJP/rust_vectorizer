extern crate vectorizer;
#[macro_use]
extern crate ndarray;

// use vectorizer::tfidfvectorizer;
use vectorizer::tfidfvectorizer::TfidfVectorizer;
use vectorizer::ndarray_extension;
// use ndarray::{Array1, Array2};


#[test]
fn test_vec2diagonal(){
    let vec1 = array![0.5, 0.25, 0.25];
    let vec2 = array![1.0, 1.0, 1.0];

    let mat1 = ndarray_extension::vec2diagonal(vec1);
    let mat2 = ndarray_extension::vec2diagonal(vec2);
    println!("=== Testing Diagonalization ===");
    println!("X Mat1:\n{:?}", mat1);
    println!("Y Mat2:\n{:?}", mat2);
    println!("\n");

    let ans1 = array![
        [0.5, 0.0, 0.0],
        [0.0, 0.25, 0.0],
        [0.0, 0.0, 0.25]];
    let ans2 = array![
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]];
    
    assert_eq!(ans1, mat1);
    assert_eq!(ans2, mat2);
}

#[test]
fn test_fit_transform(){
    let fruits_str = "apple, banana, apple, banana, orange, three, \
                        apple. apple, banana, orange, orange, one, three";
    let numbers_str = "one, two, three, two, three, apple, three. three, four, four, one";
    let mut docs1: Vec<&str> = Vec::new();
    docs1.push(fruits_str);
    docs1.push(numbers_str);

    let mut vectorizer = TfidfVectorizer::new();
    let tfidf = vectorizer.fit_transform(docs1, "linear", 1);
    
    assert_eq!(7, vectorizer.vocabulary_.len());
    assert_eq!((2, 7), tfidf.dim());
    println!("{:?}", tfidf);

}
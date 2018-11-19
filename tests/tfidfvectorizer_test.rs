extern crate vectorizer;
#[macro_use]
extern crate ndarray;

// use vectorizer::countvectorizer::CountVectorizer;
// use ndarray::{Array1, Array2};

#[test]
fn test_get_term_frequency() {
    let x = array![
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0]
    ];
    let y = array![
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [10, 11, 12]
    ];

    let tf1 = vectorizer::_get_term_frequency(x, "linear");
    let tf2 = vectorizer::_get_term_frequency(y, "linear");
    println!("=== Testing Term Frequency ===");
    println!("X TF:\n{:?}", tf1);
    println!("Y TF:\n{:?}", tf2);
    println!("\n");

    let ans1 = array![
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0]];
    let ans2 = array![
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
        [10.0, 11.0, 12.0]];

    assert_eq!(ans1, tf1);
    assert_eq!(ans2, tf2);
}

#[test]
fn test_get_document_frequency() {
    let x = array![
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0]
    ];
    let y = array![
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [10, 11, 12]
    ];

    let df1 = vectorizer::_get_document_frequency(x);
    let df2 = vectorizer::_get_document_frequency(y);
    println!("=== Testing Document Frequency ===");
    println!("X DF:\n{:?}", df1);
    println!("Y DF:\n{:?}", df2);
    println!("\n");

    let ans1 = array![0.5, 0.25, 0.25];
    let ans2 = array![1.0, 1.0, 1.0];

    assert_eq!(ans1, df1);
    assert_eq!(ans2, df2);
}

#[test]
fn test_vec2diagonal(){
    let vec1 = array![0.5, 0.25, 0.25];
    let vec2 = array![1.0, 1.0, 1.0];

    let mat1 = vectorizer::vec2diagonal(vec1);
    let mat2 = vectorizer::vec2diagonal(vec2);
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
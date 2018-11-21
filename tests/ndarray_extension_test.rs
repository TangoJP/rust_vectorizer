extern crate vectorizer;
#[macro_use]
extern crate ndarray;

use vectorizer::ndarray_extension;

#[test]
fn test_convert_to_f64(){
    let x = array![
        [1, 2, 3],
        [2, 3, 4],
        [5, 6, 7]];
    let x_f64 = x.clone().mapv(|e| e as f64);
    let y = ndarray_extension::convert_matrix_to_f64(x.clone());
    assert_eq!(x_f64, y);

    println!("=== Testing Conversion to f64 ===");
    println!("Original:\n{:?}", x);
    println!("f64 version:\n{:?}", y);

}

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
fn test_vec2diagonal2(){
    let vec1 = array![0.5, 0.25, 0.25];
    let vec2 = array![1.0, 1.0, 1.0];
    let vec3 = array![2, 2, 2];//[2_u32, 2_u32, 2_u32];

    let mat1 = ndarray_extension::vec2diagonal2(vec1);
    let mat2 = ndarray_extension::vec2diagonal2(vec2);
    let mat3 = ndarray_extension::vec2diagonal2(vec3);

    let ans1 = array![
        [0.5, 0.0, 0.0],
        [0.0, 0.25, 0.0],
        [0.0, 0.0, 0.25]];
    let ans2 = array![
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]];
    let ans3 = array![
        [2., 0., 0.],
        [0., 2., 0.],
        [0., 0., 2.]];

    assert_eq!(ans1, mat1);
    assert_eq!(ans2, mat2);
    assert_eq!(ans3, mat3);

    println!("=== Testing Diagonalization with f64 conversion ===");
    println!("X Mat1:\n{:?}", mat1);
    println!("Y Mat2:\n{:?}", mat2);
    println!("Y Mat2:\n{:?}", mat3);
    println!("\n");
}

#[test]
fn test_l1_normalizatin(){
    let x = array![
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 2.0],
        [4.0, 0.0, 3.0]];

    let ans_x = array![
        [1.0/3., 1.0/3., 1.0/3.],
        [0.0, 1.0/3., 2.0/3.],
        [4.0/7., 0.0, 3.0/7.]];
    
    let l1 = ndarray_extension::l1_normalize(x);
    assert_eq!(ans_x, l1);
    println!("=== Testing L1 Normalization ===");
    println!("L1 Matrix = {:?}", l1);
}

#[test]
fn test_l2_normalization(){
    let x = array![
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 2.0],
        [4.0, 0.0, 3.0]];

    let n1 = f64::sqrt(3.);
    let n2 = f64::sqrt(5.);
    let n3 = f64::sqrt(25.);

    let ans_x = array![
        [1.0/n1, 1.0/n1, 1.0/n1],
        [0.0/n2, 1.0/n2, 2.0/n2],
        [4.0/n3, 0.0/n3, 3.0/n3]];
    let ans_norm = array![n1, n2, n3];

    let rnorms = ndarray_extension::row_l2_norms(x.clone());
    let l2 = ndarray_extension::l2_normalize(x.clone());

    assert_eq!(ans_norm, rnorms);
    assert_eq!(ans_x, l2);

    println!("=== Testing L2 Normalization ===");
    println!("L2 Norms = {:?}", rnorms);
    println!("L2 Matrix = {:?}", l2);

}
// use ndarray;
use ndarray::{Array1, Array2, Axis};
use num::{NumCast, Zero, ToPrimitive}; //{PrimInt, Unsigned, BigInt, BigRational};
use std::clone::Clone;


/// Convert an Array2<T> into an Array2<f64> of the content.
/// It binds the Type T with Clone + ToPrimitive
/// 
/// # Examples
/// ```
/// extern crate ndarray;
/// extern crate vectorizer;
/// 
/// use ndarray::{arr1, arr2};
/// use vectorizer::ndarray_extension;
/// 
/// let mut x = arr2(&[
///        [1, 2, 3],
///        [2, 3, 4],
///        [5, 6, 7]]);
/// x = x.clone().mapv(|e| e as u32);
/// let x_f64 = x.clone().mapv(|e| e as f64);
/// let y = ndarray_extension::convert_matrix_to_f64(x.clone());
/// assert_eq!(x_f64, y);
///
/// println!("=== Testing Conversion to f64 ===");
/// println!("u32 version:\n{:?}", x);
/// println!("f64 version:\n{:?}", y);
/// ```
/// 
pub fn convert_matrix_to_f64<T: Clone + ToPrimitive>(array: Array2<T>) -> Array2<f64> {
        let array_f64 = array.mapv(|e| e.to_f64().unwrap());
        array_f64
}

/// Count number of non-zero rows for each column. 
/// *** The non-zero columns for each row to be implemented
pub fn bincount(matrix: Array2<u64>)  -> Array1<f64> {
    let (num_rows, num_columns) = matrix.dim();
    let mut bincounts = Array1::<f64>::zeros(num_columns);
    for index_row in 0..num_rows {
        for index_col in 0..num_columns {
            if matrix[[index_row, index_col]] != 0 {
                bincounts[index_col] += 1.;
            }
        }
    }
    bincounts
}

/// Convert an Array1<T> into diagonal Array2<f64>.
/// It binds the Type T with Clone + ToPrimitive
/// 
/// # Examples
/// ```
/// extern crate ndarray;
/// extern crate vectorizer;
/// 
/// use ndarray::{arr1, arr2};
/// use vectorizer::ndarray_extension;
/// 
/// let vec1 = arr1(&[0.5, 0.25, 0.25]);
/// let vec2 = arr1(&[1, 1, 1]);
/// 
/// let ans1 = arr2(&[
///    [0.5, 0.0, 0.0],
///    [0.0, 0.25, 0.0],
///    [0.0, 0.0, 0.25]]);
/// let ans2 = arr2(&[
///     [1.0, 0.0, 0.0],
///     [0.0, 1.0, 0.0],
///     [0.0, 0.0, 1.0]]);
///
/// let mat1 = ndarray_extension::vec2diagonal(vec1);
/// let mat2 = ndarray_extension::vec2diagonal(vec2);
/// assert_eq!(ans1, mat1);
/// assert_eq!(ans2, mat2);
/// 
/// println!("=== Testing Diagonalization ===");
/// println!("X Mat1:\n{:?}", mat1);
/// println!("Y Mat2:\n{:?}", mat2);
/// println!("\n");
/// ```
/// 
pub fn vec2diagonal<T: Clone + Zero + ToPrimitive>(vector: Array1<T>) -> Array2<f64> {
    let length = vector.len();
    let mut matrix = Array2::<T>::zeros((length, length));
    for i in 0..length {
        matrix[[i, i]] = vector[i].clone();//T::from(vector[i].clone()).unwrap();
    }
    matrix.mapv(|e| e.to_f64().unwrap())
}

/// Calculates L2 norm for each row of an Array2<T>. An L2 norm is
/// calculated as sqrt(n1^2, n2^2, ..., nN^2), where ni denotes i-th element
/// in a row. The outpu will be an Array2<f64>
/// 
pub fn row_l2_norms<T: NumCast + Clone>(x: Array2<T>) -> Array1<f64> {
    let mut x_f64 = convert_matrix_to_f64(x);
    x_f64 = x_f64.mapv(|e| e.powi(2));
    let squared_sums = x_f64.sum_axis(Axis(1));
    squared_sums.mapv(f64::sqrt)
}

/// L1 normalize an Array2<T>. Each individual row is normalized. L1 norm
/// is the sum of the absolute values of all the elements in a row.
/// 
/// # Examples
/// ```
/// extern crate ndarray;
/// extern crate vectorizer;
/// 
/// use ndarray::{arr1, arr2};
/// use vectorizer::ndarray_extension;
/// 
/// let x = arr2(&[
///     [1.0, 1.0, 1.0],
///     [0.0, 1.0, 2.0],
///     [4.0, 0.0, 3.0]]);
/// 
/// let ans_x = arr2(&[
///     [1.0/3., 1.0/3., 1.0/3.],
///     [0.0, 1.0/3., 2.0/3.],
///     [4.0/7., 0.0, 3.0/7.]]);
/// 
/// let l1 = ndarray_extension::l1_normalize(x);
/// assert_eq!(ans_x, l1);
/// 
/// println!("=== Testing L1 Normalization ===");
/// println!("L1 Matrix = {:?}", l1);
/// ```
/// 
pub fn l1_normalize<T: NumCast + Clone>(x: Array2<T>) -> Array2<f64> {
    let mut x_f64 = convert_matrix_to_f64(x);
    x_f64 = x_f64.mapv(f64::abs);
    let norms = x_f64.sum_axis(Axis(1));
    let num_rows = norms.len();
    let row_norms = norms.into_shape((num_rows, 1)).unwrap();
    x_f64 / row_norms
}

/// L2 normalize an Array2<T>. Each individual row is normalized. L2 norms
/// are calculated using row_l2_norms().
/// 
/// # Examples
/// ```
/// extern crate ndarray;
/// extern crate vectorizer;
/// 
/// use ndarray::{arr1, arr2};
/// use vectorizer::ndarray_extension;
/// 
/// let x = arr2(&[
///     [1.0, 1.0, 1.0],
///     [0.0, 1.0, 2.0],
///     [4.0, 0.0, 3.0]]);
/// 
/// let n1 = f64::sqrt(3.);
/// let n2 = f64::sqrt(5.);
/// let n3 = f64::sqrt(25.);
/// 
/// let ans_x = arr2(&[
///     [1.0/n1, 1.0/n1, 1.0/n1],
///     [0.0/n2, 1.0/n2, 2.0/n2],
///     [4.0/n3, 0.0/n3, 3.0/n3]]);
/// let ans_norm = arr1(&[n1, n2, n3]);
/// 
/// let rnorms = ndarray_extension::row_l2_norms(x.clone());
/// let l2 = ndarray_extension::l2_normalize(x.clone());
/// 
/// assert_eq!(ans_norm, rnorms);
/// assert_eq!(ans_x, l2);
/// 
/// println!("=== Testing L2 Normalization ===");
/// println!("L2 Norms = {:?}", rnorms);
/// println!("L2 Matrix = {:?}", l2);
/// ```
/// 
pub fn l2_normalize<T: NumCast + Clone>(x: Array2<T>) -> Array2<f64> {
    let x_f64 = convert_matrix_to_f64(x);
    let norms = row_l2_norms(x_f64.clone());
    let num_rows = norms.len();
    let row_norms = norms.into_shape((num_rows, 1)).unwrap();
    x_f64 / row_norms
}

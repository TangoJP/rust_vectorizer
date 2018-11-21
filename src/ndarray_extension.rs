// use ndarray;
use ndarray::{Array1, Array2, Axis};
use num::{NumCast, Zero, ToPrimitive}; //{PrimInt, Unsigned, BigInt, BigRational};
use std::clone::Clone;


/// Convert an Array2<T> into an Array2<f64> of the content.
/// It binds the Type T with Clone + ToPrimitive
/// 
pub fn convert_matrix_to_f64<T: Clone + ToPrimitive>(array: Array2<T>) -> Array2<f64> {
        let array_f64 = array.mapv(|e| e.to_f64().unwrap());
        array_f64
}

/// Convert an Array1<T> into diagonal Array2<f64>.
/// It binds the Type T with Clone + ToPrimitive
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
pub fn l2_normalize<T: NumCast + Clone>(x: Array2<T>) -> Array2<f64> {
    let x_f64 = convert_matrix_to_f64(x);
    let norms = row_l2_norms(x_f64.clone());
    let num_rows = norms.len();
    let row_norms = norms.into_shape((num_rows, 1)).unwrap();
    x_f64 / row_norms
}


// #[cfg(test)]
// mod tests {
//     use super::*;

//     // #[test]
//     // #[ignore]
//     // fn test_to_f64(){
//     //     let mut a_u64 = Array2::<u64>::ones((3, 3));
//     //     let mut a_f64 = Array2::<f64>::ones((3, 3));

//     //     // a_u64 = a_u64.mapv(|e| e as u64);
//     //     // a_f64 = a_f64.mapv(|e| e as f64);

//     //     assert_eq!(a_f64, a_u64.to_f64());
//     // }
// }
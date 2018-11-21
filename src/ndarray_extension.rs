// use ndarray;
use ndarray::{Array1, Array2, Axis};
use num::{Num, NumCast, Zero, Float, ToPrimitive}; //{PrimInt, Unsigned, BigInt, BigRational};
use std::clone::Clone;


/// Convert an Array2<O> into an Array2<f64> of the content (O for old type)
/// 
/// It binds the Type O (for old) with NumCast and Clone and f64 with NumCast
/// 
pub fn convert_matrix_to_f64<O, f64>(array: Array2<O>) -> Array2<f64> 
    where O: NumCast + Clone,
        f64: NumCast {
        let array_f64 = array.mapv(|e| f64::from(e).unwrap());
        array_f64
}

/// Convert an Array1<f64> into diagonal Array2<f64>.
/// 
/// The input must be Array1<f64>, and not Array1 of other element types.
/// Use vec2diagonal2() for generic underlying elements.
/// 
pub fn vec2diagonal(vector: Array1<f64>) -> Array2<f64>{
    let length = vector.len();
    let mut matrix = Array2::<f64>::zeros((length, length));
    for i in 0..length {
        matrix[[i, i]] = vector[i];
    }
    matrix
}

/// Convert an Array1<O> into diagonal Array2<f64> (O for old type)
/// 
/// Generic version of vec2diagonal(). Input will always be Array2<f64>
/// 
pub fn vec2diagonal2<O,  f64>(vector: Array1<O>) -> Array2<f64>
    where O: NumCast + Clone,
        f64: NumCast + Clone + Zero {
    let length = vector.len();
    let mut matrix = Array2::<f64>::zeros((length, length));
    for i in 0..length {
        matrix[[i, i]] = f64::from(vector[i].clone()).unwrap();
    }
    matrix
}

pub fn vec2diagonal3<O,  N>(vector: Array1<O>) -> Array2<f64>
    where O: Num + NumCast + Clone + Zero + ToPrimitive {
    let length = vector.len();
    let mut matrix = Array2::<O>::zeros((length, length));
    for i in 0..length {
        matrix[[i, i]] = vector[i].clone();//O::from(vector[i].clone()).unwrap();
    }
    matrix.mapv(|e| e.to_f64().unwrap())
}

/// Calculates L2 norm for each row of an Array2<f64>. An L2 norm is
/// calculated as sqrt(n1^2, n2^2, ..., nN^2), where ni denotes i-th element
/// in a row.
/// 
/// The input must be Array2<f64> (i.e. elements must be f64 type)
/// 
pub fn row_l2_norms(mut x: Array2<f64>) -> Array1<f64> {
    x = x.mapv(|e| e.powi(2));
    let squared_sums = x.sum_axis(Axis(1));
    squared_sums.mapv(f64::sqrt)
}

pub fn row_l2_norms2<O, f64>(x: Array2<O>) -> Array1<f64>
    where O: NumCast + Clone,
        f64: NumCast + Clone + Zero + Float {
    let mut x_f64 = convert_matrix_to_f64(x);
    x_f64 = x_f64.mapv(|e: f64| e.powi(2));
    let squared_sums = x_f64.sum_axis(Axis(1));
    squared_sums.mapv(f64::sqrt)
}

/// L1 normalize an Array2<f64>. Each individual row is normalized. L1 norm
/// is the sum of the absolute values of all the elements in a row.
/// 
pub fn l1_normalize(mut x: Array2<f64>) -> Array2<f64> {
    x = x.mapv(f64::abs);
    let norms = x.sum_axis(Axis(1));
    let num_rows = norms.len();
    let row_norms = norms.into_shape((num_rows, 1)).unwrap();
    x / row_norms
}

/// L2 normalize an Array2<f64>. Each individual row is normalized. L2 norms
/// are calculated using row_l2_norms().
/// 
pub fn l2_normalize(x: Array2<f64>) -> Array2<f64> {
    let norms = row_l2_norms(x.clone());
    let num_rows = norms.len();
    let row_norms = norms.into_shape((num_rows, 1)).unwrap();
    x / row_norms
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
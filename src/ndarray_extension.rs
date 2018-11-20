// use ndarray;
use ndarray::{Array1, Array2, Axis};
// use std::any::Any;
use std::convert::From;

// pub trait DataConvert {
//     fn to_f64<O>(self) -> Array2<f64>
//         where f64: std::convert::From<O>,
//              O: std::marker::Sized,
//              O: std::clone::Clone;
// }

// impl<O> DataConvert for Array2<O> {
//     fn to_f64<O>(self) -> Array2<f64> {
//         let a = self.mapv(|e| f64::from(e));
//         a
//     }
// }

// pub fn convert_numeric2f64<O>(mut e:O) -> f64 {
//     type Output = f64;

// pub fn convert_to_f64<O, N>(mut array: Array2<O>) -> Array2<N> {
//     type Output = f64;
//     where N: convert {
//         array = array.mapv(|e| N::from(e));
//         array
// }

pub fn vec2diagonal(vector: Array1<f64>) -> Array2<f64>{
    
    let length = vector.len();
    let mut matrix = Array2::<f64>::zeros((length, length));
    for i in 0..length {
        matrix[[i, i]] = vector[i];
    }
    matrix
}

pub fn row_norms(mut x: Array2<f64>) -> Array1<f64> {
    x = x.mapv(|e| e.powi(2));
    let squared_sums = x.sum_axis(Axis(1));
    squared_sums.mapv(f64::sqrt)
}

pub fn l1_normalize(mut x: Array2<f64>) -> Array2<f64> {
    x = x.mapv(f64::abs);
    let norms = x.sum_axis(Axis(1));
    let num_rows = norms.len();
    let row_norms = norms.into_shape((num_rows, 1)).unwrap();
    x / row_norms
}

pub fn l2_normalize(x: Array2<f64>) -> Array2<f64> {
    let norms = row_norms(x.clone());
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
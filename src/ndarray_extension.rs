use ndarray::{Array1, Array2, Axis};


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
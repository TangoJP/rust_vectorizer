use ndarray::{Array1, Array2};


pub fn vec2diagonal(vector: Array1<f64>) -> Array2<f64>{
    let length = vector.len();
    let mut matrix = Array2::<f64>::zeros((length, length));
    for i in 0..length {
        matrix[[i, i]] = vector[i];
    }
    matrix
}

fn _normalize_matrix() {}
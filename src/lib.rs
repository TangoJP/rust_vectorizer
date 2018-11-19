extern crate ndarray;
extern crate  regex;
// extern crate indexmap;

use ndarray::{Array1, Array2};
// use indexmap::IndexMap;

pub mod tokenizer;          // implement trait related to tokenization
pub mod countvectorizer;    // countvectorizer module


fn _normalize_matrix() {}

pub fn vec2diagonal(vector: Array1<f64>) -> Array2<f64>{
    let length = vector.len();
    let mut matrix = Array2::<f64>::zeros((length, length));
    for i in 0..length {
        matrix[[i, i]] = vector[i];
    }
    matrix
}

pub fn _get_term_frequency(countvector: Array2<u32>, method: &str) -> Array2<f64>{
    // First pass. For now, if method == "ln", takes natural log of each element
    // Otherwise, just take the original counts.
    // **Algorithm follows scikit-learn implementation

    let term_frequency = countvector.mapv(|element| element as f64);
    if method == "ln" {
        term_frequency.mapv(f64::ln) + 1.0  // addition of 1 per sklearn
    } else { 
        term_frequency
    }
}

pub fn _get_document_frequency(countvector: Array2<u32>) -> Array1<f64>{
    // First pass. Refactor to make it more efficient

    let (num_rows, num_columns) = countvector.dim();
    let mut document_frequency = Array1::<f64>::zeros(num_columns);
    for index_row in 0..num_rows {
        for index_col in 0..num_columns {
            if countvector[[index_row, index_col]] != 0 {
                document_frequency[index_col] += 1.;
            }
        }
    }
    document_frequency = document_frequency/(num_rows as f64);
    document_frequency
}

pub fn _get_idf_matrix(countvector: Array2<u32>, smooth_idf: u64) -> Array2<f64>{

    // get countvector dimension and get document frequency vector
    let (num_rows, _) = countvector.dim();
    let mut df = _get_document_frequency(countvector);

    // smoothe by smooth_idf (see sklearn)
    df = df + (smooth_idf as f64);
    let n_samples = (num_rows as f64) + (smooth_idf as f64);
 
    // Caclulate idf and convert to diagonal matrix
    let mut idf = n_samples / df;
    idf = idf.mapv(f64::ln) + 1.;
    vec2diagonal(idf)
}

pub fn tfidi_transform(countvector: Array2<u32>, tf_method: &str, smooth_idf: u64) -> Array2<f64> {

    let tf = _get_term_frequency(countvector.clone(), tf_method);
    let idf = _get_idf_matrix(countvector, smooth_idf);
    let tfidf = tf.dot(&idf);
    tfidf
}

// pub struct TfidfVectorizer<'a> {
//     // pub vocabulary_ : HashMap<&'a str, i32>,
// }

// impl<'a> TfidfVectorizer<'a> {}
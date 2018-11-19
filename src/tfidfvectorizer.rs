use std::collections::HashMap;
use ndarray::{Array1, Array2};
use ndarray_extension;
use countvectorizer::CountVectorizer;

pub struct TfidfVectorizer<'a> {
    pub vocabulary_: HashMap<&'a str, u32>,
}

impl<'a> TfidfVectorizer<'a> {
    //
    // All functions are public for now for testing purpose
    //

    pub fn new() -> TfidfVectorizer<'a> {
        let map: HashMap<&'a str, u32> = HashMap::new();

        // Return a new instance
        TfidfVectorizer {
            vocabulary_: map,
        }
    }

    fn _create_countvector(&mut self, docs: Vec<&'a str>) -> Array2<u32> {
        let mut count_vectorizer = CountVectorizer::new();
        let countvector = count_vectorizer.fit_transform(docs);
        self.vocabulary_ = count_vectorizer.vocabulary_;
        countvector
    }

    fn _get_term_frequency(countvector: Array2<u32>, method: &str) -> Array2<f64>{
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

    fn _get_document_frequency(countvector: Array2<u32>) -> Array1<f64>{
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

    fn _get_idf_matrix(countvector: Array2<u32>, smooth_idf: u64) -> Array2<f64>{

        // get countvector dimension and get document frequency vector
        let (num_rows, _) = countvector.dim();
        let mut df = TfidfVectorizer::_get_document_frequency(countvector);

        // smoothe by smooth_idf (see sklearn)
        df = df + (smooth_idf as f64);
        let n_samples = (num_rows as f64) + (smooth_idf as f64);
    
        // Caclulate idf and convert to diagonal matrix
        let mut idf = n_samples / df;
        idf = idf.mapv(f64::ln) + 1.;
        ndarray_extension::vec2diagonal(idf)
    }

    fn _tfidi_transform(countvector: Array2<u32>, tf_method: &str, smooth_idf: u64) -> Array2<f64> {

        let tf = TfidfVectorizer::_get_term_frequency(countvector.clone(), tf_method);
        let idf = TfidfVectorizer::_get_idf_matrix(countvector, smooth_idf);
        let tfidf = tf.dot(&idf);
        tfidf
    }

    pub fn fit_transform(&mut self, docs: Vec<&'a str>, tf_method: &str, smooth_idf: u64) -> Array2<f64> {
        let countvector = self._create_countvector(docs);
        let tfidfvector = TfidfVectorizer::_tfidi_transform(countvector, tf_method, smooth_idf);
        tfidfvector
    }
}



#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_countvector(){
        let fruits_str = "apple, banana, apple, banana, orange, three, \
                        apple. apple, banana, orange, orange, ONE, three";
        let numbers_str = "one, two, three, two, three, apple, three. three, four, four, ONE";
        let mut docs1: Vec<&str> = Vec::new();
        docs1.push(fruits_str);
        docs1.push(numbers_str);

        let mut vectorizer = TfidfVectorizer::new();
        assert_eq!(0, vectorizer.vocabulary_.len());    // Before counting

        let countvector = vectorizer._create_countvector(docs1.clone());
        assert_eq!(8, vectorizer.vocabulary_.len());    // After counting

        let apple_col_index = vectorizer.vocabulary_["apple"];
        assert_eq!(0, apple_col_index);
        assert_eq!(4, countvector[[0, apple_col_index as usize]]);
        assert_eq!(1, countvector[[1, apple_col_index as usize]]);

        println!("=== Example1 ===");
        println!("Doc0 :{:?}", fruits_str);
        println!("Doc1 :{:?}", numbers_str);

        println!("CountVector :\n{:?}", countvector);
        println!("\n");
    }

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

        let tf1 = TfidfVectorizer::_get_term_frequency(x, "linear");
        let tf2 = TfidfVectorizer::_get_term_frequency(y, "linear");
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

        let df1 = TfidfVectorizer::_get_document_frequency(x);
        let df2 = TfidfVectorizer::_get_document_frequency(y);
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
    fn test_tfidf_transform(){
        let x = array![
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [2, 0, 0]
        ];
        let y = array![
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [10, 11, 12]
        ];

        let tfidf1 = TfidfVectorizer::_tfidi_transform(x, "linear", 0);
        let tfidf2 = TfidfVectorizer::_tfidi_transform(y, "linear", 0);
        println!("X tf-idf:\n{:?}", tfidf1);
        println!("Y tf-idf:\n{:?}", tfidf2);

    }
}




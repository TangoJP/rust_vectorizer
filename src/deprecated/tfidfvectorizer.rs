use std::collections::HashMap;
use ndarray::{Array1, Array2};
use ndarray_extension;
use countvectorizer::CountVectorizer;

/// Transforms a ollection of documents (i.e. Vec of string slices) using
/// Term Frequency - Inverted Document Frequency (Tf-Idf) transformation. 
/// The basic flow of the computation follows that from scikit-learn. See
/// also CountVectorizer.
/// 
pub struct TfidfVectorizer<'a> {
    pub vocabulary_: HashMap<&'a str, u64>,
    pub smooth_idf: bool,
    pub sublinear_tf: bool,
    pub norm: &'a str,
}

impl<'a> TfidfVectorizer<'a> {
 
    /// Create a new instance of TfidfVectorizer. Initialized with an empty
    /// vocabulary map (HashMap<&str, u64> type), smooth_idf=true, sublinear
    /// _tf=false, and norm="l2" Currently only implments those default 
    /// parameters. Other options to be implemented.
    /// 
    pub fn new() -> TfidfVectorizer<'a> {
        let map: HashMap<&'a str, u64> = HashMap::new();

        // Return a new instance
        TfidfVectorizer {
            vocabulary_: map,
            smooth_idf: true,
            sublinear_tf: false,
            norm: "l2",
        }
    }

    fn _create_countvector(&mut self, docs: Vec<&'a str>) -> Array2<u64> {
        // CountVectorization by CountVectorizer
        let mut count_vectorizer = CountVectorizer::new();
        let countvector = count_vectorizer.fit_transform(docs);
        self.vocabulary_ = count_vectorizer.vocabulary_;
        countvector
    }

    fn _get_term_frequency(&self, countvector: Array2<u64>) -> Array2<f64>{
        // Convert to f64 with or without sublinear adjustment

        let term_frequency = countvector.mapv(|element| element as f64);

        // if sublinear_tf, pre-process countvector
        if self.sublinear_tf {
            term_frequency.mapv(f64::ln) + 1.0  // addition of 1 per sklearn
        } else { 
            term_frequency
        }
    }

    fn _get_document_frequency(&self, countvector: Array2<u64>) -> Array1<f64>{
        // Count number of documents that contain each word
        // *implementation is probably inefficient. It's a first-pass

        let (num_rows, num_columns) = countvector.dim();
        let mut document_frequency = Array1::<f64>::zeros(num_columns);
        for index_row in 0..num_rows {
            for index_col in 0..num_columns {
                if countvector[[index_row, index_col]] != 0 {
                    document_frequency[index_col] += 1.;
                }
            }
        }
        document_frequency
    }

    fn _get_idf_matrix(&self, countvector: Array2<u64>) -> Array2<f64>{
        // create idf matrix to multiply tf matrix with

        // get countvector dimension and get document frequency vector
        let (num_rows, _) = countvector.dim();
        let mut df = self._get_document_frequency(countvector);

        // smoothe by smooth_idf (see sklearn)
        let smoother = (self.smooth_idf as u8) as f64;
        df = df + smoother;
        let n_samples = (num_rows as f64) + smoother;
    
        // Caclulate idf and convert to diagonal matrix
        let mut idf = n_samples / df;
        idf = idf.mapv(f64::ln) + 1.;
        ndarray_extension::vec2diagonal(idf)
    }

    fn _tfidi_transform(&self, countvector: Array2<u64>) -> Array2<f64> {
        // Convert CountVector to Tf-Idf Vector

        let tf = self._get_term_frequency(countvector.clone());
        let idf = self._get_idf_matrix(countvector);
        let tfidf = tf.dot(&idf);
        ndarray_extension::l2_normalize(tfidf)  // l2 normalize. To be refactored
    }

    /// Fit and tfidf transform the collection of documents. It returns
    /// a transformed array. The computed vocabulary HashMap is available
    /// via vocabulary_ field of the struct after fit_transform() method is
    /// called.
    pub fn fit_transform(&mut self, docs: Vec<&'a str>) -> Array2<f64> {
        // Public API for transformation
        let countvector = self._create_countvector(docs);
        let tfidfvector = self._tfidi_transform(countvector);
        tfidfvector
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore]
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
    #[ignore]
    fn test_internal_methods() {
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

        let vectorizer1 = TfidfVectorizer::new();
        let vectorizer2 = TfidfVectorizer::new();

        // test _get_term_frequency()
        let tf1 = vectorizer1._get_term_frequency(x.clone());
        let tf2 = vectorizer2._get_term_frequency(y.clone());

        let ans_tf1 = array![
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0]];
        let ans_tf2 = array![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0]];

        assert_eq!(ans_tf1, tf1);
        assert_eq!(ans_tf2, tf2);

        // test _get_document_frequency()
        let df1 = vectorizer1._get_document_frequency(x);
        let df2 = vectorizer2._get_document_frequency(y);

        let ans_df1 = array![2., 1., 1.];
        let ans_df2 = array![4., 4., 4.];

        assert_eq!(ans_df1, df1);
        assert_eq!(ans_df2, df2);

        // print results
        println!("=== Testing Term Frequency ===");
        println!("X TF:\n{:?}", tf1);
        println!("Y TF:\n{:?}", tf2);
        println!("\n");

        println!("=== Testing Document Frequency ===");
        println!("X DF:\n{:?}", df1);
        println!("Y DF:\n{:?}", df2);
        println!("\n");
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
            [1, 2, 3],           [4, 5, 6],
            [7, 8, 9],
            [10, 11, 12]
        ];

        let vectorizer1 = TfidfVectorizer::new();
        let vectorizer2 = TfidfVectorizer::new();

        let tfidf1 = vectorizer1._tfidi_transform(x);
        let tfidf2 = vectorizer2._tfidi_transform(y);
        println!("X tf-idf:\n{:?}", tfidf1);
        println!("Y tf-idf:\n{:?}", tfidf2);

    }
}

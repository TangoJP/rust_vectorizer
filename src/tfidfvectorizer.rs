use std::collections::HashMap;
use ndarray::{Array1, Array2};
use ndarray_extension;
use countvectorizer::CountVectorizer;

/// Transforms a collection of documents (i.e. Vec of string slices) using
/// Term Frequency - Inverted Document Frequency (Tf-Idf) transformation. 
/// The basic flow of the computation follows that from scikit-learn. See
/// also countvectorizer::CountVectorizer.
/// 
pub struct TfidfVectorizer {
    /// HashMap containing the vocabulary (token as String) as keys and their 
    /// IDs (u64) as values  
    pub vocabulary_: HashMap<String, u64>,

    /// A range of n-values for n-grams to be included. For example
    /// ngram_range: (1, 3) would include uni-, bi-, and tr-grams. See also
    /// tokenizer::Tokenizer fr details.
    pub ngram_range : (u32, u32),

    /// The case of the resulting tokens. Default is no conversion. Options 
    /// are "upper" and "lower". Other inputs will use default. See also
    /// tokenizer::Tokenizer fr details.
    pub case: String,

    /// If true, add 1 to document frequencies to smooth idf weights, preventing
    /// zero divisions. Default is true.
    pub smooth_idf: bool,

    /// If true, replace term frequency (tf) with 1 + ln(tf) (sublinear scaling). 
    /// Default is false.
    pub sublinear_tf: bool,

    /// Type of norm used for normalization. Options are "l1", "l2", "none".
    /// Default is "l2"
    pub norm: String,
}

impl TfidfVectorizer {
 
    /// Create a new instance of TfidfVectorizer. Initialized with an empty
    /// vocabulary map (HashMap<String, u64> type), smooth_idf=true, sublinear
    /// _tf=false, and norm="l2" Currently only implments those default 
    /// parameters. Other options to be implemented.
    /// 
    pub fn new(ngram_range : (u32, u32), case: &str) -> TfidfVectorizer {
        let map: HashMap<String, u64> = HashMap::new();

        // Return a new instance
        TfidfVectorizer {
            vocabulary_: map,
            ngram_range : ngram_range,
            case: case.to_string(),
            smooth_idf: true,
            sublinear_tf: false,
            norm: "l2".to_string(),
        }
    }

    fn _create_countvector(&mut self, docs: Vec<&str>) -> Array2<u64> {
        // CountVectorization by CountVectorizer
        let mut count_vectorizer = CountVectorizer::new(self.ngram_range, self.case.as_str());
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
        ndarray_extension::bincount(countvector)

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
    /// 
    /// # Examples
    /// The text below is an excerpt from the Foreword section of “The Rust 
    /// Programming Language.” Each paragraph is padded as a "doc."
    /// ```
    /// extern crate vectorizer;
    /// 
    /// use vectorizer::tfidfvectorizer::TfidfVectorizer;
    /// 
    /// let docs = vec![
    ///     "It wasn’t always so clear, but the Rust programming language is \
    ///      fundamentally about empowerment: no matter what kind of code you \
    ///      are writing now, Rust empowers you to reach farther, to program with \
    ///      confidence in a wider variety of domains than you did before.", 
    ///     "Take, for example, “systems-level” work that deals with low-level \
    ///      details of memory management, data representation, and concurrency. \
    ///      Traditionally, this realm of programming is seen as arcane, \
    ///      accessible only to a select few who have devoted the necessary years \
    ///      learning to avoid its infamous pitfalls. And even those who practice \
    ///      it do so with caution, lest their code be open to exploits, crashes, \
    ///      or corruption.",
    ///     "Rust breaks down these barriers by eliminating the old pitfalls and \
    ///      providing a friendly, polished set of tools to help you along the \
    ///      way. Programmers who need to “dip down” into lower-level control can \
    ///      do so with Rust, without taking on the customary risk of crashes or \
    ///      security holes, and without having to learn the fine points of a \
    ///      fickle toolchain. Better yet, the language is designed to guide you \
    ///      naturally towards reliable code that is efficient in terms of speed \
    ///      and memory usage.",
    ///     "Programmers who are already working with low-level code can use Rust \
    ///      to raise their ambitions. For example, introducing parallelism in \
    ///      Rust is a relatively low-risk operation: the compiler will catch the \
    ///      classical mistakes for you. And you can tackle more aggressive \
    ///      optimizations in your code with the confidence that you won’t \
    ///      accidentally introduce crashes or vulnerabilities.",
    ///     "But Rust isn’t limited to low-level systems programming. It’s \
    ///     expressive and ergonomic enough to make CLI apps, web servers, and \
    ///     many other kinds of code quite pleasant to write — you’ll find simple \
    ///     examples of both later in the book. Working with Rust allows you to \
    ///     build skills that transfer from one domain to another; you can learn \
    ///     Rust by writing a web app, then apply those same skills to target your \
    ///     Raspberry Pi.",
    ///     "This book fully embraces the potential of Rust to empower its users. \
    ///      It’s a friendly and approachable text intended to help you level up \
    ///      not just your knowledge of Rust, but also your reach and confidence \
    ///      as a programmer in general. So dive in, get ready to learn—and \
    ///      welcome to the Rust community!"
    /// ];
    /// 
    /// // Transformation
    /// let mut vectorizer = TfidfVectorizer::new((1, 1), "lower");
    /// let x = vectorizer.fit_transform(docs);
    /// 
    /// println!("=== Example (Foreword by Matsakis & Turon in Rust Book)===");
    /// 
    /// // Print Word_id: Word correspondence
    /// for (k, v) in vectorizer.vocabulary_.iter() {
    ///     println!("(Word_id: Word) : ({:?}:{:?})", v, k);
    /// }
    /// 
    /// // Print the Tf-Idf array
    /// println!("Tf-Idf Vector :\n{:?}", x);
    /// ```
    /// 
    pub fn fit_transform(&mut self, docs: Vec<&str>) -> Array2<f64> {
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

        let mut vectorizer = TfidfVectorizer::new((1, 2), "lower");
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

        let vectorizer1 = TfidfVectorizer::new((1, 2), "lower");
        let vectorizer2 = TfidfVectorizer::new((1, 2), "lower");

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
    #[ignore]
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

        let vectorizer1 = TfidfVectorizer::new((1, 2), "lower");
        let vectorizer2 = TfidfVectorizer::new((1, 2), "lower");

        let tfidf1 = vectorizer1._tfidi_transform(x);
        let tfidf2 = vectorizer2._tfidi_transform(y);
        println!("X tf-idf:\n{:?}", tfidf1);
        println!("Y tf-idf:\n{:?}", tfidf2);

    }
}

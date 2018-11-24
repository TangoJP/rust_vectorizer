use std::vec::Vec;
use std::collections::HashMap;
use ndarray::Array2;
use tokenizer::Tokenizer;


/// Struct that convers a collection of documents (i.e. Vec<&str>) into a
/// frequency vector.
/// 
pub struct CountVectorizer {
    /// HashMap containing the vocabulary (token as String) as keys and their 
    /// IDs (u64) as values    
    pub vocabulary_ : HashMap<String, u64>,
    
    /// A range of n-values for n-grams to be included. For example
    /// ngram_range: (1, 3) would include uni-, bi-, and tr-grams. See also
    /// tokenizer::Tokenizer fr details.
    pub ngram_range : (u32, u32),
    
    /// The case of the resulting tokens. Default is no conversion. Options 
    /// are "upper" and "lower". Other inputs will use default. See also
    /// tokenizer::Tokenizer fr details.
    pub case: String,
}

impl CountVectorizer {

    /// Create a new instance of CountVectorizer. Initialized with an empty
    /// vocabulary map (HashMap<String, u64> type). ngrams_range parameter
    /// to be added soon.
    /// 
    pub fn new(ngram_range: (u32, u32), case: &str) -> CountVectorizer {
        let map: HashMap<String, u64> = HashMap::new();

        // Return a new instance
        CountVectorizer {
            vocabulary_: map,
            ngram_range: ngram_range,
            case: case.to_string(),
        }
    }

    // Function to conver Vec<HashMap<u64, u64>> into Array2<u64>, where each
    // column corresponds to a key in the HashMap and the value of the HashMap
    // the count for that key String for a row in the resulting matrix, which 
    // represents a document.
    //
    fn _sort_vocabulary_count(&self, vec_of_map: Vec<HashMap<u64, u64>>) -> Array2<u64>{
        let num_rows = vec_of_map.len();
        let num_columns = self.vocabulary_.len();
        let mut sorted_vec = Array2::<u64>::zeros((num_rows, num_columns));

        for i in 0..num_rows {
            for key in vec_of_map[i].keys() {
                sorted_vec[[i, (*key as usize)]] = *vec_of_map[i].get(key).unwrap();
            }
        }
        sorted_vec
    }

    /// Transform the collection of documents into word frequency count
    /// matrix. 'fit' part refers to establishment of the vocabulary HashMap.
    /// 
    /// # Examples
    /// The text below is an excerpt from the Foreword section of “The Rust 
    /// Programming Language.” Each paragraph is padded as a "doc."
    /// ```
    /// extern crate vectorizer;
    /// 
    /// use vectorizer::countvectorizer::CountVectorizer;
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
    /// let mut vectorizer = CountVectorizer::new((1, 1), "lower");
    /// let x = vectorizer.fit_transform(docs);
    /// 
    /// println!("=== Example (Foreword by Matsakis & Turon in Rust Book)===");
    /// 
    /// // Print Word_id: Word correspondence
    /// for (k, v) in vectorizer.vocabulary_.iter() {
    ///     println!("(Word_id: Word) : ({:?}:{:?})", v, k);
    /// }
    /// 
    /// // Print the Count array
    /// println!("Count Vector :\n{:?}", x);
    /// ```
    pub fn fit_transform(&mut self, docs: Vec<&str>) -> Array2<u64> {
        // tokenize the document collection
        let tk = Tokenizer::new(self.ngram_range, self.case.as_str());
        let _tokenized_docs = tk.tokenize(docs);

        // Vec to store vocab. count HashMap. Variable to return.
        let mut vec_of_map: Vec<HashMap<u64, u64>> = Vec::new();

        // Collect vocabulary
        let mut vocab_indexer: u64 = 0;             // indexer for unique words

        for _doc in _tokenized_docs {
            // HashMap to store vocab. counts for a doc
            let mut _vocab_counts: HashMap<u64, u64> = HashMap::new();

            for _token in _doc {
                // if _token is a new word, add to vocabulary_ and vocabulary_counts_
                if !self.vocabulary_.contains_key(_token.as_str()) {
                    self.vocabulary_.insert(_token, vocab_indexer.clone());
                    _vocab_counts.insert(vocab_indexer.clone(), 1);
                    vocab_indexer = vocab_indexer + 1;
                } else {        // Otherwise add vocab counts
                    let vocab_ind = self.vocabulary_[_token.as_str()];
                    *(_vocab_counts).entry(vocab_ind).or_insert(0) += 1;
                };
            }
            vec_of_map.push(_vocab_counts);
        }
        let sorted_vec = self._sort_vocabulary_count(vec_of_map);
        sorted_vec   // Return the Vec of count HashMaps
    }

    /// Utility function to create a reverse vocabulary map, where the token
    /// ID is the key and the String the value
    /// 
    pub fn reverse_vocabulary_hashmap(&self) -> HashMap<u64, String> {
        // Utility method that returns a HashMap for vocab where k and v are swapped
        let mut vocabulary_inverted: HashMap<u64, String> = HashMap::new();
        for (k, v) in self.vocabulary_.iter() {
        vocabulary_inverted.insert(*v, k.to_string());
        }
        vocabulary_inverted
    }

}

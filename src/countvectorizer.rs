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

// #[cfg(test)]
// mod tests {
//     use super::*;

//     #[test]
//     fn test (){}

// }
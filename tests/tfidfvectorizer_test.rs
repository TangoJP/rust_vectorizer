extern crate vectorizer;

use vectorizer::tfidfvectorizer::TfidfVectorizer;



#[test]
#[ignore]
fn test_fit_transform(){
    let fruits_str = "apple, banana, apple, banana, orange, three, \
                        apple. apple, banana, orange, orange, one, three";
    let numbers_str = "one, two, three, two, three, apple, three. three, four, four, one";
    let mut docs1: Vec<&str> = Vec::new();
    docs1.push(fruits_str);
    docs1.push(numbers_str);

    let mut vectorizer = TfidfVectorizer::new((1, 2), "lower");
    let tfidf = vectorizer.fit_transform(docs1);
    
    assert_eq!(24, vectorizer.vocabulary_.len());
    assert_eq!((2, 24), tfidf.dim());

    println!("Vocabulary: {:?}",vectorizer.vocabulary_);
    println!("Tf-Idf Matrix:\n{:?}", tfidf);

}

#[test]
#[ignore]
fn tfidf_example() {
    // all vec strings were convereted to lowercase for this example
    let docs = vec![
        "It wasn’t always so clear, but the Rust programming language is \
         fundamentally about empowerment: no matter what kind of code you \
         are writing now, Rust empowers you to reach farther, to program with \
         confidence in a wider variety of domains than you did before.", 
        "Take, for example, “systems-level” work that deals with low-level \
         details of memory management, data representation, and concurrency. \
         Traditionally, this realm of programming is seen as arcane, \
         accessible only to a select few who have devoted the necessary years \
         learning to avoid its infamous pitfalls. And even those who practice \
         it do so with caution, lest their code be open to exploits, crashes, \
         or corruption.",
        "Rust breaks down these barriers by eliminating the old pitfalls and \
         providing a friendly, polished set of tools to help you along the \
         way. Programmers who need to “dip down” into lower-level control can \
         do so with Rust, without taking on the customary risk of crashes or \
         security holes, and without having to learn the fine points of a \
         fickle toolchain. Better yet, the language is designed to guide you \
         naturally towards reliable code that is efficient in terms of speed \
         and memory usage.",
        "Programmers who are already working with low-level code can use Rust \
         to raise their ambitions. For example, introducing parallelism in \
         Rust is a relatively low-risk operation: the compiler will catch the \
         classical mistakes for you. And you can tackle more aggressive \
         optimizations in your code with the confidence that you won’t \
         accidentally introduce crashes or vulnerabilities.",
        "But Rust isn’t limited to low-level systems programming. It’s \
        expressive and ergonomic enough to make CLI apps, web servers, and \
        many other kinds of code quite pleasant to write — you’ll find simple \
        examples of both later in the book. Working with Rust allows you to \
        build skills that transfer from one domain to another; you can learn \
        Rust by writing a web app, then apply those same skills to target your \
        Raspberry Pi.",
        "This book fully embraces the potential of Rust to empower its users. \
         It’s a friendly and approachable text intended to help you level up \
         not just your knowledge of Rust, but also your reach and confidence \
         as a programmer in general. So dive in, get ready to learn—and \
         welcome to the Rust community!"
    ];

    // Check vocabulary size
    let mut vectorizer = TfidfVectorizer::new((1, 1), "lower");
    let x = vectorizer.fit_transform(docs.clone());

    println!("=== Example3 (Foreword by Matsakis & Turon in Rust Book)===");

    // print Word_id: Word correspondence
    for (k, v) in vectorizer.vocabulary_.iter() {
        println!("(Word_id: Word) : ({:?}:{:?})", v, k);
    }

    println!("CountVector :\n{:?}", x);
    println!("\n");
}
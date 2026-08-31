fn main() {
    if let Err(error) = indexsail::cli::execute(std::env::args().skip(1), std::io::stdout()) {
        eprintln!("error: {error}");
        std::process::exit(2);
    }
}

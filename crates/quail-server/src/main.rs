fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    quail_server::run(std::env::args().skip(1))
}

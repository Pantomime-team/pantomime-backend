use axum::{routing::get, Router};

const ADDR: &str = "0.0.0.0:3000"; // TODO: environment variable

#[tokio::main]
async fn main() {
    // Setup up the router
    let app = Router::new().route("/", get(get_root));

    // TODO: logging library
    println!("Starting the server on {}...", ADDR);

    // Run the server with hyper
    axum::Server::bind(&ADDR.parse().expect("failed to parse ip address"))
        .serve(app.into_make_service())
        .await
        .expect("server panicked");

    println!("Shutting down...");
}

async fn get_root() -> &'static str {
    "Hello, root"
}

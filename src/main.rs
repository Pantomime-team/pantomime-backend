use axum::{
    extract::{DefaultBodyLimit, Multipart},
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};

const ADDR: &str = "0.0.0.0:80"; // TODO: environment variable

#[tokio::main]
async fn main() {
    // Setup up the router
    let app = Router::new()
        .route("/", get(get_root))
        .route("/predict", post(process_video))
        .route_layer(DefaultBodyLimit::max(200 * 1024 * 1024));

    // TODO: logging library
    println!("Starting the server on {}...", ADDR);

    // Run the server with hyper
    axum::Server::bind(&ADDR.parse().expect("failed to parse ip address"))
        .serve(app.into_make_service())
        .await
        .expect("server panicked");

    println!("Shutting down...");
}

async fn process_video(multipart: Multipart) -> Result<Json<String>, StatusCode> {
    match receive(multipart).await {
        Ok(v) => Ok(v),
        Err(_) => Err(StatusCode::INTERNAL_SERVER_ERROR),
    }
}

async fn receive(mut multipart: Multipart) -> std::io::Result<Json<String>> {
    use std::io::Write;

    println!("Receiving a file...");
    let mut file = Vec::new();
    while let Some(field) = multipart.next_field().await.unwrap() {
        // let name = field.name().unwrap().to_string();
        let data = field.bytes().await.unwrap();
        file.write_all(&data)?;
    }

    // TODO: process

    Ok(Json(
        "TODO: Rust model side is not implemented yet".to_string(),
    ))
}

async fn get_root() -> &'static str {
    "Hello, root"
}

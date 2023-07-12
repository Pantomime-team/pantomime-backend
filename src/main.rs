use std::sync::Arc;

use anyhow::{Context, Result};
use axum::{
    extract::{DefaultBodyLimit, Multipart, State},
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};
use serde::{de::DeserializeOwned, Deserialize};

const ADDR: &str = "0.0.0.0:80"; // TODO: environment variable

#[derive(Deserialize)]
struct Config {
    model_path: std::path::PathBuf,
    window_size: usize,
    frame_interval: usize,
    // mean: [f32; 3],
    // std: [f32; 3],
}

fn load_file_str(path: impl AsRef<std::path::Path>) -> Result<String> {
    let path = path.as_ref();
    std::fs::read_to_string(path).context(format!("when loading file {:?}", path))
}

fn load_file_bytes(path: impl AsRef<std::path::Path>) -> Result<String> {
    let path = path.as_ref();
    std::fs::read_to_string(path).context(format!("when loading file {:?}", path))
}

fn load_toml<T: DeserializeOwned>(path: impl AsRef<std::path::Path>) -> Result<T> {
    let path = path.as_ref();
    let s = load_file_str(path)?;
    let value = toml::from_str(&s).context(format!("when deserializing toml file {:?}", path))?;
    Ok(value)
}

struct Model {}

impl Model {
    pub fn load(path: impl AsRef<std::path::Path>) -> Result<Self> {
        load_file_str(path).context("when loading model weights")?;
        Ok(Self {})
    }

    pub fn predict(&self, data: &[u8]) -> String {
        "TODO: Rust model side is not implemented yet".to_string()
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // Load config
    let config: Config = load_toml("config.toml").context("when loading config")?;

    // Load model
    let model = Model::load(&config.model_path).context("when loading model")?;

    // Setup up the router
    let app = Router::new()
        .route("/", get(get_root))
        .route("/predict", post(predict))
        .route_layer(DefaultBodyLimit::max(200 * 1024 * 1024))
        .with_state(Arc::new(model));

    // TODO: logging library
    println!("Starting the server on {}...", ADDR);

    // Run the server with hyper
    axum::Server::bind(&ADDR.parse().expect("failed to parse ip address"))
        .serve(app.into_make_service())
        .await
        .expect("server panicked");

    println!("Shutting down...");

    Ok(())
}

async fn get_root() -> &'static str {
    "Hello, root"
}

/// Make a prediction with a model using data in the attached file.
async fn predict(
    State(model): State<Arc<Model>>,
    multipart: Multipart,
) -> Result<Json<String>, StatusCode> {
    let data = err_500(receive_file(multipart).await)?;
    let result = model.predict(&data);
    Ok(Json(result))
}

/// Convert error variant into a 500 status code: internal server error.
fn err_500<T, E>(res: Result<T, E>) -> Result<T, StatusCode> {
    res.map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)
}

/// Load the file as bytes from a multipart message.
async fn receive_file(mut multipart: Multipart) -> std::io::Result<Vec<u8>> {
    use std::io::Write;

    println!("Receiving a file...");
    let mut file = Vec::new();
    while let Some(field) = multipart.next_field().await.unwrap() {
        // let name = field.name().unwrap().to_string();
        let data = field.bytes().await.unwrap();
        file.write_all(&data)?;
    }

    Ok(file)
}

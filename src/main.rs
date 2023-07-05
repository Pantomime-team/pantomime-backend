mod constants;

use std::{collections::VecDeque, sync::Arc};

use anyhow::{anyhow, bail, Context, Result};
use axum::{
    extract::{DefaultBodyLimit, Multipart, State},
    http::StatusCode,
    response::Html,
    routing::{get, post},
    Json, Router,
};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use socketioxide::{adapter::LocalAdapter, Socket, SocketIoLayer};
use tch::{vision::image, Tensor};
use tokio::sync::Mutex;
use tower::ServiceBuilder;
use tower_http::cors::CorsLayer;
use tracing::{debug, error, info, level_filters::LevelFilter};
use tracing_subscriber::FmtSubscriber;

const ADDR: &str = "0.0.0.0:1111"; // TODO: environment variable

fn report_err<T, E: std::fmt::Debug>(res: Result<T, E>) -> Option<T> {
    match res {
        Ok(value) => Some(value),
        Err(err) => {
            error!("An error occured: {:?}", err);
            None
        }
    }
}

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

fn load_toml<T: DeserializeOwned>(path: impl AsRef<std::path::Path>) -> Result<T> {
    let path = path.as_ref();
    let s = load_file_str(path)?;
    let value = toml::from_str(&s).context(format!("when deserializing toml file {:?}", path))?;
    Ok(value)
}

struct Model {
    window_size: usize,
    module: tch::CModule,
}

impl Model {
    pub fn load(window_size: usize, path: impl AsRef<std::path::Path>) -> Result<Self> {
        let path = path.as_ref();
        let model = tch::CModule::load(path)
            .context(format!("when loading model weights from {:?}", path))?;
        Ok(Self {
            window_size,
            module: model,
        })
    }

    pub fn predict(&self, images: &[Tensor]) -> Result<String> {
        if images.len() != self.window_size {
            bail!(
                "input array expected to have length {}, found {}",
                self.window_size,
                images.len()
            );
        }

        let input = Tensor::stack(images, 1).unsqueeze(0).unsqueeze(0);
        let input = input.totype(tch::Kind::Float);
        let res = self
            .module
            .forward_ts(&[input])
            .context("during a forward pass")?;
        let argmax = res.argmax(-1, false);
        info!("model result: {:?}", argmax);
        // argmax.

        Ok("TODO: Rust model side is not implemented yet".to_string())
    }
}

struct AppState {
    images: Mutex<VecDeque<Tensor>>,
    model: Model,
}

impl AppState {
    pub fn new(model: Model) -> Self {
        Self {
            images: Mutex::new(VecDeque::new()),
            model,
        }
    }

    pub async fn predict(&self, image_data: &[u8]) -> Result<Option<String>> {
        // Load image
        use base64::Engine;
        let prefix = "data:image/jpeg;base64,";
        let image_data = image_data
            .strip_prefix(prefix.as_bytes())
            .ok_or_else(|| anyhow!("Expected prefix {:?}", prefix))?;
        let image_url = base64::engine::general_purpose::STANDARD
            .decode(image_data)
            .context("when decoding image data")?;
        let image = image::load_from_memory(&image_url).context("when loading image data")?;

        // Update the image list
        let mut images = self.images.lock().await;
        images.push_back(image);

        if images.len() > self.model.window_size {
            // Extract data
            let mut data = images.split_off(self.model.window_size);
            std::mem::swap(&mut data, &mut images);
            let data: Vec<Tensor> = data.into_iter().collect();

            // Apply the model
            let result = self
                .model
                .predict(&data)
                .context("when applying the model")?;
            return Ok(Some(result));
        }

        // Wait for more frames
        Ok(None)
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // Setup logging
    tracing::subscriber::set_global_default(
        FmtSubscriber::builder()
            .with_max_level(if cfg!(debug_assertions) {
                LevelFilter::DEBUG
            } else {
                LevelFilter::INFO
            })
            .finish(),
    )
    .context("when setting up logger")?;

    // Load config
    let config: Config = load_toml("config.toml").context("when loading config")?;

    // Load model
    let model =
        Model::load(config.window_size, &config.model_path).context("when loading model")?;

    let state = Arc::new(AppState::new(model));

    // Setup SocketIO
    let ns = socketioxide::Namespace::builder()
        .add("/", {
            let state = state.clone();
            move |socket| {
                let state = state.clone();
                async move {
                    report_err(handle_socket_io(state, socket).await);
                }
            }
        })
        .build();

    // Setup up the router
    let app = Router::new()
        .route("/", get(get_root))
        .layer(
            ServiceBuilder::new()
                .layer(CorsLayer::permissive()) // Enable CORS policy
                .layer(SocketIoLayer::new(ns)),
        )
        // .route("/predict", post(predict))
        // .route_layer(DefaultBodyLimit::max(200 * 1024 * 1024))
        .with_state(state);

    info!("Starting the server on {}...", ADDR);

    // Run the server with hyper
    axum::Server::bind(&ADDR.parse().expect("failed to parse ip address"))
        .serve(app.into_make_service())
        .await
        .expect("server panicked");

    info!("Shutting down...");

    Ok(())
}

async fn get_root() -> Html<&'static str> {
    // "Hello, root"
    Html(include_str!("index.html"))
}

async fn handle_socket_io(state: Arc<AppState>, socket: Arc<Socket<LocalAdapter>>) -> Result<()> {
    info!("Socket.IO connected: {:?} {:?}", socket.ns(), socket.sid);
    // let data: serde_json::Value = socket.handshake.data().context("during handshake")?;
    // socket.emit("auth", data).ok();

    socket.on("data_stream", move |socket, data: String, _, _| {
        let state = state.clone();
        async move {
            debug!("Received socketio event on {:?}", socket.sid);
            if let Some(Some(result)) = report_err(state.predict(data.as_bytes()).await) {
                #[derive(Serialize, Debug)]
                struct Gloss {
                    gloss: String,
                }
                let gloss = Gloss { gloss: result };
                debug!("Emitting to data_recognised: {:?}", gloss);
                report_err(socket.emit("data_recognised", gloss));
            }
        }
    });

    Ok(())
}

// /// Make a prediction with a model using data in the attached file.
// async fn predict(
//     State(state): State<Arc<AppState>>,
//     // State(model): State<Arc<Model>>,
//     multipart: Multipart,
// ) -> Result<Json<String>, StatusCode> {
//     let image_data = err_500(receive_file(multipart).await)?;

//     let res = state
//         .predict(&image_data)
//         .await
//         .context("when loading next image");

//     let res = err_500(res)?; // TODO: better error - some are invalid data
//     Ok(Json(res))
// }

/// Convert error variant into a 500 status code: internal server error.
fn err_500<T, E: std::fmt::Debug>(res: Result<T, E>) -> Result<T, StatusCode> {
    res.map_err(|err| {
        error!("Internal error: {:?}", err);
        StatusCode::INTERNAL_SERVER_ERROR
    })
}

/// Load the file as bytes from a multipart message.
async fn receive_file(mut multipart: Multipart) -> std::io::Result<Vec<u8>> {
    use std::io::Write;

    debug!("Receiving a file...");
    let mut file = Vec::new();
    while let Some(field) = multipart.next_field().await.unwrap() {
        // let name = field.name().unwrap().to_string();
        let data = field.bytes().await.unwrap();
        file.write_all(&data)?;
    }

    Ok(file)
}

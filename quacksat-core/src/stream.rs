//! `robot.stream` (Duck remote, quacksat patch 3): the head camera as a
//! stream of JPEGs for the agent's remote page. While on, a thread of its
//! own pulls frames from mediad's `media.frame` socket at `fps` — the same
//! request, upright-and-downsample path and JPEG encoder `robot.get_frame`
//! uses ([`crate::camera::grab_jpeg`]) — and hands each one to a [`Sink`];
//! the agent backend's sink sends it as one binary WebSocket frame.
//!
//! It ends on its own when the sink fails (the connection is gone) or after
//! [`MAX_FAILURES`] camera failures in a row, timeouts included (the next
//! `robot.stream {on: true}` then answers "stream: camera failed"). The 10 s
//! idle stop ([`IDLE`]) is the session loop's: only it knows when the agent
//! last said anything. Frame bytes are never logged.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use base64::Engine as _;
use serde_json::{Value, json};

use crate::camera;

pub const DEFAULT_FPS: u64 = 8;
pub const MIN_FPS: u64 = 1;
pub const MAX_FPS: u64 = 10;
pub const DEFAULT_WIDTH: u64 = 320;
pub const MIN_WIDTH: u64 = 160;
pub const MAX_WIDTH: u64 = 640;
/// A frame above this is dropped and counted, never sent.
pub const MAX_FRAME_BYTES: usize = 100 * 1024;
/// Camera failures in a row that end the stream.
pub const MAX_FAILURES: u32 = 3;
/// Nothing at all from the agent for this long ends the stream (the session loop checks).
pub const IDLE: Duration = Duration::from_secs(10);
/// The longest single sleep between frames, so a stop is noticed quickly.
const NAP: Duration = Duration::from_millis(20);

/// Where frames go: true when the frame was sent, false once the connection is gone.
pub type Sink = Arc<dyn Fn(&[u8]) -> bool + Send + Sync>;

#[derive(Clone)]
struct Latest {
    jpeg: Vec<u8>,
    width: usize,
    height: usize,
}

struct Running {
    stop: Arc<AtomicBool>,
    done: Arc<AtomicBool>,
    failed: Arc<AtomicBool>,
}

#[derive(Default)]
pub struct Stream {
    running: Option<Running>,
    latest: Arc<Mutex<Option<Latest>>>,
    dropped: Arc<AtomicU64>,
    sent: Arc<AtomicU64>,
    camera_failed: bool,
    fps: u64,
    width: u64,
}

impl Stream {
    /// (Re)starts the stream: any earlier one is stopped first.
    pub fn start(&mut self, socket: &str, fps: u64, width: u64, sink: Sink) {
        self.start_capped(socket, fps, width, sink, MAX_FRAME_BYTES);
    }

    /// [`Stream::start`] with another frame cap (tests).
    pub fn start_capped(&mut self, socket: &str, fps: u64, width: u64, sink: Sink, max_bytes: usize) {
        self.stop();
        let running = Running {
            stop: Arc::new(AtomicBool::new(false)),
            done: Arc::new(AtomicBool::new(false)),
            failed: Arc::new(AtomicBool::new(false)),
        };
        let job = Job {
            socket: socket.to_string(),
            fps: fps.max(1),
            width: width as usize,
            max_bytes,
            sink,
            stop: running.stop.clone(),
            failed: running.failed.clone(),
            latest: self.latest.clone(),
            dropped: self.dropped.clone(),
            sent: self.sent.clone(),
        };
        let done = running.done.clone();
        let spawned = std::thread::Builder::new()
            .name("camera-stream".to_string())
            .spawn(move || {
                job.pump();
                done.store(true, Ordering::SeqCst);
            });
        if spawned.is_err() {
            tracing::warn!("camera stream: no thread for it");
            return;
        }
        tracing::info!(fps, width, "camera stream on");
        self.running = Some(running);
        self.fps = fps;
        self.width = width;
    }

    pub fn stop(&mut self) {
        if let Some(running) = self.running.take() {
            running.stop.store(true, Ordering::SeqCst);
            if running.failed.load(Ordering::SeqCst) {
                self.camera_failed = true;
            }
            tracing::info!(
                sent = self.sent.load(Ordering::SeqCst),
                dropped = self.dropped.load(Ordering::SeqCst),
                "camera stream off"
            );
        }
        if let Ok(mut latest) = self.latest.lock() {
            *latest = None;
        }
    }

    /// Whether frames are flowing; a stream that ended by itself is reaped here.
    pub fn is_on(&mut self) -> bool {
        if self.running.as_ref().is_some_and(|r| r.done.load(Ordering::SeqCst)) {
            self.stop();
        }
        self.running.is_some()
    }

    /// True once after the stream ended on camera failures.
    pub fn take_camera_failure(&mut self) -> bool {
        self.is_on();
        std::mem::take(&mut self.camera_failed)
    }

    /// The newest frame while streaming, shaped like `robot.get_frame`'s answer.
    pub fn latest(&self) -> Option<Value> {
        let live = self.running.as_ref().is_some_and(|r| !r.done.load(Ordering::SeqCst));
        if !live {
            return None;
        }
        let latest = self.latest.lock().ok()?.clone()?;
        Some(json!({
            "jpeg_b64": base64::engine::general_purpose::STANDARD.encode(&latest.jpeg),
            "width": latest.width,
            "height": latest.height,
        }))
    }

    /// (fps, width) of the last start.
    pub fn params(&self) -> (u64, u64) {
        (self.fps, self.width)
    }

    pub fn dropped(&self) -> u64 {
        self.dropped.load(Ordering::SeqCst)
    }
}

struct Job {
    socket: String,
    fps: u64,
    width: usize,
    max_bytes: usize,
    sink: Sink,
    stop: Arc<AtomicBool>,
    failed: Arc<AtomicBool>,
    latest: Arc<Mutex<Option<Latest>>>,
    dropped: Arc<AtomicU64>,
    sent: Arc<AtomicU64>,
}

impl Job {
    fn pump(self) {
        let period = Duration::from_secs_f64(1.0 / self.fps as f64);
        let mut next = Instant::now();
        let mut failures = 0u32;
        while !self.stop.load(Ordering::SeqCst) {
            let now = Instant::now();
            if now < next {
                std::thread::sleep((next - now).min(NAP));
                continue;
            }
            // A slow frame never makes the next ones burst to catch up.
            let target = next + period;
            next = if target < now { now + period } else { target };
            match camera::grab_jpeg(&self.socket, self.width) {
                Ok((jpeg, width, height)) => {
                    failures = 0;
                    if jpeg.len() > self.max_bytes {
                        self.dropped.fetch_add(1, Ordering::SeqCst);
                        continue;
                    }
                    if self.stop.load(Ordering::SeqCst) {
                        break;
                    }
                    if let Ok(mut latest) = self.latest.lock() {
                        *latest = Some(Latest { jpeg: jpeg.clone(), width, height });
                    }
                    if !(self.sink)(&jpeg) {
                        tracing::info!("camera stream: the agent connection is gone");
                        break;
                    }
                    self.sent.fetch_add(1, Ordering::SeqCst);
                }
                Err(error) => {
                    failures += 1;
                    tracing::debug!(%error, failures, "camera stream: frame skipped");
                    if failures >= MAX_FAILURES {
                        tracing::warn!("camera stream: {MAX_FAILURES} camera failures in a row — stopped");
                        self.failed.store(true, Ordering::SeqCst);
                        break;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
pub(crate) mod testing {
    //! A fake mediad that serves one frame per connection, forever.
    use serde_json::json;
    use std::io::{BufRead, BufReader, Write};
    use std::os::unix::net::UnixListener;

    pub fn serving_mediad(width: u32, height: u32) -> (tempfile::TempDir, String) {
        let (header, pixels) = crate::camera::test_frame(width, height, 0);
        let header = serde_json::to_value(&header).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let socket = dir.path().join("media.sock");
        let listener = UnixListener::bind(&socket).unwrap();
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(stream) = stream else { break };
                let mut line = String::new();
                if BufReader::new(stream.try_clone().unwrap()).read_line(&mut line).is_err() {
                    continue;
                }
                let mut out =
                    serde_json::to_vec(&json!({"jsonrpc": "2.0", "id": 1, "result": header})).unwrap();
                out.push(b'\n');
                let mut writer = stream;
                let _ = writer.write_all(&out);
                let _ = writer.write_all(&pixels);
            }
        });
        (dir, socket.to_str().unwrap().to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::testing::serving_mediad;
    use super::*;

    fn counting_sink() -> (Sink, Arc<Mutex<Vec<Vec<u8>>>>) {
        let got = Arc::new(Mutex::new(Vec::new()));
        let seen = got.clone();
        let sink: Sink = Arc::new(move |jpeg: &[u8]| {
            seen.lock().unwrap().push(jpeg.to_vec());
            true
        });
        (sink, got)
    }

    fn count(got: &Arc<Mutex<Vec<Vec<u8>>>>) -> usize {
        got.lock().unwrap().len()
    }

    #[test]
    fn starts_sends_jpegs_and_stops() {
        let (_dir, socket) = serving_mediad(64, 36);
        let (sink, got) = counting_sink();
        let mut stream = Stream::default();
        stream.start(&socket, 8, 320, sink);
        std::thread::sleep(Duration::from_millis(700));
        assert!(stream.is_on());
        let n = count(&got);
        assert!(n >= 3, "{n}");
        assert!(got.lock().unwrap().iter().all(|j| j.starts_with(&[0xFF, 0xD8, 0xFF]) && j.ends_with(&[0xFF, 0xD9])));
        stream.stop();
        let after = count(&got);
        std::thread::sleep(Duration::from_millis(400));
        assert!(count(&got) <= after + 1, "frames kept coming after stop");
        assert!(!stream.is_on());
        assert!(stream.latest().is_none());
    }

    #[test]
    fn the_rate_is_within_a_frame_a_second() {
        let (_dir, socket) = serving_mediad(64, 36);
        let (sink, got) = counting_sink();
        let mut stream = Stream::default();
        stream.start(&socket, 8, 320, sink);
        std::thread::sleep(Duration::from_secs(3));
        stream.stop();
        let n = count(&got);
        assert!((21..=27).contains(&n), "8 fps for 3 s gave {n} frames");
    }

    #[test]
    fn an_oversized_frame_is_dropped_and_counted() {
        let (_dir, socket) = serving_mediad(64, 36);
        let (sink, got) = counting_sink();
        let mut stream = Stream::default();
        stream.start_capped(&socket, 8, 320, sink, 100);
        std::thread::sleep(Duration::from_millis(600));
        assert_eq!(count(&got), 0);
        assert!(stream.dropped() >= 2);
        assert!(stream.is_on(), "a big frame is skipped, not a failure");
        stream.stop();
    }

    #[test]
    fn three_camera_failures_stop_the_stream_and_are_reported_once() {
        let dir = tempfile::tempdir().unwrap();
        let missing = dir.path().join("nobody.sock");
        let (sink, got) = counting_sink();
        let mut stream = Stream::default();
        stream.start(missing.to_str().unwrap(), 10, 320, sink);
        std::thread::sleep(Duration::from_millis(800));
        assert!(!stream.is_on());
        assert_eq!(count(&got), 0);
        assert!(stream.take_camera_failure());
        assert!(!stream.take_camera_failure(), "reported once");
    }

    #[test]
    fn a_sink_that_fails_ends_the_stream() {
        let (_dir, socket) = serving_mediad(64, 36);
        let mut stream = Stream::default();
        stream.start(&socket, 8, 320, Arc::new(|_: &[u8]| false));
        std::thread::sleep(Duration::from_millis(500));
        assert!(!stream.is_on());
        assert!(!stream.take_camera_failure(), "a closed connection is not a camera failure");
    }

    #[test]
    fn the_latest_frame_is_shaped_like_get_frame() {
        let (_dir, socket) = serving_mediad(64, 36);
        let (sink, _got) = counting_sink();
        let mut stream = Stream::default();
        stream.start(&socket, 8, 320, sink);
        std::thread::sleep(Duration::from_millis(400));
        let frame = stream.latest().expect("a frame");
        assert_eq!(frame["width"], 64);
        assert_eq!(frame["height"], 36);
        assert!(frame["jpeg_b64"].as_str().unwrap().starts_with("/9j/"));
        assert_eq!(stream.params(), (8, 320));
        stream.stop();
    }
}

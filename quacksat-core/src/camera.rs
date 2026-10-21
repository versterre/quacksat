//! `robot.get_frame`: one picture from the head camera, on demand.
//!
//! The pixels come from mediad's local frame socket (`media.frame`,
//! `duck_ipc_proto::socket::MEDIA`): one NDJSON JSON-RPC 2.0 request, one
//! result line carrying a [`proto::MediaFrameHeader`], then exactly
//! `bytes` raw UYVY bytes. The frame is turned upright and sampled down
//! with Pollen's own `uyvy` crate (the same arithmetic mediad's snapshot
//! route uses, so the picture cannot come out sideways here and upright
//! there), then encoded as a JPEG and handed to the agent as base64.
//!
//! One connection per call, nothing kept open: a frame is asked for
//! seldom, and mediad serves exactly one frame per connection.
//!
//! Errors are soft and carry no path and no payload: the text goes to an
//! LLM. Frame bytes are never logged.

use base64::Engine as _;
use duck_ipc_proto as proto;
use serde_json::{Value, json};
use std::io::{BufRead, BufReader, Read, Write};
use std::os::unix::net::UnixStream;
use std::time::{Duration, Instant};

/// Everything — connect, request, header, pixels — inside this.
pub const DEADLINE: Duration = Duration::from_secs(2);
/// The longest side of the JPEG the agent gets.
pub const LONGEST: usize = 640;
/// JPEG quality: about 35 to 55 KB at 360x640.
pub const QUALITY: u8 = 70;
/// A header line is a few hundred bytes; anything this long is not one.
const MAX_HEADER_BYTES: u64 = 64 * 1024;

/// Why a frame did not come. `Display` is the exact text of the tool's
/// `error`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CameraError {
    /// No `[camera] frame_socket` configured.
    Unsupported,
    /// The deadline passed before the whole frame arrived.
    Timeout,
    /// Anything else, by kind only.
    Other(String),
}

impl std::fmt::Display for CameraError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unsupported => f.write_str("unsupported"),
            Self::Timeout => f.write_str("camera timeout"),
            Self::Other(kind) => write!(f, "camera error: {kind}"),
        }
    }
}

impl From<std::io::Error> for CameraError {
    fn from(error: std::io::Error) -> Self {
        use std::io::ErrorKind;
        match error.kind() {
            ErrorKind::TimedOut | ErrorKind::WouldBlock => Self::Timeout,
            // The kind's own text ("entity not found", "connection
            // refused"): never the OS message, which can carry the path.
            kind => Self::Other(kind.to_string()),
        }
    }
}

fn other(kind: &str) -> CameraError {
    CameraError::Other(kind.to_string())
}

/// The tool: one frame from `socket`, as `{jpeg_b64, width, height}`.
/// An empty `socket` is a robot without a camera lane: "unsupported".
pub fn grab(socket: &str) -> Result<Value, CameraError> {
    if socket.trim().is_empty() {
        return Err(CameraError::Unsupported);
    }
    let started = Instant::now();
    let (header, pixels) = fetch(socket, started + DEADLINE)?;
    let fetched = started.elapsed();
    let (jpeg, width, height) = encode(&header, &pixels)?;
    tracing::debug!(
        fetch_ms = fetched.as_millis() as u64,
        total_ms = started.elapsed().as_millis() as u64,
        jpeg_bytes = jpeg.len(),
        width,
        height,
        "camera frame"
    );
    Ok(json!({
        "jpeg_b64": base64::engine::general_purpose::STANDARD.encode(&jpeg),
        "width": width,
        "height": height,
    }))
}

/// The time left before `deadline`, or a timeout when there is none.
fn remaining(deadline: Instant) -> Result<Duration, CameraError> {
    deadline
        .checked_duration_since(Instant::now())
        .filter(|left| !left.is_zero())
        .ok_or(CameraError::Timeout)
}

/// Shrink the read timeout to what is left of the deadline.
///
/// macOS refuses `SO_RCVTIMEO` with EINVAL once the peer has closed its
/// end — which mediad does right after the last pixel, while some of
/// them still sit in our buffer. Those reads cannot block, so that
/// refusal is not an error.
fn read_timeout(stream: &UnixStream, deadline: Instant) -> Result<(), CameraError> {
    match stream.set_read_timeout(Some(remaining(deadline)?)) {
        Err(error) if error.kind() != std::io::ErrorKind::InvalidInput => Err(error.into()),
        _ => Ok(()),
    }
}

/// Ask mediad for one frame: the header and exactly its `bytes` of UYVY.
fn fetch(socket: &str, deadline: Instant) -> Result<(proto::MediaFrameHeader, Vec<u8>), CameraError> {
    // A unix connect does not wait on a listener's backlog the way TCP
    // does: it is there, refused, or missing at once.
    let stream = UnixStream::connect(socket)?;
    stream.set_write_timeout(Some(remaining(deadline)?))?;
    let request = proto::Request {
        jsonrpc: proto::JSONRPC_VERSION.to_owned(),
        id: Some(proto::Id::Number(1)),
        method: proto::method::MEDIA_FRAME.to_string(),
        params: None,
    };
    let mut line = serde_json::to_vec(&request).map_err(|_| other("bad request"))?;
    line.push(b'\n');
    (&stream).write_all(&line)?;
    (&stream).flush()?;

    let mut reader = BufReader::with_capacity(64 * 1024, &stream);
    read_timeout(&stream, deadline)?;
    let mut header_line = Vec::new();
    (&mut reader)
        .take(MAX_HEADER_BYTES)
        .read_until(b'\n', &mut header_line)?;
    if header_line.last() != Some(&b'\n') {
        return Err(if header_line.is_empty() { other("closed") } else { other("bad header") });
    }
    let response: proto::Response =
        serde_json::from_slice(&header_line).map_err(|_| other("bad header"))?;
    if response.error.is_some() {
        // mediad's text ("no frame arrived within the capture timeout")
        // stays in mediad's log.
        return Err(other("no frame"));
    }
    let header = response
        .result_as::<proto::MediaFrameHeader>()
        .map_err(|_| other("bad header"))?;
    if !header.valid_uyvy() {
        return Err(other("bad frame"));
    }

    // Pixels in chunks, the read timeout shrinking with the deadline, so
    // a mediad that trickles cannot hold the session loop past it.
    let mut pixels = vec![0u8; header.bytes];
    let mut filled = 0;
    while filled < pixels.len() {
        read_timeout(&stream, deadline)?;
        match reader.read(&mut pixels[filled..]) {
            Ok(0) => return Err(other("short frame")),
            Ok(n) => filled += n,
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => {}
            Err(error) => return Err(error.into()),
        }
    }
    Ok((header, pixels))
}

/// UYVY → upright RGB, longest side [`LONGEST`] → JPEG at [`QUALITY`].
/// Returns the JPEG and its size.
pub fn encode(
    header: &proto::MediaFrameHeader,
    pixels: &[u8],
) -> Result<(Vec<u8>, usize, usize), CameraError> {
    if !header.valid_uyvy() || pixels.len() != header.bytes {
        return Err(other("bad frame"));
    }
    let turn = uyvy::Turn::from_degrees(header.rotate).ok_or_else(|| other("bad frame"))?;
    let mut rgb = Vec::new();
    let (width, height) = uyvy::rgb_from_uyvy(
        pixels,
        header.width as usize,
        header.height as usize,
        LONGEST,
        turn,
        &mut rgb,
    );
    let (w, h) = (
        u16::try_from(width).map_err(|_| other("bad frame"))?,
        u16::try_from(height).map_err(|_| other("bad frame"))?,
    );
    let mut jpeg = Vec::with_capacity(64 * 1024);
    jpeg_encoder::Encoder::new(&mut jpeg, QUALITY)
        .encode(&rgb, w, h, jpeg_encoder::ColorType::Rgb)
        .map_err(|_| other("encode"))?;
    Ok((jpeg, width, height))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::net::UnixListener;

    /// A synthetic UYVY frame: a horizontal luma ramp, neutral chroma.
    fn frame(width: u32, height: u32, rotate: u32) -> (proto::MediaFrameHeader, Vec<u8>) {
        let mut data = Vec::with_capacity((width * height * 2) as usize);
        for _y in 0..height {
            for x in (0..width).step_by(2) {
                let luma = (16 + (x * 219) / width) as u8;
                data.extend_from_slice(&[128, luma, 128, luma]);
            }
        }
        let header = proto::MediaFrameHeader {
            width,
            height,
            format: "UYVY".to_string(),
            bytes: data.len(),
            captured_at_unix_us: 1,
            rotate,
        };
        (header, data)
    }

    /// The size a baseline JPEG declares in its SOF0 segment.
    fn sof_size(jpeg: &[u8]) -> (usize, usize) {
        let at = jpeg.windows(2).position(|w| w == [0xFF, 0xC0]).expect("SOF0");
        let height = u16::from_be_bytes([jpeg[at + 5], jpeg[at + 6]]) as usize;
        let width = u16::from_be_bytes([jpeg[at + 7], jpeg[at + 8]]) as usize;
        (width, height)
    }

    #[test]
    fn a_uyvy_frame_becomes_a_jpeg_of_the_right_size() {
        let (header, data) = frame(64, 36, 0);
        let (jpeg, width, height) = encode(&header, &data).unwrap();
        assert_eq!(&jpeg[..3], &[0xFF, 0xD8, 0xFF], "JPEG magic");
        assert_eq!(&jpeg[jpeg.len() - 2..], &[0xFF, 0xD9], "JPEG end");
        assert_eq!((width, height), (64, 36));
        assert_eq!(sof_size(&jpeg), (64, 36));
    }

    #[test]
    fn a_quarter_turn_mount_comes_out_upright() {
        let (header, data) = frame(64, 36, 90);
        let (jpeg, width, height) = encode(&header, &data).unwrap();
        assert_eq!((width, height), (36, 64));
        assert_eq!(sof_size(&jpeg), (36, 64));
    }

    #[test]
    fn a_big_frame_is_sampled_down_to_the_longest_side() {
        // The robot's sensor: 1280x720, mounted a quarter turn off.
        let (header, data) = frame(1280, 720, 90);
        let (_, width, height) = encode(&header, &data).unwrap();
        assert_eq!((width, height), (360, 640));
    }

    #[test]
    fn a_frame_that_lies_about_its_size_is_refused() {
        let (header, mut data) = frame(64, 36, 0);
        data.pop();
        assert_eq!(encode(&header, &data).unwrap_err(), other("bad frame"));
    }

    /// One fake mediad: reads the request, answers with `header` (a
    /// result line) and then `tail`.
    fn fake_mediad(header: Value, tail: Vec<u8>) -> (tempfile::TempDir, String, std::thread::JoinHandle<()>) {
        let dir = tempfile::tempdir().unwrap();
        let socket = dir.path().join("media.sock");
        let listener = UnixListener::bind(&socket).unwrap();
        let server = std::thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            let mut reader = BufReader::new(stream.try_clone().unwrap());
            let mut line = String::new();
            reader.read_line(&mut line).unwrap();
            let request: proto::Request = serde_json::from_str(&line).unwrap();
            assert_eq!(request.method, "media.frame");
            let mut out = serde_json::to_vec(&json!({"jsonrpc": "2.0", "id": request.id, "result": header})).unwrap();
            out.push(b'\n');
            let mut writer = stream;
            writer.write_all(&out).unwrap();
            let _ = writer.write_all(&tail);
            let _ = writer.flush();
        });
        (dir, socket.to_str().unwrap().to_string(), server)
    }

    #[test]
    fn a_frame_from_mediad_round_trips_through_base64() {
        let (header, data) = frame(64, 36, 0);
        let (_dir, socket, server) = fake_mediad(serde_json::to_value(&header).unwrap(), data.clone());
        let answer = grab(&socket).unwrap();
        server.join().unwrap();
        assert_eq!(answer["width"], 64);
        assert_eq!(answer["height"], 36);
        let jpeg = base64::engine::general_purpose::STANDARD
            .decode(answer["jpeg_b64"].as_str().unwrap())
            .unwrap();
        assert_eq!(jpeg, encode(&header, &data).unwrap().0);
    }

    #[test]
    fn no_socket_configured_is_unsupported() {
        assert_eq!(grab("").unwrap_err().to_string(), "unsupported");
        assert_eq!(grab("  ").unwrap_err().to_string(), "unsupported");
    }

    #[test]
    fn an_unreachable_socket_is_a_camera_error_without_the_path() {
        let dir = tempfile::tempdir().unwrap();
        let missing = dir.path().join("nobody-here.sock");
        let error = grab(missing.to_str().unwrap()).unwrap_err().to_string();
        assert!(error.starts_with("camera error: "), "{error}");
        assert!(!error.contains("nobody-here"), "{error}");

        // A socket file nobody listens on any more (mediad died).
        let stale = dir.path().join("stale.sock");
        drop(UnixListener::bind(&stale).unwrap());
        let error = grab(stale.to_str().unwrap()).unwrap_err().to_string();
        assert!(error.starts_with("camera error: "), "{error}");
        assert!(!error.contains("stale"), "{error}");
    }

    #[test]
    fn a_mediad_that_never_answers_is_a_timeout() {
        let dir = tempfile::tempdir().unwrap();
        let socket = dir.path().join("media.sock");
        let listener = UnixListener::bind(&socket).unwrap();
        let server = std::thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            std::thread::sleep(DEADLINE + Duration::from_millis(500));
            drop(stream);
        });
        let started = Instant::now();
        let error = grab(socket.to_str().unwrap()).unwrap_err();
        assert_eq!(error, CameraError::Timeout);
        assert_eq!(error.to_string(), "camera timeout");
        assert!(started.elapsed() < DEADLINE + Duration::from_millis(300));
        server.join().unwrap();
    }

    #[test]
    fn a_short_frame_and_a_refusal_are_camera_errors() {
        let (header, mut data) = frame(64, 36, 0);
        data.truncate(100);
        let (_dir, socket, server) = fake_mediad(serde_json::to_value(&header).unwrap(), data);
        assert_eq!(grab(&socket).unwrap_err().to_string(), "camera error: short frame");
        server.join().unwrap();

        let (_dir, socket, server) = fake_mediad(json!({"width": 3}), Vec::new());
        assert_eq!(grab(&socket).unwrap_err().to_string(), "camera error: bad header");
        server.join().unwrap();
    }
}

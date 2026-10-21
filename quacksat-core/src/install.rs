//! `robot.install_policy`, `robot.policy_status`, `robot.remove_policy`:
//! put a community policy on the duck as a *user* skill and say when it
//! can be asked for.
//!
//! The file comes from the Hugging Face Hub (`org/repo`, fetched from
//! `https://huggingface.co/<org>/<repo>/resolve/main/<file>`) or from an
//! `https://` URL on an allowlisted host (a signed URL the agent's
//! gateway made for a private policy). It lands in
//! `[install] dir/<name>/<file>` — written to `<file>.part` and renamed,
//! so robotd never sees half a file — and is then registered with
//! robotd's `robot.setSkill`. robotd writes `[[policy.skill]]` and
//! reloads itself, which sends the robot to its home pose; that is what
//! `robot.policy_status` follows, through `robot.policies`.
//!
//! Only names that are not one of the robot's own skills: a user slot
//! cannot replace the gait, the stand or a stock trick. Errors are one
//! fixed line each — never a path, never what robotd said — because the
//! text goes to an LLM and from there out loud. Logs carry names, sizes
//! and durations, never file contents.

use serde_json::{Value, json};
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

/// Where policies land when `[install] dir` is not set: the duck's own
/// writable data directory (the unit's `StateDirectory=quacksat`).
pub const DEFAULT_DIR: &str = "/var/lib/quacksat/policies";
/// What `file` is when the call does not say: the Hub's convention.
pub const DEFAULT_FILE: &str = "policy.onnx";
/// The only slot a call may ask for.
pub const USER_SLOT: &str = "user";
/// How long a skill runs when the call does not say. robotd requires a
/// duration for a skill it does not already have; this is the value the
/// Policy Playground sends when a manifest has no `duration_s`
/// (`pollen-robotics/microduck-policy-playground`, "held for 3s").
pub const DEFAULT_DURATION_S: f64 = 3.0;
/// The longest a call may ask a skill to run.
pub const MAX_DURATION_S: f64 = 60.0;
/// The largest policy accepted. The official set's are under 1 MB.
pub const MAX_BYTES: u64 = 20 * 1024 * 1024;
/// The whole download — connect, redirects, body — inside this.
pub const DOWNLOAD_TIMEOUT: Duration = Duration::from_secs(60);
/// The robot's own skills and slots: never a user slot's name.
pub const STOCK: &[&str] = &[
    "walk",
    "stand",
    "sitstand",
    "sit_toggle",
    "ground_pick",
    "kick_left",
    "kick_right",
    "roulade",
];
/// Hosts an `https://` source may name. Anything else goes through the
/// Hub form or is refused.
const HOSTS: &[&str] = &["huggingface.co", "storage.googleapis.com"];
/// How long a reinstall (a name robotd already had) reads as "homing"
/// when no homing has been seen yet: robotd answers `homed` for the old
/// entry until the reload it was asked for starts, some tens of ms later.
const REINSTALL_GRACE: Duration = Duration::from_secs(5);
/// How long a pending name may stay unlisted on a robot that homed
/// since the install before it reads as `error`: a reload takes about
/// two seconds on the sim, and robotd drops a change that arrives while
/// another reload is in flight (measured 2026-10-21), so a robot back
/// home without the name is not going to list it.
const LOST_AFTER: Duration = Duration::from_secs(10);
/// How long `robot.install_policy` and `robot.remove_policy` wait for a
/// robot that is homing (another reload in flight) before calling robotd
/// anyway. See [`LOST_AFTER`] for why they wait.
pub const SETTLE_TIMEOUT: Duration = Duration::from_secs(10);

pub const DOWNLOAD_FAILED: &str = "install: download failed";
pub const NOT_ONNX: &str = "install: not an onnx file";
pub const TOO_LARGE: &str = "install: too large";
pub const SLOT_REFUSED: &str = "install: slot refused";
pub const SET_SKILL_REFUSED: &str = "install: setSkill refused";
pub const REMOVE_REFUSED: &str = "install: remove refused";
pub const ROBOT_UNAVAILABLE: &str = "install: robot unavailable";

/// One `robot.install_policy` call, checked.
#[derive(Debug, Clone, PartialEq)]
pub struct Request {
    pub name: String,
    pub file: String,
    /// Where the bytes come from, resolved from `source`.
    pub url: String,
    pub duration_s: f64,
    pub chain: Option<bool>,
}

impl Request {
    /// Check a call's arguments. Everything but the source is refused as
    /// `install: slot refused`; a source that is neither a Hub repo nor
    /// an allowlisted `https://` URL is `install: download failed`.
    pub fn parse(args: &Value) -> Result<Self, &'static str> {
        let name = args.get("name").and_then(Value::as_str).ok_or(SLOT_REFUSED)?;
        validate_name(name)?;
        match args.get("slot") {
            None | Some(Value::Null) => {}
            Some(slot) if slot.as_str() == Some(USER_SLOT) => {}
            Some(_) => return Err(SLOT_REFUSED),
        }
        let file = match args.get("file") {
            None | Some(Value::Null) => DEFAULT_FILE,
            Some(file) => file.as_str().ok_or(SLOT_REFUSED)?,
        };
        validate_file(file)?;
        let duration_s = match args.get("duration_s") {
            None | Some(Value::Null) => DEFAULT_DURATION_S,
            Some(value) => match value.as_f64() {
                Some(d) if d > 0.0 && d <= MAX_DURATION_S => d,
                _ => return Err(SLOT_REFUSED),
            },
        };
        // robotd's `chain` is a boolean; a string spelling of one is
        // accepted because the gateway's contract carries it as a string.
        let chain = match args.get("chain") {
            None | Some(Value::Null) => None,
            Some(Value::Bool(chain)) => Some(*chain),
            Some(Value::String(chain)) if chain == "true" => Some(true),
            Some(Value::String(chain)) if chain == "false" || chain.is_empty() => Some(false),
            Some(_) => return Err(SLOT_REFUSED),
        };
        let source = args.get("source").and_then(Value::as_str).ok_or(DOWNLOAD_FAILED)?;
        let url = download_url(source, file)?;
        Ok(Self { name: name.to_string(), file: file.to_string(), url, duration_s, chain })
    }
}

/// The URL a `source` is fetched from: `org/repo` on the Hub, or an
/// `https://` URL on an allowlisted host, taken as given.
pub fn download_url(source: &str, file: &str) -> Result<String, &'static str> {
    if source.chars().any(|c| c.is_whitespace() || c.is_control()) {
        return Err(DOWNLOAD_FAILED);
    }
    if let Some(rest) = source.strip_prefix("https://") {
        // Everything up to the first `/`, `?` or `#` is the authority; an
        // authority with userinfo or a port is not an allowlisted host.
        let host = rest.split(['/', '?', '#']).next().unwrap_or("");
        if HOSTS.contains(&host) {
            return Ok(source.to_string());
        }
        return Err(DOWNLOAD_FAILED);
    }
    let part = |s: &str| {
        !s.is_empty()
            && s.len() <= 96
            && !s.starts_with('.')
            && !s.contains("..")
            && s.chars().all(|c| c.is_ascii_alphanumeric() || "_.-".contains(c))
    };
    match source.split('/').collect::<Vec<_>>().as_slice() {
        [org, repo] if part(org) && part(repo) => {
            Ok(format!("https://huggingface.co/{org}/{repo}/resolve/main/{file}"))
        }
        _ => Err(DOWNLOAD_FAILED),
    }
}

/// A file name, not a path: `[A-Za-z0-9_.-]+\.onnx`, no `..`.
pub fn validate_file(file: &str) -> Result<(), &'static str> {
    let ok = file.len() > ".onnx".len()
        && file.len() <= 128
        && file.ends_with(".onnx")
        && !file.starts_with('.')
        && !file.contains("..")
        && file.chars().all(|c| c.is_ascii_alphanumeric() || "_.-".contains(c));
    if ok { Ok(()) } else { Err(SLOT_REFUSED) }
}

/// A user slot's name: `[a-z][a-z0-9_]{1,31}`, and none of [`STOCK`].
pub fn validate_name(name: &str) -> Result<(), &'static str> {
    let first_ok = name.chars().next().is_some_and(|c| c.is_ascii_lowercase());
    let rest_ok = (2..=32).contains(&name.len())
        && name.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_');
    if first_ok && rest_ok && !STOCK.contains(&name) { Ok(()) } else { Err(SLOT_REFUSED) }
}

/// An ONNX file is a protobuf `ModelProto` whose first field is
/// `ir_version` (field 1, varint): its first byte is the tag `0x08`. An
/// HTML error page, a JSON answer or an LFS pointer is not.
pub fn looks_like_onnx(bytes: &[u8]) -> bool {
    bytes.first() == Some(&0x08)
}

/// The install `robot.policy_status` is following, if any: set by a
/// successful `robot.install_policy`, cleared once that name is homed.
#[derive(Debug, Clone)]
pub struct Pending {
    pub name: String,
    /// `robot.policies`' `change_error` just before `robot.setSkill`: a
    /// failure left over from an earlier change is not this install's.
    pub change_error: Option<String>,
    /// Whether robotd already had a skill by this name (a reinstall).
    pub was_present: bool,
    pub since: Instant,
    /// A `homed: false` has been seen since the install.
    pub seen_homing: bool,
}

impl Pending {
    /// Taken from `robot.policies`' answer just before `robot.setSkill`.
    pub fn before(name: &str, policies: &Value) -> Self {
        Self {
            name: name.to_string(),
            change_error: change_error(policies),
            was_present: has_skill(policies, name),
            since: Instant::now(),
            seen_homing: false,
        }
    }
}

fn has_skill(policies: &Value, name: &str) -> bool {
    policies
        .get("skills")
        .and_then(Value::as_array)
        .is_some_and(|skills| skills.iter().any(|s| s.as_str() == Some(name)))
}

fn change_error(policies: &Value) -> Option<String> {
    policies.get("change_error").and_then(Value::as_str).map(str::to_string)
}

/// The state of `name` in a `robot.policies` answer, with nothing pending.
///
/// robotd lists the configured skills by name (`skills`) and says
/// whether the robot is at its home pose (`homed`, `None` from a robot
/// too old to say). A listed skill is `homed` once the robot is,
/// `homing` until then; one that is not listed is `missing`, or `error`
/// when robotd reports a failed change. Anything unreadable is `error`.
pub fn state_of(policies: &Value, name: &str) -> &'static str {
    status(policies, name, None).0
}

/// [`state_of`], following an install: right after `robot.setSkill` the
/// new name is not listed until robotd's reload finishes (about two
/// seconds on the sim, the robot homing meanwhile), so a pending name
/// that is not listed yet is `homing`, not `missing`. The `detail` is
/// one line, without paths.
pub fn status(policies: &Value, name: &str, pending: Option<&mut Pending>) -> (&'static str, Option<String>) {
    if !policies.is_object() || policies.get("skills").is_some_and(|s| !s.is_array()) {
        return ("error", Some("unreadable answer from the robot".to_string()));
    }
    let homed = policies.get("homed").and_then(Value::as_bool);
    let listed = has_skill(policies, name);
    let error = change_error(policies);
    let mut pending = pending.filter(|p| p.name == name);
    if let Some(p) = pending.as_deref_mut()
        && homed == Some(false)
    {
        p.seen_homing = true;
    }
    if listed {
        return match homed {
            Some(false) => ("homing", None),
            Some(true) => {
                // A reinstall reads `homed` for the old entry until the
                // reload starts: wait for the homing, a few seconds at most.
                if let Some(p) = pending.as_deref()
                    && p.was_present
                    && !p.seen_homing
                    && p.since.elapsed() < REINSTALL_GRACE
                {
                    return ("homing", None);
                }
                ("homed", None)
            }
            None => ("homed", Some("this robot does not say when it is homed".to_string())),
        };
    }
    match (pending, error) {
        (Some(p), Some(error)) if p.change_error.as_deref() != Some(error.as_str()) => {
            ("error", Some(one_line(&error)))
        }
        (None, Some(error)) => ("error", Some(one_line(&error))),
        (Some(p), _) if p.seen_homing && homed == Some(true) && p.since.elapsed() >= LOST_AFTER => {
            ("error", Some("the robot homed without loading it; install it again".to_string()))
        }
        (Some(_), _) => ("homing", None),
        (None, None) => ("missing", None),
    }
}

/// robotd's words as one short line, with anything that looks like an
/// absolute path replaced.
fn one_line(text: &str) -> String {
    let line = text.lines().next().unwrap_or("").trim();
    let words: Vec<&str> = line
        .split(' ')
        .map(|w| if w.trim_start_matches(['"', '\'', '(', '`']).starts_with('/') { "<path>" } else { w })
        .collect();
    let mut line = words.join(" ");
    if line.len() > 160 {
        let mut cut = 157;
        while !line.is_char_boundary(cut) {
            cut -= 1;
        }
        line.truncate(cut);
        line.push_str("...");
    }
    line
}

/// `dir/<name>/<file>`, absolute: robotd resolves nothing relative.
pub fn target(dir: &Path, name: &str, file: &str) -> PathBuf {
    let dir = std::path::absolute(dir).unwrap_or_else(|_| dir.to_path_buf());
    dir.join(name).join(file)
}

/// Download `url` to `dir/<name>/<file>` with the size cap and the magic
/// check. Returns the path and the bytes written.
pub fn fetch_to(dir: &Path, name: &str, file: &str, url: &str) -> Result<(PathBuf, u64), &'static str> {
    let started = Instant::now();
    let path = target(dir, name, file);
    let folder = path.parent().ok_or(DOWNLOAD_FAILED)?;
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .timeout_global(Some(DOWNLOAD_TIMEOUT))
        .build()
        .into();
    let response = agent.get(url).call().map_err(|error| {
        tracing::info!(name, error = %kind_of(&error), "policy download failed");
        DOWNLOAD_FAILED
    })?;
    let declared = response
        .headers()
        .get("content-length")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.trim().parse::<u64>().ok());
    if declared.is_some_and(|len| len > MAX_BYTES) {
        tracing::info!(name, bytes = declared, "policy too large");
        return Err(TOO_LARGE);
    }
    let mut buf = Vec::new();
    response
        .into_body()
        .into_reader()
        .take(MAX_BYTES + 1)
        .read_to_end(&mut buf)
        .map_err(|_| DOWNLOAD_FAILED)?;
    if buf.len() as u64 > MAX_BYTES {
        tracing::info!(name, "policy too large");
        return Err(TOO_LARGE);
    }
    if !looks_like_onnx(&buf) {
        tracing::info!(name, bytes = buf.len(), "not an onnx file");
        return Err(NOT_ONNX);
    }
    fs::create_dir_all(folder).map_err(|_| DOWNLOAD_FAILED)?;
    let part = folder.join(format!("{file}.part"));
    if fs::write(&part, &buf).and_then(|()| fs::rename(&part, &path)).is_err() {
        let _ = fs::remove_file(&part);
        return Err(DOWNLOAD_FAILED);
    }
    tracing::info!(
        name,
        bytes = buf.len(),
        download_ms = started.elapsed().as_millis() as u64,
        "policy downloaded"
    );
    Ok((path, buf.len() as u64))
}

/// Drop `dir/<name>` after robotd let go of the skill. Best effort.
pub fn remove_files(dir: &Path, name: &str) {
    if validate_name(name).is_ok() {
        let folder = target(dir, name, DEFAULT_FILE);
        if let Some(folder) = folder.parent() {
            let _ = fs::remove_dir_all(folder);
        }
    }
}

/// The kind of a download error, for the log: never the URL, which can
/// carry a signature.
fn kind_of(error: &ureq::Error) -> String {
    match error {
        ureq::Error::StatusCode(code) => format!("http {code}"),
        ureq::Error::Timeout(_) => "timeout".to_string(),
        ureq::Error::Io(io) => io.kind().to_string(),
        _ => "transport".to_string(),
    }
}

/// The result of an accepted install.
pub fn installed(name: &str, bytes: u64) -> Value {
    json!({"name": name, "bytes": bytes, "stage": "installing"})
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::io::{BufRead, BufReader, Write};
    use std::net::TcpListener;

    #[test]
    fn hub_source_becomes_a_resolve_url() {
        assert_eq!(
            download_url("RemiFabre/microduck-flamingo-cycle", "policy.onnx").unwrap(),
            "https://huggingface.co/RemiFabre/microduck-flamingo-cycle/resolve/main/policy.onnx"
        );
        assert_eq!(
            download_url("https://storage.googleapis.com/b/x.onnx?sig=1", "policy.onnx").unwrap(),
            "https://storage.googleapis.com/b/x.onnx?sig=1"
        );
    }

    #[test]
    fn rejects_bad_sources() {
        for s in [
            "",
            "file:///etc/passwd",
            "http://huggingface.co/a/b",
            "a",
            "a/b/c",
            "../x",
            "https://evil.example/x.onnx",
            "https://huggingface.co@evil.example/x.onnx",
            "https://huggingface.co.evil.example/x.onnx",
            "https://huggingface.co:8443/x.onnx",
            "a/b c",
            "a/..",
            "org/répo",
        ] {
            assert_eq!(download_url(s, "policy.onnx").unwrap_err(), DOWNLOAD_FAILED, "{s}");
        }
    }

    #[test]
    fn rejects_traversal_in_file() {
        for f in ["../policy.onnx", "a/b.onnx", "policy.bin", "", "x.ONNX ", ".onnx", "..onnx", "p\u{e9}.onnx"] {
            assert_eq!(validate_file(f).unwrap_err(), SLOT_REFUSED, "{f}");
        }
        assert!(validate_file("policy.onnx").is_ok());
    }

    #[test]
    fn stock_skill_names_and_bad_names_are_refused() {
        let long = "x".repeat(40);
        for n in [
            "walk", "stand", "sitstand", "sit_toggle", "ground_pick", "kick_left", "kick_right",
            "roulade", "Flamingo", "a", "has space", long.as_str(), "1abc", "../x", "flamingo-cycle",
            "fla\u{e9}",
        ] {
            assert_eq!(validate_name(n).unwrap_err(), SLOT_REFUSED, "{n}");
        }
        assert!(validate_name("flamingo_cycle").is_ok());
    }

    #[test]
    fn onnx_magic_check() {
        assert!(looks_like_onnx(&[0x08, 0x07, 0x12, 0x00]));
        assert!(!looks_like_onnx(b"<!doctype html>"));
        assert!(!looks_like_onnx(&[]));
    }

    #[test]
    fn a_request_takes_the_defaults_and_refuses_other_slots() {
        let request = Request::parse(&json!({"source": "org/repo", "name": "flamingo_cycle"})).unwrap();
        assert_eq!(request.file, "policy.onnx");
        assert_eq!(request.duration_s, DEFAULT_DURATION_S);
        assert_eq!(request.chain, None);
        assert_eq!(request.url, "https://huggingface.co/org/repo/resolve/main/policy.onnx");

        let request = Request::parse(&json!({
            "source": "org/repo", "name": "bow", "slot": "user", "file": "bow.onnx",
            "duration_s": 2.5, "chain": "true"
        }))
        .unwrap();
        assert_eq!((request.duration_s, request.chain), (2.5, Some(true)));
        assert!(request.url.ends_with("/bow.onnx"));

        for args in [
            json!({"source": "org/repo"}),
            json!({"source": "org/repo", "name": "walk"}),
            json!({"source": "org/repo", "name": "bow", "slot": "walk"}),
            json!({"source": "org/repo", "name": "bow", "file": "../x.onnx"}),
            json!({"source": "org/repo", "name": "bow", "duration_s": 0}),
            json!({"source": "org/repo", "name": "bow", "duration_s": 600}),
            json!({"source": "org/repo", "name": "bow", "duration_s": "3"}),
            json!({"source": "org/repo", "name": "bow", "chain": "maybe"}),
        ] {
            assert_eq!(Request::parse(&args).unwrap_err(), SLOT_REFUSED, "{args}");
        }
        assert_eq!(Request::parse(&json!({"name": "bow"})).unwrap_err(), DOWNLOAD_FAILED);
        assert_eq!(
            Request::parse(&json!({"name": "bow", "source": "file:///etc/passwd"})).unwrap_err(),
            DOWNLOAD_FAILED
        );
    }

    #[test]
    fn policy_status_maps_robotd_states() {
        // robot.policies' real shape (duck-ipc-proto `PoliciesResult`):
        // configured skill names, and whether the robot is at home.
        let homed = json!({"mode": "walk", "skills": ["roulade", "flamingo_cycle"], "homed": true});
        assert_eq!(state_of(&homed, "flamingo_cycle"), "homed");
        assert_eq!(state_of(&homed, "nope"), "missing");
        let homing = json!({"skills": ["flamingo_cycle"], "homed": false});
        assert_eq!(state_of(&homing, "flamingo_cycle"), "homing");
        let failed = json!({"skills": [], "homed": true, "change_error": "no such file: /var/lib/x.onnx"});
        assert_eq!(state_of(&failed, "flamingo_cycle"), "error");
        let (_, detail) = status(&failed, "flamingo_cycle", None);
        assert_eq!(detail.unwrap(), "no such file: <path>");
        assert_eq!(state_of(&json!([1, 2]), "x"), "error");
        assert_eq!(state_of(&json!({"skills": "x"}), "x"), "error");
        let old = json!({"skills": ["flamingo_cycle"]});
        assert_eq!(state_of(&old, "flamingo_cycle"), "homed");
    }

    #[test]
    fn a_pending_install_is_homing_until_robotd_lists_it() {
        let before = json!({"skills": ["roulade"], "homed": true, "change_error": "old failure"});
        let mut pending = Pending::before("flamingo_cycle", &before);
        // Before the reload starts, while it homes, and done.
        assert_eq!(status(&before, "flamingo_cycle", Some(&mut pending)).0, "homing");
        let homing = json!({"skills": ["roulade"], "homed": false, "change_error": "old failure"});
        assert_eq!(status(&homing, "flamingo_cycle", Some(&mut pending)).0, "homing");
        let done = json!({"skills": ["roulade", "flamingo_cycle"], "homed": true});
        assert_eq!(status(&done, "flamingo_cycle", Some(&mut pending)).0, "homed");
        // A new failure is this install's.
        let failed = json!({"skills": ["roulade"], "homed": true, "change_error": "bad shape"});
        assert_eq!(status(&failed, "flamingo_cycle", Some(&mut pending)), ("error", Some("bad shape".to_string())));
    }

    #[test]
    fn a_change_robotd_dropped_reads_as_error_once_the_robot_is_home() {
        let before = json!({"skills": [], "homed": true});
        let mut pending = Pending::before("flamingo_cycle", &before);
        let homing = json!({"skills": [], "homed": false});
        assert_eq!(status(&homing, "flamingo_cycle", Some(&mut pending)).0, "homing");
        assert_eq!(status(&before, "flamingo_cycle", Some(&mut pending)).0, "homing");
        pending.since -= LOST_AFTER;
        let (state, detail) = status(&before, "flamingo_cycle", Some(&mut pending));
        assert_eq!(state, "error");
        assert!(detail.unwrap().contains("install it again"));
    }

    #[test]
    fn a_reinstall_waits_for_the_homing_it_triggered() {
        let before = json!({"skills": ["flamingo_cycle"], "homed": true});
        let mut pending = Pending::before("flamingo_cycle", &before);
        assert_eq!(status(&before, "flamingo_cycle", Some(&mut pending)).0, "homing");
        let homing = json!({"skills": ["flamingo_cycle"], "homed": false});
        assert_eq!(status(&homing, "flamingo_cycle", Some(&mut pending)).0, "homing");
        assert_eq!(status(&before, "flamingo_cycle", Some(&mut pending)).0, "homed");
    }

    /// A one-shot HTTP server on loopback answering with `head` (status
    /// line and headers, without the blank line) and `body`.
    pub(crate) fn serve(head: &'static str, body: Vec<u8>) -> (String, std::thread::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}/policy.onnx", listener.local_addr().unwrap());
        let server = std::thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            let mut reader = BufReader::new(stream.try_clone().unwrap());
            loop {
                let mut line = String::new();
                if reader.read_line(&mut line).unwrap_or(0) == 0 || line == "\r\n" {
                    break;
                }
            }
            let mut stream = stream;
            let _ = write!(stream, "{head}\r\nConnection: close\r\n\r\n");
            let _ = stream.write_all(&body);
            let _ = stream.flush();
        });
        (url, server)
    }

    pub(crate) fn ok_head(len: usize) -> &'static str {
        Box::leak(format!("HTTP/1.1 200 OK\r\nContent-Length: {len}").into_boxed_str())
    }

    #[test]
    fn a_policy_lands_in_its_folder_and_nothing_else_does() {
        let dir = tempfile::tempdir().unwrap();
        let body = vec![0x08, 0x08, 0x12, 0x07];
        let (url, server) = serve(ok_head(body.len()), body.clone());
        let (path, bytes) = fetch_to(dir.path(), "flamingo_cycle", "policy.onnx", &url).unwrap();
        server.join().unwrap();
        assert_eq!(bytes, 4);
        assert_eq!(path, dir.path().join("flamingo_cycle/policy.onnx"));
        assert_eq!(fs::read(&path).unwrap(), body);
        assert!(!dir.path().join("flamingo_cycle/policy.onnx.part").exists());

        let (url, server) = serve(ok_head(15), b"<!doctype html>".to_vec());
        assert_eq!(fetch_to(dir.path(), "html_page", "policy.onnx", &url).unwrap_err(), NOT_ONNX);
        server.join().unwrap();
        assert!(!dir.path().join("html_page").exists());

        let (url, server) = serve("HTTP/1.1 200 OK\r\nContent-Length: 30000000", Vec::new());
        assert_eq!(fetch_to(dir.path(), "huge", "policy.onnx", &url).unwrap_err(), TOO_LARGE);
        server.join().unwrap();

        let (url, server) = serve("HTTP/1.1 404 Not Found\r\nContent-Length: 0", Vec::new());
        assert_eq!(fetch_to(dir.path(), "gone", "policy.onnx", &url).unwrap_err(), DOWNLOAD_FAILED);
        server.join().unwrap();
    }

    #[test]
    fn a_body_past_the_cap_without_a_length_is_too_large() {
        let dir = tempfile::tempdir().unwrap();
        let mut body = vec![0u8; (MAX_BYTES + 10) as usize];
        body[0] = 0x08;
        let (url, server) = serve("HTTP/1.1 200 OK", body);
        assert_eq!(fetch_to(dir.path(), "huge", "policy.onnx", &url).unwrap_err(), TOO_LARGE);
        server.join().unwrap();
        assert!(!dir.path().join("huge").exists());
    }
}

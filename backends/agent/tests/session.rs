//! Full agent-protocol conversation against a scripted bridge over a
//! real localhost WebSocket: session.start handshake, wake → audio
//! streaming → utterance.end, tool calls (unsupported / unknown /
//! no-robot), TTS round with follow-up listen.start, clean close.

use std::net::TcpListener;
use std::sync::mpsc::sync_channel;

use quacksat_backend_agent::session::{Deps, run_session};
use quacksat_core::audio::FRAME_SAMPLES;
use quacksat_core::config::Config;
use quacksat_core::listen::MIN_LISTEN_FRAMES;
use quacksat_core::playback::Player;
use quacksat_core::tools::Robot;
use quacksat_core::wake;
use serde_json::{Value, json};
use tungstenite::Message;

fn loud_frame() -> Vec<i16> {
    (0..FRAME_SAMPLES)
        .map(|i| if i % 2 == 0 { 8000 } else { -8000 })
        .collect()
}

fn quiet_frame() -> Vec<i16> {
    vec![10i16; FRAME_SAMPLES]
}

fn fake_aplay(dir: &std::path::Path) -> String {
    let script = dir.join("fake-aplay");
    // Exits immediately: tests feed frames in bursts, and a lingering
    // ack player would gate them all away in one drain.
    std::fs::write(&script, "#!/bin/sh\nexit 0\n").unwrap();
    use std::os::unix::fs::PermissionsExt;
    let mut perms = std::fs::metadata(&script).unwrap().permissions();
    perms.set_mode(0o755);
    std::fs::set_permissions(&script, perms).unwrap();
    // Run it once here: macOS inspects a freshly written executable on
    // its first exec, which can take longer than the player's settle
    // window — the ack then looked "still playing" and gated every test
    // frame away (a 1-in-3 failure until this line).
    // On Linux a fresh executable can briefly be "text file busy" while a
    // parallel test's fork still holds its write descriptor: retry.
    for attempt in 0.. {
        match std::process::Command::new(&script).status() {
            Ok(_) => break,
            Err(e) if e.raw_os_error() == Some(26) && attempt < 50 => {
                std::thread::sleep(std::time::Duration::from_millis(20))
            }
            Err(e) => panic!("cannot run the fake aplay: {e}"),
        }
    }
    script.to_str().unwrap().to_string()
}

#[test]
fn full_conversation_flow() {
    let config: Config = toml::from_str("backend = \"agent\"").unwrap();
    let dir = tempfile::tempdir().unwrap();
    let aplay = fake_aplay(dir.path());

    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let (frames_tx, frames_rx) = sync_channel::<Vec<i16>>(256);

    std::thread::scope(|scope| {
        let config_ref = &config;
        // The satellite side under test.
        let satellite = scope.spawn(move || {
            let (ws, _) = tungstenite::connect(format!("ws://127.0.0.1:{port}")).expect("connect");
            let mut detector = wake::from_config(&config_ref.wake).unwrap();
            let mut player = Player::with_program("ignored", &aplay);
            let mut robot = Robot::detached();
            run_session(
                ws,
                &mut Deps {
                    config: config_ref,
                    frames: &frames_rx,
                    detector: detector.as_mut(),
                    player: &mut player,
                    robot: &mut robot,
                },
            )
        });

        // The scripted bridge.
        let (stream, _) = listener.accept().unwrap();
        let mut bridge = tungstenite::accept(stream).unwrap();
        let next_json = |bridge: &mut tungstenite::WebSocket<std::net::TcpStream>| -> Value {
            loop {
                match bridge.read().unwrap() {
                    Message::Text(text) => return serde_json::from_str(&text).unwrap(),
                    Message::Binary(_) => continue,
                    other => panic!("unexpected message: {other:?}"),
                }
            }
        };
        let send = |bridge: &mut tungstenite::WebSocket<std::net::TcpStream>, event: Value| {
            bridge.send(Message::text(event.to_string())).unwrap();
        };

        // 1. Handshake: session.start with the tool catalog.
        let start = next_json(&mut bridge);
        assert_eq!(start["type"], "session.start");
        assert_eq!(start["audio"]["rate"], 16_000);
        let tools: Vec<&str> = start["tools"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| t["name"].as_str().unwrap())
            .collect();
        assert!(tools.contains(&"robot.move"));
        assert!(tools.contains(&"robot.get_frame"));
        send(&mut bridge, json!({"type": "session.ready", "version": 1}));

        // 2. Tool calls while idle: unsupported, unknown, and no-robot.
        send(
            &mut bridge,
            json!({"type": "tool.call", "id": "t1", "name": "robot.get_frame", "args": {}}),
        );
        let result = next_json(&mut bridge);
        assert_eq!(result["type"], "tool.result");
        assert_eq!(result["id"], "t1");
        assert_eq!(result["ok"], false);
        assert_eq!(result["error"], "unsupported");

        send(
            &mut bridge,
            json!({"type": "tool.call", "id": "t2", "name": "robot.fly", "args": {}}),
        );
        let result = next_json(&mut bridge);
        assert_eq!(result["ok"], false);
        assert_eq!(result["error"], "unknown tool `robot.fly`");

        send(
            &mut bridge,
            json!({"type": "tool.call", "id": "t3", "name": "robot.state", "args": {}}),
        );
        let result = next_json(&mut bridge);
        assert_eq!(result["ok"], false);
        assert_eq!(result["error"], "robot unreachable");

        // 3. Wake: quiet pre-roll then speech; expect wake, binary audio,
        // and utterance.end once silence returns.
        for _ in 0..2 {
            frames_tx.send(quiet_frame()).unwrap();
        }
        for _ in 0..2 {
            frames_tx.send(loud_frame()).unwrap();
        }
        let wake = next_json(&mut bridge);
        assert_eq!(wake["type"], "wake");
        // Let the local wake ack finish before feeding the utterance:
        // burst-fed frames arrive faster than real time, and the
        // half-duplex gate would otherwise drain them all away while
        // the ack child lingers.
        std::thread::sleep(std::time::Duration::from_millis(500));
        for _ in 0..9 {
            frames_tx.send(loud_frame()).unwrap();
        }
        // Silence to the end of the listening window: the turn is held
        // open for MIN_LISTEN_FRAMES whatever it hears, so the hangover
        // alone no longer closes it (`quacksat_core::listen`).
        for _ in 0..MIN_LISTEN_FRAMES + 10 {
            frames_tx.send(quiet_frame()).unwrap();
        }

        let mut audio_frames = 0;
        loop {
            match bridge.read().unwrap() {
                Message::Binary(bytes) => {
                    assert_eq!(bytes.len(), FRAME_SAMPLES * 2);
                    audio_frames += 1;
                }
                Message::Text(text) => {
                    let event: Value = serde_json::from_str(&text).unwrap();
                    assert_eq!(event["type"], "utterance.end");
                    break;
                }
                other => panic!("unexpected message: {other:?}"),
            }
        }
        assert!(audio_frames >= 12, "pre-roll + burst must be streamed");

        // 4. TTS round, then a follow-up turn without a wake word.
        send(
            &mut bridge,
            json!({"type": "tts.start", "rate": 22_050, "channels": 1}),
        );
        bridge.send(Message::binary(vec![0u8; 4410])).unwrap();
        send(&mut bridge, json!({"type": "tts.end"}));
        send(&mut bridge, json!({"type": "listen.start"}));
        send(&mut bridge, json!({"type": "ping", "text": "alive"}));
        let pong = next_json(&mut bridge);
        assert_eq!(pong["type"], "pong");
        assert_eq!(pong["text"], "alive");

        // The follow-up turn streams without any wake event — but not
        // before the duck's own reply has stopped coming out of the
        // speaker: `is_sounding` holds the mic shut for the tail after
        // the playback program exits, and these frames are burst-fed.
        std::thread::sleep(std::time::Duration::from_millis(400));
        for _ in 0..6 {
            frames_tx.send(loud_frame()).unwrap();
        }
        let mut followup_frames = 0;
        while followup_frames < 6 {
            match bridge.read().unwrap() {
                Message::Binary(_) => followup_frames += 1,
                Message::Text(text) => {
                    let event: Value = serde_json::from_str(&text).unwrap();
                    panic!("no events expected during follow-up, got {event}");
                }
                other => panic!("unexpected message: {other:?}"),
            }
        }

        // 5. Bridge closes; the session ends cleanly.
        bridge.close(None).unwrap();
        let _ = bridge.read(); // flush the close handshake
        satellite.join().unwrap().unwrap();
    });
}

/// A bridge that never answers: after `utterance.end` the satellite must
/// give up at `[thinking] timeout_s` and play the sad ack locally (the
/// counting fake aplay sees wake ack + sad ack = two runs).
#[test]
fn reply_timeout_ends_with_sad_ack() {
    let config: Config =
        toml::from_str("backend = \"agent\"\n[thinking]\ndelay_s = 0.05\ntimeout_s = 0.2\n")
            .unwrap();
    let dir = tempfile::tempdir().unwrap();
    let plays = dir.path().join("plays");
    let script = dir.path().join("counting-aplay");
    std::fs::write(
        &script,
        format!("#!/bin/sh\necho run >> {}\nexit 0\n", plays.display()),
    )
    .unwrap();
    use std::os::unix::fs::PermissionsExt;
    let mut perms = std::fs::metadata(&script).unwrap().permissions();
    perms.set_mode(0o755);
    std::fs::set_permissions(&script, perms).unwrap();
    // Run it once and wipe what that wrote: macOS inspects a freshly
    // written executable on its first exec, which outlasts the player's
    // open check — the ack child then looks alive while the test bursts
    // its utterance in, and the half-duplex gate eats the lot.
    // On Linux a fresh executable can briefly be "text file busy" while a
    // parallel test's fork still holds its write descriptor: retry.
    for attempt in 0.. {
        match std::process::Command::new(&script).status() {
            Ok(_) => break,
            Err(e) if e.raw_os_error() == Some(26) && attempt < 50 => {
                std::thread::sleep(std::time::Duration::from_millis(20))
            }
            Err(e) => panic!("cannot run the fake aplay: {e}"),
        }
    }
    std::fs::write(&plays, "").unwrap();
    let aplay = script.to_str().unwrap().to_string();

    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let (frames_tx, frames_rx) = sync_channel::<Vec<i16>>(256);

    std::thread::scope(|scope| {
        let config_ref = &config;
        let satellite = scope.spawn(move || {
            let (ws, _) = tungstenite::connect(format!("ws://127.0.0.1:{port}")).expect("connect");
            let mut detector = wake::from_config(&config_ref.wake).unwrap();
            let mut player = Player::with_program("ignored", &aplay);
            let mut robot = Robot::detached();
            run_session(
                ws,
                &mut Deps {
                    config: config_ref,
                    frames: &frames_rx,
                    detector: detector.as_mut(),
                    player: &mut player,
                    robot: &mut robot,
                },
            )
        });

        let (stream, _) = listener.accept().unwrap();
        let mut bridge = tungstenite::accept(stream).unwrap();
        let start: Value = match bridge.read().unwrap() {
            Message::Text(text) => serde_json::from_str(&text).unwrap(),
            other => panic!("unexpected message: {other:?}"),
        };
        assert_eq!(start["type"], "session.start");

        // Wake on a short burst.
        for _ in 0..2 {
            frames_tx.send(quiet_frame()).unwrap();
        }
        for _ in 0..2 {
            frames_tx.send(loud_frame()).unwrap();
        }
        loop {
            match bridge.read().unwrap() {
                Message::Binary(_) => continue,
                Message::Text(text) => {
                    let event: Value = serde_json::from_str(&text).unwrap();
                    assert_eq!(event["type"], "wake");
                    break;
                }
                other => panic!("unexpected message: {other:?}"),
            }
        }
        // Let the local wake ack finish before feeding the utterance:
        // burst-fed frames arrive faster than real time, and the
        // half-duplex gate would otherwise drain them all away while
        // the ack child lingers.
        std::thread::sleep(std::time::Duration::from_millis(500));
        for _ in 0..20 {
            frames_tx.send(loud_frame()).unwrap();
        }
        for _ in 0..MIN_LISTEN_FRAMES + 10 {
            frames_tx.send(quiet_frame()).unwrap();
        }
        loop {
            match bridge.read().unwrap() {
                Message::Binary(_) => continue,
                Message::Text(text) => {
                    let event: Value = serde_json::from_str(&text).unwrap();
                    assert_eq!(event["type"], "utterance.end");
                    break;
                }
                other => panic!("unexpected message: {other:?}"),
            }
        }

        // The bridge stays silent past timeout_s; the satellite keeps
        // polling on its own and must give up with the sad ack.
        std::thread::sleep(std::time::Duration::from_millis(800));
        let log = std::fs::read_to_string(&plays).unwrap_or_default();
        assert_eq!(
            log.lines().count(),
            2,
            "expected wake ack + sad ack, got: {log:?}"
        );

        bridge.close(None).unwrap();
        let _ = bridge.read(); // flush the close handshake
        satellite.join().unwrap().unwrap();
    });
}

/// The duck's own sentence (`quacksat_core::announce`): handed to a
/// bridge that lists `say` in session.ready, in the language it reports,
/// and the wake word waits until the clip is over; a bridge that does not
/// list it never gets one.
#[test]
fn the_ducks_own_sentence_goes_to_a_bridge_that_says() {
    use quacksat_core::announce::{Announcer, Event, Job};

    for says in [true, false] {
        let config: Config = toml::from_str("backend = \"agent\"").unwrap();
        let dir = tempfile::tempdir().unwrap();
        let aplay = fake_aplay(dir.path());
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let (_frames_tx, frames_rx) = sync_channel::<Vec<i16>>(256);
        let announcer = Announcer::detached();
        announcer.push(Event::Arrived(Job::GoToPlace("cucina".into())));

        std::thread::scope(|scope| {
            let config_ref = &config;
            let announcer_ref = &announcer;
            let satellite = scope.spawn(move || {
                let (ws, _) = tungstenite::connect(format!("ws://127.0.0.1:{port}")).expect("connect");
                let mut detector = wake::from_config(&config_ref.wake).unwrap();
                let mut player = Player::with_program("ignored", &aplay);
                let mut robot = Robot::detached();
                robot.announcer = Some(announcer_ref.clone());
                run_session(
                    ws,
                    &mut Deps {
                        config: config_ref,
                        frames: &frames_rx,
                        detector: detector.as_mut(),
                        player: &mut player,
                        robot: &mut robot,
                    },
                )
            });

            let (stream, _) = listener.accept().unwrap();
            let mut bridge = tungstenite::accept(stream).unwrap();
            let next_json = |bridge: &mut tungstenite::WebSocket<std::net::TcpStream>| -> Value {
                loop {
                    match bridge.read().unwrap() {
                        Message::Text(text) => return serde_json::from_str(&text).unwrap(),
                        Message::Binary(_) => continue,
                        other => panic!("unexpected message: {other:?}"),
                    }
                }
            };
            assert_eq!(next_json(&mut bridge)["type"], "session.start");
            let ready = if says {
                json!({"type": "session.ready", "version": 1, "features": ["say"], "language": "it"})
            } else {
                json!({"type": "session.ready", "version": 1})
            };
            bridge.send(Message::text(ready.to_string())).unwrap();

            if says {
                let say = next_json(&mut bridge);
                assert_eq!(say["type"], "say");
                assert_eq!(say["text"], "Sono arrivata in cucina.");
                assert_eq!(say["lang"], "it");
                assert!(say["id"].as_str().is_some_and(|id| !id.is_empty()));
                let start = json!({"type": "tts.start", "rate": 22_050, "channels": 1, "say": say["id"]});
                bridge.send(Message::text(start.to_string())).unwrap();
                bridge.send(Message::binary(vec![0u8; 4410])).unwrap();
                bridge.send(Message::text(json!({"type": "tts.end"}).to_string())).unwrap();
            } else {
                // Give the session time to take the sentence and drop it.
                std::thread::sleep(std::time::Duration::from_millis(300));
            }
            // Nothing else was sent: the next text is the pong.
            bridge.send(Message::text(json!({"type": "ping", "n": 1}).to_string())).unwrap();
            assert_eq!(next_json(&mut bridge)["type"], "pong");
            assert_eq!(announcer.next(), None, "the sentence was taken once");

            bridge.close(None).unwrap();
            let _ = bridge.read();
            satellite.join().unwrap().unwrap();
        });
    }
}

/// A fake mediad on a temp socket serving a 64x36 frame per connection.
fn fake_mediad() -> (tempfile::TempDir, String) {
    use std::io::{BufRead, BufReader, Write};
    use std::os::unix::net::UnixListener;
    let (width, height) = (64u32, 36u32);
    let mut pixels = Vec::new();
    for _y in 0..height {
        for x in (0..width).step_by(2) {
            let luma = (16 + (x * 219) / width) as u8;
            pixels.extend_from_slice(&[128, luma, 128, luma]);
        }
    }
    let header = json!({"width": width, "height": height, "format": "UYVY", "bytes": pixels.len(),
                        "captured_at_unix_us": 1, "rotate": 0});
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
            let mut out = serde_json::to_vec(&json!({"jsonrpc": "2.0", "id": 1, "result": header})).unwrap();
            out.push(b'\n');
            let mut writer = stream;
            let _ = writer.write_all(&out);
            let _ = writer.write_all(&pixels);
        }
    });
    (dir, socket.to_str().unwrap().to_string())
}

fn is_jpeg(bytes: &[u8]) -> bool {
    bytes.starts_with(&[0xFF, 0xD8, 0xFF]) && bytes.ends_with(&[0xFF, 0xD9])
}

#[test]
fn streaming_sends_jpegs_and_a_wake_turn_still_works() {
    let config: Config = toml::from_str("backend = \"agent\"").unwrap();
    let dir = tempfile::tempdir().unwrap();
    let aplay = fake_aplay(dir.path());
    let (_media_dir, media) = fake_mediad();
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let (frames_tx, frames_rx) = sync_channel::<Vec<i16>>(256);

    std::thread::scope(|scope| {
        let config_ref = &config;
        let satellite = scope.spawn(move || {
            let (ws, _) = tungstenite::connect(format!("ws://127.0.0.1:{port}")).expect("connect");
            let mut detector = wake::from_config(&config_ref.wake).unwrap();
            let mut player = Player::with_program("ignored", &aplay);
            let mut robot = Robot::detached();
            robot.camera = media;
            run_session(ws, &mut Deps { config: config_ref, frames: &frames_rx, detector: detector.as_mut(),
                                        player: &mut player, robot: &mut robot })
        });
        let (stream, _) = listener.accept().unwrap();
        let mut bridge = tungstenite::accept(stream).unwrap();
        let next_text = |bridge: &mut tungstenite::WebSocket<std::net::TcpStream>| -> Value {
            loop {
                match bridge.read().unwrap() {
                    Message::Text(text) => return serde_json::from_str(&text).unwrap(),
                    Message::Binary(_) => continue,
                    other => panic!("unexpected message: {other:?}"),
                }
            }
        };
        let start = next_text(&mut bridge);
        let names: Vec<&str> = start["tools"].as_array().unwrap().iter().map(|t| t["name"].as_str().unwrap()).collect();
        assert!(names.contains(&"robot.stream"));
        bridge.send(Message::text(json!({"type": "session.ready", "version": 1}).to_string())).unwrap();
        bridge.send(Message::text(json!({"type": "tool.call", "id": "s1", "name": "robot.stream",
                                         "args": {"on": true, "fps": 8}}).to_string())).unwrap();
        let result = next_text(&mut bridge);
        assert_eq!(result["id"], "s1");
        assert_eq!(result["data"], json!({"on": true, "fps": 8, "width": 320}));

        let until = std::time::Instant::now() + std::time::Duration::from_secs(1);
        let mut jpegs = 0;
        while std::time::Instant::now() < until {
            if let Message::Binary(bytes) = bridge.read().unwrap() {
                assert!(is_jpeg(&bytes), "only video is binary before a wake");
                jpegs += 1;
            }
        }
        assert!(jpegs >= 5, "{jpegs}");

        // A wake turn while streaming: mic frames and JPEGs interleave, the turn still closes.
        for _ in 0..2 { frames_tx.send(quiet_frame()).unwrap(); }
        for _ in 0..2 { frames_tx.send(loud_frame()).unwrap(); }
        assert_eq!(next_text(&mut bridge)["type"], "wake");
        std::thread::sleep(std::time::Duration::from_millis(500));
        for _ in 0..9 { frames_tx.send(loud_frame()).unwrap(); }
        for _ in 0..MIN_LISTEN_FRAMES + 10 { frames_tx.send(quiet_frame()).unwrap(); }
        let (mut audio, mut video) = (0, 0);
        loop {
            match bridge.read().unwrap() {
                Message::Binary(bytes) if is_jpeg(&bytes) => video += 1,
                Message::Binary(bytes) => { assert_eq!(bytes.len(), FRAME_SAMPLES * 2); audio += 1; }
                Message::Text(text) => {
                    let event: Value = serde_json::from_str(&text).unwrap();
                    assert_eq!(event["type"], "utterance.end");
                    break;
                }
                other => panic!("unexpected message: {other:?}"),
            }
        }
        assert!(audio >= 12, "pre-roll + speech must be streamed ({audio})");
        assert!(video >= 1, "video keeps flowing during the turn");
        bridge.close(None).unwrap();
        let _ = bridge.read();
        satellite.join().unwrap().unwrap();
    });
}

#[test]
fn the_stream_stops_after_ten_quiet_seconds() {
    let config: Config = toml::from_str("backend = \"agent\"").unwrap();
    let dir = tempfile::tempdir().unwrap();
    let aplay = fake_aplay(dir.path());
    let (_media_dir, media) = fake_mediad();
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let (_frames_tx, frames_rx) = sync_channel::<Vec<i16>>(8);
    std::thread::scope(|scope| {
        let config_ref = &config;
        let satellite = scope.spawn(move || {
            let (ws, _) = tungstenite::connect(format!("ws://127.0.0.1:{port}")).expect("connect");
            let mut detector = wake::from_config(&config_ref.wake).unwrap();
            let mut player = Player::with_program("ignored", &aplay);
            let mut robot = Robot::detached();
            robot.camera = media;
            run_session(ws, &mut Deps { config: config_ref, frames: &frames_rx, detector: detector.as_mut(),
                                        player: &mut player, robot: &mut robot })
        });
        let (stream, _) = listener.accept().unwrap();
        stream.set_read_timeout(Some(std::time::Duration::from_millis(200))).unwrap();
        let mut bridge = tungstenite::accept(stream).unwrap();
        bridge.send(Message::text(json!({"type": "session.ready", "version": 1}).to_string())).unwrap();
        bridge.send(Message::text(json!({"type": "tool.call", "id": "s1", "name": "robot.stream",
                                         "args": {"on": true}}).to_string())).unwrap();
        let started = std::time::Instant::now();
        let (mut early, mut late) = (0, 0);
        while started.elapsed() < std::time::Duration::from_millis(11_500) {
            match bridge.read() {
                Ok(Message::Binary(bytes)) if is_jpeg(&bytes) => {
                    if started.elapsed() < std::time::Duration::from_secs(9) { early += 1 }
                    else if started.elapsed() > std::time::Duration::from_millis(10_700) { late += 1 }
                }
                Ok(_) => {}
                Err(tungstenite::Error::Io(e)) if matches!(e.kind(), std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut) => {}
                Err(e) => panic!("{e}"),
            }
        }
        assert!(early > 20, "{early}");
        assert_eq!(late, 0, "frames still came 10 s after the agent went quiet");
        bridge.close(None).unwrap();
        let _ = bridge.read();
        satellite.join().unwrap().unwrap();
    });
}

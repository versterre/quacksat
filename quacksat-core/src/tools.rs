//! The robot tool surface (ADR 0004 §4, protocol spec "Tool surface v1"):
//! declared in session.start, executed here behind an exhaustive
//! allowlist with satellite-side clamps. A new tool does not exist on the
//! wire until it is added to BOTH `catalog()` and `execute()`.
//!
//! Tools act on a [`Robot`]: the robotd request lane, and — when the
//! navigation daemon is listening on its socket — the nav lane, whose
//! catalog is spliced here
//! and routed back to it by name.

use crate::config::Config;
use crate::body::{clamp, move_params, notify, number, request, require_str, timed_move, trimmed, with_robot};
use crate::nav_client::NavLane;
use crate::robotd::{Control, Lane};
use duck_ipc_proto as proto;
use serde_json::{Value, json};

/// Hard caps an LLM can never exceed, whatever it asks for. The speed
/// and yaw caps live with the body's lane in [`crate::body`], which is
/// what actually builds the wire command.
const MAX_MOVE_DURATION_S: f64 = 3.0;
const MAX_LOOK_XY_M: f64 = 3.0;
const MIN_LOOK_Z_M: f64 = -0.2;
const MAX_LOOK_Z_M: f64 = 2.0;
const MAX_HEAD_PITCH_RAD: f64 = 0.6;
const MAX_HEAD_YAW_RAD: f64 = 1.2;
const MAX_HEAD_ROLL_RAD: f64 = 0.5;

/// What `robot.do` answered to before daemon 0.14 made the skill table
/// config, and what a stock robot still answers to. The fallback when the
/// robot does not say: an older daemon refuses `robot.skills` with
/// METHOD_NOT_FOUND, and an unreachable one says nothing at all.
pub const STOCK_SKILLS: [&str; 5] = [
    "ground_pick",
    "kick_left",
    "kick_right",
    "sit_toggle",
    "roulade",
];

fn stock_skills() -> Vec<String> {
    STOCK_SKILLS.iter().map(|name| (*name).to_string()).collect()
}

/// Ask the robot what it can do, once, at connect time.
///
/// Since daemon 0.14 (API v22) a skill is a `[[policy.skill]]` entry, so
/// which ones exist is this robot's business and not ours to assume —
/// `proto::Skill` stopped being an enum for the same reason. The answer
/// carries the configured table and the two the daemon drives itself
/// (`ground_pick`, `sit_toggle`); an agent choosing a skill needs both.
fn skills_of(control: &mut Option<Control>) -> Vec<String> {
    let reported = request(control, &proto::Call::RobotSkills)
        .ok()
        .filter(|response| response.error.is_none())
        .and_then(|response| response.result_as::<proto::SkillsResult>().ok())
        .map(|result| {
            result
                .skills
                .into_iter()
                .map(|skill| skill.name)
                .chain(result.built_in)
                .collect::<Vec<String>>()
        })
        .filter(|names| !names.is_empty());
    match reported {
        Some(names) => {
            tracing::info!(skills = ?names, "the robot listed its skills");
            names
        }
        None => {
            tracing::info!("the robot did not list its skills — assuming the stock five");
            stock_skills()
        }
    }
}

pub struct Robot {
    /// The request lane: discrete intents and queries, dialled again
    /// when it dies (`Lane`).
    pub lane: Lane,
    /// The `[gait]` section: yaw trim and per-side gains for every walk.
    pub gait: crate::config::GaitConfig,
    /// What `robot.skill` may be asked for: the robot's own list, or the
    /// stock five when it does not say. See [`skills_of`].
    pub skills: Vec<String>,
    /// The navigation daemon, when one answers on its socket: its tools
    /// are announced beside the satellite's and executed there
    /// (`quack-navd`, the split of 2026-09-22).
    pub nav: Option<NavLane>,
    /// What the duck says on its own: told of every job a tool call
    /// here starts, so it can say how the job ended. Set by the
    /// backends that can speak outside a turn (`direct`, `agent`).
    pub announcer: Option<crate::announce::Announcer>,
    /// mediad's frame socket for `robot.get_frame` (`[camera]
    /// frame_socket`); empty when this robot has no camera lane.
    pub camera: String,
    /// Where `robot.install_policy` writes policies (`[install] dir`).
    pub install_dir: std::path::PathBuf,
    /// The install `robot.policy_status` is following, if any.
    pub pending_install: Option<crate::install::Pending>,
}

impl Robot {
    /// Connect every lane the config asks for. Nothing here is fatal: a
    /// missing robotd or navigation daemon degrades to "the tool says
    /// so" at call time.
    pub fn connect(config: &Config) -> Self {
        let mut lane = Lane::connect(&config.robotd_socket);
        let skills = skills_of(&mut lane.control);
        Self {
            lane,
            gait: config.gait.clone(),
            skills,
            nav: NavLane::probe(&config.nav),
            announcer: None,
            camera: config.camera.frame_socket.clone(),
            install_dir: config.install.dir.clone().into(),
            pending_install: None,
        }
    }

    /// Give the lane its chance to come back, and re-read the skill
    /// list when it does: a robot that restarted under us — which is
    /// what an update looks like from here — may not have the same one.
    pub fn redial(&mut self) {
        if !self.lane.redial() {
            return;
        }
        let skills = skills_of(&mut self.lane.control);
        if skills != self.skills {
            tracing::info!(?skills, "the robot came back with a different skill list");
        }
        self.skills = skills;
    }

    /// No robotd, no navigation (tests, dry runs).
    pub fn detached() -> Self {
        Self {
            lane: Lane::detached(),
            gait: crate::config::GaitConfig::default(),
            skills: stock_skills(),
            nav: None,
            announcer: None,
            camera: String::new(),
            install_dir: crate::install::DEFAULT_DIR.into(),
            pending_install: None,
        }
    }
}

/// The catalog announced in `session.start`. JSON-Schema parameters,
/// directly projectable to OpenAI tools and MCP listings.
pub fn catalog(robot: &Robot) -> Value {
    let mut tools = vec![
        json!({
            "name": "robot.sound",
            "description": "Play an expressive duck sound. Use for reactions or when asked to \
        quack or make a sound. Tags: alarm (loud alert), greet (hello), inquire (questioning), \
        peck, chirp (short acknowledgement), coo (affectionate).",
            "parameters": {
                "type": "object",
                "properties": {"tag": {"type": "string", "enum": ["alarm", "greet", "inquire", "peck", "chirp", "coo"]}},
                "required": ["tag"]
            }
        }),
        json!({
            "name": "robot.look",
            "description": "Aim the duck's gaze at a point in space. Use when asked to look at \
        something or somewhere. Coordinates in meters from the duck's chest: x forward, y left, \
        z up (the floor is about 0.12 m below; a standing person's face is around z=1.5 at their \
        distance). The gaze holds until changed. Example: look at something on the floor one \
        meter ahead: x=1.0, z=-0.1.",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "number", "description": "meters forward of the duck"},
                    "y": {"type": "number", "description": "meters to the duck's left (negative = right)"},
                    "z": {"type": "number", "description": "meters above the duck's chest (floor is -0.12)"}
                },
                "required": ["x"]
            }
        }),
        json!({
            "name": "robot.head",
            "description": "Strike an expressive head pose (for looking AT something use \
        robot.look instead): roll tilts the head sideways like a curious dog, yaw turns it, \
        pitch nods it. Angles in radians, clamped; omitted angles return to center. The pose \
        holds until the next call; call with no arguments to re-center.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pitch": {"type": "number", "description": "nod up/down, about -0.6 to 0.6"},
                    "yaw": {"type": "number", "description": "turn left/right, about -1.2 to 1.2, + is left"},
                    "roll": {"type": "number", "description": "sideways tilt, about -0.5 to 0.5"}
                }
            }
        }),
        json!({
            "name": "robot.skill",
            "description": "Run a one-shot skill; it takes a few seconds. ground_pick pecks at \
        the ground, kick_left/kick_right kick, sit_toggle sits down or stands back up (it \
        toggles), roulade does a somersault. The list below is this robot's own — since daemon \
        0.14 its skills are configurable, so a name that is not there does not exist here.",
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string", "enum": &robot.skills}},
                "required": ["name"]
            }
        }),
        json!({
            "name": "robot.move",
            "description": "Walk or turn for a bounded time, then stop automatically. Use when \
        asked to move, approach, back away, or turn. Command vx=0.3 to walk: the gait does not \
        start below about 0.25 m/s, and 0.3 commanded moves the duck roughly 10 cm per second, \
        so vx=0.3 with duration_s=3 walks about 30 cm. The duck cannot turn in place: to turn, \
        walk with vx=0.3 and vyaw=0.7 (left) or -0.7 (right) for about 2 s per 90 degrees, \
        moving 15-20 cm meanwhile. Nothing watches where this walk goes — the cliff guard rides \
        with the navigation's own journeys, not with this tool — so keep it short and look first. \
        Values are clamped (0.3 m/s, 1.0 rad/s, 3 s max). For longer distances \
        call repeatedly, checking robot.state in between.",
            "parameters": {
                "type": "object",
                "properties": {
                    "vx": {"type": "number", "description": "m/s forward (+) / backward (-)"},
                    "vy": {"type": "number", "description": "m/s sidestep left (+) / right (-)"},
                    "vyaw": {"type": "number", "description": "rad/s turn, + is left/counterclockwise"},
                    "duration_s": {"type": "number", "minimum": 0.1, "maximum": MAX_MOVE_DURATION_S}
                },
                "required": ["duration_s"]
            }
        }),
        json!({
            "name": "robot.state",
            "description": "Current robot status: health, battery, mode. Use it before and \
        after moving, or when asked how the robot is doing.",
            "parameters": {"type": "object", "properties": {}}
        }),
    ];
    let get_frame = if robot.camera.trim().is_empty() {
        "Grab a camera frame. Not supported yet on this robot — if it \
    fails, tell the user you cannot see yet."
    } else {
        "Grab what the duck's head camera sees right now: an upright JPEG \
    (jpeg_b64, base64) with its width and height. Use it when asked what the duck \
    sees, or to look before walking. If it fails, say you cannot see right now."
    };
    tools.push(json!({
        "name": "robot.get_frame",
        "description": get_frame,
        "parameters": {"type": "object", "properties": {}}
    }));
    tools.push(json!({
        "name": "robot.install_policy",
        "description": "Install a policy from the Hugging Face Hub (source org/repo) or an \
    https URL as a new user skill named `name`; the duck then homes (moves its joints) for \
    a few seconds. Follow it with robot.policy_status until homed. Never replaces the \
    robot's own skills.",
        "parameters": {
            "type": "object",
            "properties": {
                "source": {"type": "string", "description": "Hub repo `org/repo`, or an https URL"},
                "file": {"type": "string", "description": "the .onnx in the repo, default policy.onnx"},
                "name": {"type": "string", "description": "the skill's name: lowercase letters, digits, _"},
                "slot": {"type": "string", "enum": ["user"]},
                "duration_s": {"type": "number", "description": "seconds it runs, default 3", "minimum": 0.1, "maximum": crate::install::MAX_DURATION_S},
                "chain": {"type": "string", "enum": ["true", "false"], "description": "whether a request while it runs queues another"}
            },
            "required": ["source", "name"]
        }
    }));
    tools.push(json!({
        "name": "robot.policy_status",
        "description": "Where an installed skill stands: homing (the duck is moving to its \
    home pose), homed (ready for robot.skill), missing or error.",
        "parameters": {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"]
        }
    }));
    tools.push(json!({
        "name": "robot.remove_policy",
        "description": "Take an installed user skill off the duck again.",
        "parameters": {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"]
        }
    }));
    if let Some(nav) = robot.nav.as_ref() {
        // The navigation daemon's own tools, announced as if they were
        // ours: the agent sees one robot, `execute` routes by name.
        tools.extend(nav.catalog());
    }
    Value::Array(tools)
}

/// Run one tool call. Every name here is in `catalog()`, or belongs to
/// the navigation daemon.
pub fn execute(name: &str, args: &Value, robot: &mut Robot) -> Result<Value, String> {
    // Anything the navigation daemon answers for goes there: the map,
    // the places, the steps and the journeys live in `quack-navd` since
    // the split of 2026-09-22, and the satellite only carries the wire.
    if let Some(nav) = &mut robot.nav
        && nav.handles(name)
    {
        let answer = nav.call(name, args);
        // A journey answers at once and runs on over there: the
        // announcer follows it and says how it ended (`announce`).
        if let (Some(announcer), Ok(answer)) = (&robot.announcer, &answer) {
            announcer.tool_called(name, args, answer);
        }
        return answer;
    }
    // The camera is mediad's, not robotd's: a frame does not wait on
    // (or redial) the robot's lane.
    if name == "robot.get_frame" {
        return crate::camera::grab(&robot.camera).map_err(|error| error.to_string());
    }
    match name {
        "robot.install_policy" => return install_policy(args, robot),
        "robot.policy_status" => return policy_status(args, robot),
        "robot.remove_policy" => return remove_policy(args, robot),
        _ => {}
    }
    // Everything below needs the robot, so this is where a lane that
    // died gets one attempt to come back.
    robot.redial();
    let control = &mut robot.lane.control;
    match name {
        "robot.sound" => {
            let tag = require_str(args, "tag")?;
            let tag: proto::SoundTag = serde_json::from_value(json!(tag))
                .map_err(|_| format!("unknown sound tag `{tag}`"))?;
            if tag == proto::SoundTag::Wheee {
                // A held ride makes no sense as a one-shot LLM tool.
                return Err("unknown sound tag `wheee`".to_string());
            }
            let result = intent(
                control,
                &proto::Call::RobotSound(proto::SoundParams { tag, hold: None }),
            )?;
            intent_outcome(result)
        }
        "robot.look" => {
            let params = proto::LookParams {
                x: number(args, "x").clamp(-MAX_LOOK_XY_M, MAX_LOOK_XY_M),
                y: number(args, "y").clamp(-MAX_LOOK_XY_M, MAX_LOOK_XY_M),
                z: number(args, "z").clamp(MIN_LOOK_Z_M, MAX_LOOK_Z_M),
                neck_pitch: 0.0,
            };
            let response = request(control, &proto::Call::RobotLook(params))?;
            if let Some(error) = &response.error {
                return Err(format!("robot refused look: {error}"));
            }
            let clamped = response
                .result_as::<proto::LookResult>()
                .map(|r| r.clamped)
                .unwrap_or(false);
            // `clamped` tells the agent the point is beyond the head's
            // reach — the gaze is the closest approximation, not a lock.
            Ok(json!({"done": true, "clamped": clamped}))
        }
        "robot.head" => {
            let params = proto::HeadParams {
                neck_pitch: 0.0,
                head_pitch: clamp(number(args, "pitch"), MAX_HEAD_PITCH_RAD),
                head_yaw: clamp(number(args, "yaw"), MAX_HEAD_YAW_RAD),
                head_roll: clamp(number(args, "roll"), MAX_HEAD_ROLL_RAD),
            };
            // Head is a persistent slot in robotd and deliberately not
            // deadmanned: one notification is enough.
            notify(control, &proto::Call::RobotHead(params))?;
            Ok(json!({"done": true}))
        }
        "robot.skill" => {
            let skill_name = require_str(args, "name")?;
            // Checked against what this robot answered `robot.skills`
            // with, not against a list compiled in: `proto::Skill` is a
            // plain name since daemon 0.14, so nothing on the wire
            // refuses a typo before it reaches the robot any more.
            if !robot.skills.iter().any(|skill| skill == skill_name) {
                return Err(format!("unknown skill `{skill_name}`"));
            }
            let skill: proto::Skill = skill_name.to_string();
            let result = intent(control, &proto::Call::RobotDo(proto::DoParams { skill }))?;
            intent_outcome(result)
        }
        "robot.move" => {
            let duration = args
                .get("duration_s")
                .and_then(Value::as_f64)
                .ok_or("duration_s is required")?
                .clamp(0.1, MAX_MOVE_DURATION_S);
            let params = trimmed(&robot.gait, move_params(args));
            // No heading hold here: the yaw it closes on is the cliff
            // guard's, and the guard belongs to the navigation daemon.
            let walked = timed_move(control, params, duration)?;
            // Cut short when somebody pressed stop (`body::halt`): the
            // agent should know the walk did not go the whole way.
            let walked = (walked * 10.0).round() / 10.0;
            if walked + 0.05 < duration {
                return Ok(json!({"done": true, "walked_s": walked, "stopped": true}));
            }
            Ok(json!({"done": true, "walked_s": duration}))
        }
        "robot.state" => {
            let health = request(control, &proto::Call::RobotHealth)?;
            let mode = request(control, &proto::Call::RobotMode)
                .ok()
                .and_then(|r| r.result_as::<proto::ModeResult>().ok())
                .map(|m| m.mode);
            let health: Value = health.result.unwrap_or(Value::Null);
            Ok(json!({
                "healthy": health.get("healthy"),
                "reason": health.get("reason"),
                "battery": health.get("battery"),
                "mode": mode,
            }))
        }
        other => Err(format!("unknown tool `{other}`")),
    }
}

/// `robot.install_policy`: check, download, `robot.setSkill`. Every
/// refusal is one of [`crate::install`]'s fixed lines.
fn install_policy(args: &Value, robot: &mut Robot) -> Result<Value, String> {
    let request = crate::install::Request::parse(args).map_err(str::to_string)?;
    install_with(request, robot)
}

/// [`install_policy`] once the request is checked: split out so a test
/// can point `url` at a loopback server.
fn install_with(request: crate::install::Request, robot: &mut Robot) -> Result<Value, String> {
    use crate::install::{self, ROBOT_UNAVAILABLE, SET_SKILL_REFUSED};
    // The robot first: no point fetching 20 MB for a duck that is not there.
    robot.redial();
    policies(&mut robot.lane.control)?;
    let (path, bytes) =
        install::fetch_to(&robot.install_dir, &request.name, &request.file, &request.url)
            .map_err(str::to_string)?;
    let before = settled(&mut robot.lane.control)?;
    let params = proto::SkillParams {
        name: request.name.clone(),
        path: Some(path.to_string_lossy().into_owned()),
        duration: Some(request.duration_s),
        chain: request.chain,
        ..Default::default()
    };
    let response = request_or(&mut robot.lane.control, &proto::Call::RobotSetSkill(params), ROBOT_UNAVAILABLE)?;
    let accepted = response.error.is_none()
        && response
            .result_as::<proto::IntentResult>()
            .is_ok_and(|result| result.accepted);
    if !accepted {
        tracing::info!(name = %request.name, "robotd refused the skill");
        return Err(SET_SKILL_REFUSED.to_string());
    }
    tracing::info!(name = %request.name, bytes, "skill installed, robot homing");
    robot.pending_install = Some(install::Pending::before(&request.name, &before));
    Ok(install::installed(&request.name, bytes))
}

/// `robot.policy_status`: `robot.policies`, read for one name.
fn policy_status(args: &Value, robot: &mut Robot) -> Result<Value, String> {
    let name = args
        .get("name")
        .and_then(Value::as_str)
        .ok_or(crate::install::SLOT_REFUSED)?;
    crate::install::validate_name(name)?;
    robot.redial();
    let answer = policies(&mut robot.lane.control)?;
    let (state, detail) = crate::install::status(&answer, name, robot.pending_install.as_mut());
    if state == "homed" && robot.pending_install.as_ref().is_some_and(|p| p.name == name) {
        robot.pending_install = None;
        // So `robot.skill` takes the new name from now on.
        robot.skills = skills_of(&mut robot.lane.control);
    }
    let mut data = json!({"name": name, "state": state});
    if let Some(detail) = detail {
        data["detail"] = json!(detail);
    }
    Ok(data)
}

/// `robot.remove_policy`: robotd's `robot.removeSkill`, then the files.
fn remove_policy(args: &Value, robot: &mut Robot) -> Result<Value, String> {
    use crate::install::{REMOVE_REFUSED, ROBOT_UNAVAILABLE, SLOT_REFUSED};
    let name = args.get("name").and_then(Value::as_str).ok_or(SLOT_REFUSED)?;
    crate::install::validate_name(name)?;
    robot.redial();
    settled(&mut robot.lane.control)?;
    let call = proto::Call::RobotRemoveSkill(proto::SkillNameParams { name: name.to_string() });
    let response = request_or(&mut robot.lane.control, &call, ROBOT_UNAVAILABLE)?;
    let result = response
        .error
        .is_none()
        .then(|| response.result_as::<proto::IntentResult>().ok())
        .flatten()
        .filter(|result| result.accepted)
        .ok_or(REMOVE_REFUSED)?;
    // `already` (nothing configured by that name) carries a reason.
    let removed = result.reason.is_none();
    crate::install::remove_files(&robot.install_dir, name);
    if robot.pending_install.as_ref().is_some_and(|p| p.name == name) {
        robot.pending_install = None;
    }
    robot.skills = skills_of(&mut robot.lane.control);
    tracing::info!(name, removed, "skill removed");
    Ok(json!({"name": name, "removed": removed}))
}

/// `robot.policies`' answer, or `install: robot unavailable`.
fn policies(control: &mut Option<Control>) -> Result<Value, String> {
    let unavailable = crate::install::ROBOT_UNAVAILABLE;
    let response = request_or(control, &proto::Call::RobotPolicies, unavailable)?;
    if response.error.is_some() {
        return Err(unavailable.to_string());
    }
    response.result.ok_or_else(|| unavailable.to_string())
}

/// `robot.policies` once the robot is not homing, or after
/// [`crate::install::SETTLE_TIMEOUT`] regardless: robotd drops a skill
/// change that arrives while another change's reload is in flight (an
/// install right after a remove, measured on the sim 2026-10-21), and
/// `homed: false` is how that reload shows.
fn settled(control: &mut Option<Control>) -> Result<Value, String> {
    let deadline = std::time::Instant::now() + crate::install::SETTLE_TIMEOUT;
    loop {
        let answer = policies(control)?;
        let homing = answer.get("homed").and_then(Value::as_bool) == Some(false);
        if !homing || std::time::Instant::now() >= deadline {
            if homing {
                tracing::info!("the robot is still homing; changing the skill table anyway");
            }
            return Ok(answer);
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
}

/// [`request`], with every transport failure reported as `error`.
fn request_or(
    control: &mut Option<Control>,
    call: &proto::Call,
    error: &str,
) -> Result<proto::Response, String> {
    request(control, call).map_err(|_| error.to_string())
}

/// Start or stop the "map everything" job.
fn intent(
    control: &mut Option<Control>,
    call: &proto::Call,
) -> Result<proto::IntentResult, String> {
    let robot = with_robot(control)?;
    robot.intent(call).map_err(|e| {
        *control = None;
        format!("robot lost: {e}")
    })
}

fn intent_outcome(result: proto::IntentResult) -> Result<Value, String> {
    if result.accepted {
        Ok(json!({"done": true}))
    } else {
        Err(result
            .reason
            .unwrap_or_else(|| "refused by the robot".to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unknown_tool_and_unsupported_are_soft_errors() {
        let mut robot = Robot::detached();
        assert_eq!(
            execute("robot.fly", &json!({}), &mut robot),
            Err("unknown tool `robot.fly`".to_string())
        );
        assert_eq!(
            execute("robot.get_frame", &json!({}), &mut robot),
            Err("unsupported".to_string())
        );
    }

    #[test]
    fn get_frame_goes_to_the_camera_socket_when_one_is_configured() {
        let dir = tempfile::tempdir().unwrap();
        let mut robot = Robot::detached();
        robot.camera = dir.path().join("media.sock").to_str().unwrap().to_string();
        let error = execute("robot.get_frame", &json!({}), &mut robot).unwrap_err();
        assert!(error.starts_with("camera error: "), "{error}");
        let catalog = catalog(&robot);
        let tool = catalog
            .as_array()
            .unwrap()
            .iter()
            .find(|tool| tool["name"] == "robot.get_frame")
            .unwrap();
        assert!(tool["description"].as_str().unwrap().contains("JPEG"));
    }

    #[test]
    fn robot_tools_without_a_robot_say_so() {
        let mut robot = Robot::detached();
        assert_eq!(
            execute("robot.sound", &json!({"tag": "chirp"}), &mut robot),
            Err("robot unreachable".to_string())
        );
        assert_eq!(
            execute("robot.move", &json!({"duration_s": 1.0}), &mut robot),
            Err("robot unreachable".to_string())
        );
    }

    #[test]
    fn bad_arguments_are_rejected_before_touching_the_robot() {
        let mut robot = Robot::detached();
        assert_eq!(
            execute("robot.sound", &json!({"tag": "explosion"}), &mut robot),
            Err("unknown sound tag `explosion`".to_string())
        );
        assert_eq!(
            execute("robot.sound", &json!({"tag": "wheee"}), &mut robot),
            Err("unknown sound tag `wheee`".to_string())
        );
        assert_eq!(
            execute("robot.skill", &json!({"name": "backflip"}), &mut robot),
            Err("unknown skill `backflip`".to_string())
        );
        assert_eq!(
            execute("robot.move", &json!({}), &mut robot),
            Err("duration_s is required".to_string())
        );
    }
    // The map, the places and the steps are the navigation daemon's:
    // their tests moved to `quack-nav` with the code (2026-09-22).

    /// One fake robotd that answers a single `robot.skills` request with
    /// `answer`, so the two skew cases below differ only in that answer.
    fn skills_against(answer: fn(Option<proto::Id>) -> proto::Response) -> Vec<String> {
        use std::io::{BufRead, BufReader, Write};
        use std::os::unix::net::UnixListener;

        let dir = tempfile::tempdir().unwrap();
        let socket = dir.path().join("robotd.sock");
        let listener = UnixListener::bind(&socket).unwrap();
        let server = std::thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            let mut reader = BufReader::new(stream.try_clone().unwrap());
            let mut line = String::new();
            reader.read_line(&mut line).unwrap();
            let request: proto::Request = serde_json::from_str(&line).unwrap();
            assert_eq!(request.method, "robot.skills");
            let mut out = serde_json::to_vec(&answer(request.id)).unwrap();
            out.push(b'\n');
            let mut writer = stream;
            writer.write_all(&out).unwrap();
            writer.flush().unwrap();
        });

        let mut control = Some(Control::connect(socket.to_str().unwrap()).unwrap());
        let skills = skills_of(&mut control);
        server.join().unwrap();
        skills
    }

    #[test]
    fn the_skills_are_the_robots_own_table_plus_its_built_ins() {
        let skills = skills_against(|id| {
            proto::Response::ok(
                id,
                &proto::SkillsResult {
                    skills: vec![proto::SkillParams {
                        name: "bow".to_string(),
                        ..Default::default()
                    }],
                    built_in: vec!["sit_toggle".to_string()],
                },
            )
        });
        // A robot with a configured `bow` and nothing else has a bow and a
        // sit toggle — not the five this satellite used to assume.
        assert_eq!(skills, vec!["bow".to_string(), "sit_toggle".to_string()]);
    }

    #[test]
    fn a_daemon_that_does_not_know_robot_skills_leaves_the_stock_five() {
        let skills = skills_against(|id| {
            proto::Response::err(
                id,
                proto::Error::new(-32601, "method not found: robot.skills"),
            )
        });
        assert_eq!(skills, stock_skills());
    }

    #[test]
    fn a_skill_the_robot_never_listed_is_refused_before_the_wire() {
        let mut robot = Robot::detached();
        robot.skills = vec!["bow".to_string()];
        // `proto::Skill` is a plain string since daemon 0.14, so this is
        // the only place left that can tell a typo from a skill.
        assert_eq!(
            execute("robot.skill", &json!({"name": "roulade"}), &mut robot),
            Err("unknown skill `roulade`".to_string())
        );
        // One the robot did list gets as far as the missing lane.
        assert_eq!(
            execute("robot.skill", &json!({"name": "bow"}), &mut robot),
            Err("robot unreachable".to_string())
        );
        // And the agent is told the same list the executor enforces.
        let catalog = catalog(&robot);
        let skill = catalog
            .as_array()
            .unwrap()
            .iter()
            .find(|tool| tool["name"] == "robot.skill")
            .unwrap();
        assert_eq!(skill["parameters"]["properties"]["name"]["enum"], json!(["bow"]));
    }

    #[test]
    fn a_lane_that_died_is_dialled_again_with_the_list_the_robot_has_now() {
        use std::io::{BufRead, BufReader, Write};
        use std::os::unix::net::{UnixListener, UnixStream};

        fn read_request(reader: &mut BufReader<UnixStream>) -> proto::Request {
            let mut line = String::new();
            reader.read_line(&mut line).unwrap();
            serde_json::from_str(&line).unwrap()
        }
        fn answer(stream: &mut UnixStream, response: &proto::Response) {
            let mut out = serde_json::to_vec(response).unwrap();
            out.push(b'\n');
            stream.write_all(&out).unwrap();
            stream.flush().unwrap();
        }
        fn listing(name: &str) -> proto::SkillsResult {
            proto::SkillsResult {
                skills: vec![proto::SkillParams {
                    name: name.to_string(),
                    ..Default::default()
                }],
                built_in: Vec::new(),
            }
        }

        let dir = tempfile::tempdir().unwrap();
        let socket = dir.path().join("robotd.sock");
        let listener = UnixListener::bind(&socket).unwrap();
        let server = std::thread::spawn(move || {
            // First life: it can bow. Then it goes away mid-session,
            // which is what an update doing `systemctl restart robotd`
            // looks like from here.
            let (mut stream, _) = listener.accept().unwrap();
            let mut reader = BufReader::new(stream.try_clone().unwrap());
            let request = read_request(&mut reader);
            assert_eq!(request.method, "robot.skills");
            answer(&mut stream, &proto::Response::ok(request.id, &listing("bow")));
            drop(reader);
            drop(stream);

            // Second life, same socket, another skill table — and the
            // `robot.do` that the satellite could only send because it
            // dialled again.
            let (mut stream, _) = listener.accept().unwrap();
            let mut reader = BufReader::new(stream.try_clone().unwrap());
            let request = read_request(&mut reader);
            assert_eq!(request.method, "robot.skills");
            answer(
                &mut stream,
                &proto::Response::ok(request.id, &listing("roulade")),
            );
            let request = read_request(&mut reader);
            assert_eq!(request.method, "robot.do");
            answer(
                &mut stream,
                &proto::Response::ok(request.id, &proto::IntentResult::accepted()),
            );
        });

        let config: Config = toml::from_str(&format!(
            "backend = \"direct\"\nrobotd_socket = \"{}\"\n[nav]\nenabled = false\n",
            socket.display()
        ))
        .unwrap();
        let mut robot = Robot::connect(&config);
        assert_eq!(robot.skills, vec!["bow".to_string()]);

        // The robot is gone: the call that finds out loses the lane.
        assert!(execute("robot.skill", &json!({"name": "bow"}), &mut robot).is_err());
        assert!(robot.lane.control.is_none());

        // The next one dials again and asks what this robot can do now,
        // so a skill that did not exist a second ago goes through.
        assert_eq!(
            execute("robot.skill", &json!({"name": "roulade"}), &mut robot),
            Ok(json!({"done": true}))
        );
        assert_eq!(robot.skills, vec!["roulade".to_string()]);
        // And the one it used to have is refused, rather than sent to a
        // robot that would not know it.
        assert_eq!(
            execute("robot.skill", &json!({"name": "bow"}), &mut robot),
            Err("unknown skill `bow`".to_string())
        );
        server.join().unwrap();
    }

    /// A fake robotd on one connection: every request goes to `answer`,
    /// which returns the `result` to send, and is recorded.
    fn fake_robotd(
        answer: impl Fn(&proto::Request) -> Value + Send + 'static,
    ) -> (tempfile::TempDir, Robot, std::thread::JoinHandle<Vec<proto::Request>>) {
        use std::io::{BufRead, BufReader, Write};
        use std::os::unix::net::UnixListener;

        let dir = tempfile::tempdir().unwrap();
        let socket = dir.path().join("robotd.sock");
        let listener = UnixListener::bind(&socket).unwrap();
        let server = std::thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            let mut reader = BufReader::new(stream.try_clone().unwrap());
            let mut writer = stream;
            let mut seen = Vec::new();
            loop {
                let mut line = String::new();
                if reader.read_line(&mut line).unwrap_or(0) == 0 {
                    return seen;
                }
                let request: proto::Request = serde_json::from_str(&line).unwrap();
                let result = answer(&request);
                let mut out = serde_json::to_vec(
                    &json!({"jsonrpc": "2.0", "id": request.id, "result": result}),
                )
                .unwrap();
                out.push(b'\n');
                writer.write_all(&out).unwrap();
                writer.flush().unwrap();
                seen.push(request);
            }
        });
        let mut robot = Robot::detached();
        robot.lane.control = Some(Control::connect(socket.to_str().unwrap()).unwrap());
        robot.install_dir = dir.path().join("policies");
        (dir, robot, server)
    }

    fn request_for(name: &str, url: String) -> crate::install::Request {
        let mut request =
            crate::install::Request::parse(&json!({"source": "org/repo", "name": name})).unwrap();
        request.url = url;
        request
    }

    #[test]
    fn install_refusals_need_neither_the_network_nor_the_robot() {
        let mut robot = Robot::detached();
        for (args, error) in [
            (json!({"source": "org/repo", "name": "walk"}), "install: slot refused"),
            (json!({"source": "org/repo", "name": "roulade"}), "install: slot refused"),
            (json!({"source": "org/repo", "name": "bow", "file": "../x.onnx"}), "install: slot refused"),
            (json!({"source": "org/repo", "name": "bow", "slot": "walk"}), "install: slot refused"),
            (json!({"source": "file:///etc/passwd", "name": "bow"}), "install: download failed"),
            (json!({"source": "https://evil.example/x.onnx", "name": "bow"}), "install: download failed"),
            (json!({"source": "org/repo", "name": "bow"}), "install: robot unavailable"),
        ] {
            assert_eq!(execute("robot.install_policy", &args, &mut robot), Err(error.to_string()), "{args}");
        }
        assert_eq!(
            execute("robot.policy_status", &json!({"name": "bow"}), &mut robot),
            Err("install: robot unavailable".to_string())
        );
        assert_eq!(
            execute("robot.policy_status", &json!({"name": "../bow"}), &mut robot),
            Err("install: slot refused".to_string())
        );
        assert_eq!(
            execute("robot.remove_policy", &json!({"name": "kick_left"}), &mut robot),
            Err("install: slot refused".to_string())
        );
    }

    #[test]
    fn an_install_lands_the_file_registers_it_and_is_followed_to_homed() {
        use crate::install::tests::{ok_head, serve};
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;

        // robotd as the sim behaves: homed until the reload starts, then
        // homing without the new name, then listed and homed.
        let polls = Arc::new(AtomicUsize::new(0));
        let counter = polls.clone();
        let (dir, mut robot, server) = fake_robotd(move |request| match request.method.as_str() {
            "robot.policies" => match counter.fetch_add(1, Ordering::SeqCst) {
                0 | 1 => json!({"mode": "walk", "enabled": true, "slots": [], "skills": ["roulade"], "homed": true}),
                2 => json!({"mode": "walk", "enabled": true, "slots": [], "skills": ["roulade"], "homed": false}),
                _ => json!({"mode": "walk", "enabled": true, "slots": [], "skills": ["roulade", "flamingo_cycle"], "homed": true}),
            },
            "robot.setSkill" => json!({"accepted": true}),
            "robot.skills" => json!({"skills": [{"name": "flamingo_cycle"}], "built_in": ["sit_toggle"]}),
            other => panic!("unexpected {other}"),
        });
        let body = vec![0x08, 0x08, 0x12, 0x07];
        let (url, http) = serve(ok_head(body.len()), body.clone());
        let answer = install_with(request_for("flamingo_cycle", url), &mut robot).unwrap();
        http.join().unwrap();
        assert_eq!(answer, json!({"name": "flamingo_cycle", "bytes": 4, "stage": "installing"}));
        let landed = dir.path().join("policies/flamingo_cycle/policy.onnx");
        assert_eq!(std::fs::read(&landed).unwrap(), body);

        let status = |robot: &mut Robot| {
            execute("robot.policy_status", &json!({"name": "flamingo_cycle"}), robot).unwrap()["state"].clone()
        };
        assert_eq!(status(&mut robot), "homing");
        assert_eq!(status(&mut robot), "homed");
        assert!(robot.pending_install.is_none());
        assert_eq!(robot.skills, vec!["flamingo_cycle".to_string(), "sit_toggle".to_string()]);
        assert_eq!(
            execute("robot.policy_status", &json!({"name": "other_one"}), &mut robot).unwrap(),
            json!({"name": "other_one", "state": "missing"})
        );
        drop(robot);
        let seen = server.join().unwrap();
        let set = seen.iter().find(|r| r.method == "robot.setSkill").unwrap();
        // Exactly what robotd's `SkillParams` (deny_unknown_fields) takes:
        // no slot, no chain unless asked, the Playground's 3 s.
        assert_eq!(
            set.params.clone().unwrap(),
            json!({"name": "flamingo_cycle", "path": landed.to_str().unwrap(), "duration": 3.0})
        );
    }

    #[test]
    fn a_refusal_from_robotd_is_one_line_and_a_bad_file_never_reaches_it() {
        use crate::install::tests::{ok_head, serve};

        let (_dir, mut robot, server) = fake_robotd(|request| match request.method.as_str() {
            "robot.policies" => json!({"skills": [], "homed": true}),
            "robot.setSkill" => json!({"accepted": false, "reason": "no such file: /secret/path.onnx"}),
            other => panic!("unexpected {other}"),
        });
        let (url, http) = serve(ok_head(4), vec![0x08, 1, 2, 3]);
        assert_eq!(
            install_with(request_for("bow", url), &mut robot),
            Err("install: setSkill refused".to_string())
        );
        http.join().unwrap();
        assert!(robot.pending_install.is_none());

        let (url, http) = serve(ok_head(15), b"<!doctype html>".to_vec());
        assert_eq!(install_with(request_for("bow", url), &mut robot), Err("install: not an onnx file".to_string()));
        http.join().unwrap();
        let (url, http) = serve("HTTP/1.1 200 OK\r\nContent-Length: 30000000", Vec::new());
        assert_eq!(install_with(request_for("bow", url), &mut robot), Err("install: too large".to_string()));
        http.join().unwrap();
        drop(robot);
        let seen = server.join().unwrap();
        assert_eq!(seen.iter().filter(|r| r.method == "robot.setSkill").count(), 1);
    }

    #[test]
    fn remove_policy_asks_robotd_and_drops_the_files() {
        let (dir, mut robot, server) = fake_robotd(|request| match request.method.as_str() {
            "robot.policies" => json!({"skills": ["flamingo_cycle"], "homed": true}),
            "robot.removeSkill" => json!({"accepted": true}),
            "robot.skills" => json!({"skills": [{"name": "roulade"}]}),
            other => panic!("unexpected {other}"),
        });
        let folder = dir.path().join("policies/flamingo_cycle");
        std::fs::create_dir_all(&folder).unwrap();
        std::fs::write(folder.join("policy.onnx"), [8u8]).unwrap();
        assert_eq!(
            execute("robot.remove_policy", &json!({"name": "flamingo_cycle"}), &mut robot).unwrap(),
            json!({"name": "flamingo_cycle", "removed": true})
        );
        assert!(!folder.exists());
        assert_eq!(robot.skills, vec!["roulade".to_string()]);
        drop(robot);
        let seen = server.join().unwrap();
        let remove = seen.iter().find(|r| r.method == "robot.removeSkill").unwrap();
        assert_eq!(remove.params.clone().unwrap(), json!({"name": "flamingo_cycle"}));
    }

    #[test]
    fn catalog_matches_the_executor_allowlist() {
        // Without a navigation daemon the satellite announces its own
        // tools and nothing else; with one, the daemon's catalog is
        // spliced in (see `crate::nav_client`).
        let catalog = catalog(&Robot::detached());
        let names: Vec<&str> = catalog
            .as_array()
            .unwrap()
            .iter()
            .map(|t| t["name"].as_str().unwrap())
            .collect();
        assert_eq!(
            names,
            [
                "robot.sound",
                "robot.look",
                "robot.head",
                "robot.skill",
                "robot.move",
                "robot.state",
                "robot.get_frame",
                "robot.install_policy",
                "robot.policy_status",
                "robot.remove_policy",
            ]
        );
        let mut robot = Robot::detached();
        for name in names {
            let err = execute(name, &json!({}), &mut robot).err().unwrap_or_default();
            assert!(!err.starts_with("unknown tool"), "{name}: {err}");
        }
        assert!(execute("robot.go_to", &json!({}), &mut robot).unwrap_err().starts_with("unknown tool"));

    }
}

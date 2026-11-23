//! Inline clips only. Worker-owned URLs must never reach a local decoder.
use super::{image_url, StatusCode, Value, CHAT, COUNT_TOKENS, MESSAGES, RESPONSES};
use base64::{engine::general_purpose::STANDARD, Engine as _};
use std::{
    fs::{self, File},
    io::{self, Read},
    path::Path,
    process::{Command, Stdio},
    thread,
    time::{Duration, Instant},
};

const SOURCE_BYTES: usize = 200_000_000;
const OUTPUT_BYTES: usize = 32 * 1024 * 1024;
const RGB_BYTES: f64 = 128.0 * 1024.0 * 1024.0;
const MAX_PIXELS: u64 = 4096 * 4096;
const BAD: &str = "Invalid or unsupported inline video.";
type Failure = (StatusCode, &'static str, &'static str);
fn invalid(reason: &'static str) -> Failure {
    (StatusCode::BAD_REQUEST, "invalid_inline_video", reason)
}
fn unavailable(_: io::Error) -> Failure {
    (
        StatusCode::SERVICE_UNAVAILABLE,
        "video_processing_unavailable",
        "Inline video processing is unavailable.",
    )
}

pub(super) struct Clip {
    block: String,
    field: &'static str,
}

fn data_uri(value: &str) -> Result<(&str, &str), &'static str> {
    let (mime, data) = value
        .strip_prefix("data:")
        .and_then(|s| s.split_once(";base64,"))
        .ok_or(BAD)?;
    Ok((mime, data))
}

fn demuxer(mime: &str) -> Result<&'static str, &'static str> {
    match mime {
        "video/mp4" | "video/quicktime" => Ok("mov"),
        "video/x-matroska" | "video/webm" => Ok("matroska"),
        _ => Err("Inline video must be MP4, MOV, MKV or WebM."),
    }
}

fn inspect_part(
    path: &str,
    part: &Value,
    user: bool,
) -> Result<Option<&'static str>, &'static str> {
    let kind = part["type"].as_str();
    let field = match (path, kind) {
        (CHAT, Some("video_url")) => "/video_url/url",
        (RESPONSES, Some("input_video" | "video_url")) if user => {
            if part["video_url"].is_object() {
                "/video_url/url"
            } else {
                "/video_url"
            }
        }
        (MESSAGES | COUNT_TOKENS, Some("video")) if user => match part["source"]["type"].as_str() {
            Some("url") => "/source/url",
            Some("base64") => "/source/data",
            _ => return Err(BAD),
        },
        (_, Some("input_video" | "video_url" | "video")) => {
            return Err("Invalid video block for this protocol or message role.")
        }
        (_, Some("image_url" | "input_image" | "image")) => {
            let url = image_url(part)
                .or_else(|| part["source"]["url"].as_str())
                .unwrap_or("");
            let mime = part["source"]["media_type"].as_str().unwrap_or("");
            if url
                .get(..11)
                .is_some_and(|x| x.eq_ignore_ascii_case("data:video/"))
                || mime
                    .get(..6)
                    .is_some_and(|x| x.eq_ignore_ascii_case("video/"))
            {
                return Err("Use a video block for video input.");
            }
            return Ok(None);
        }
        _ => return Ok(None),
    };
    let value = part.pointer(field).and_then(Value::as_str).ok_or(BAD)?;
    let (mime, data) = if field == "/source/data" {
        (part["source"]["media_type"].as_str().ok_or(BAD)?, value)
    } else if value.starts_with("data:") {
        data_uri(value)?
    } else if value.starts_with("https://")
        || value.starts_with("http://")
        || value.starts_with("file:///")
    {
        return Ok(None); // domain / local-path policy is enforced by the backend
    } else {
        return Err(BAD);
    };
    demuxer(mime)?;
    if data.is_empty() || data.len() > SOURCE_BYTES.div_ceil(3) * 4 {
        return Err("Empty video or video exceeds the source byte limit.");
    }
    Ok(Some(field))
}

// Only protocol message content and Anthropic tool_result.content contain video.
// Tool arguments, metadata, (possibly JSON-shaped) text and tool schemas stay opaque.
pub(super) fn locate(path: &str, root: &Value) -> Result<Option<Clip>, &'static str> {
    let mut clip = None;
    let mut inspect =
        |part: &Value, user: bool, location: &dyn Fn() -> String| -> Result<(), &'static str> {
            if let Some(field) = inspect_part(path, part, user)? {
                if clip.is_some() {
                    return Err("At most one inline video per request.");
                }
                clip = Some(Clip {
                    block: location(),
                    field,
                });
            }
            Ok(())
        };
    for (i, msg) in root.as_array().into_iter().flatten().enumerate() {
        if !matches!(msg["type"].as_str(), None | Some("message")) {
            continue;
        }
        let user = msg["role"] == "user";
        for (j, part) in msg["content"].as_array().into_iter().flatten().enumerate() {
            inspect(part, user, &|| format!("/{i}/content/{j}"))?;
            if matches!(path, MESSAGES | COUNT_TOKENS) && part["type"] == "tool_result" {
                for (k, sub) in part["content"].as_array().into_iter().flatten().enumerate() {
                    inspect(sub, user, &|| format!("/{i}/content/{j}/content/{k}"))?;
                }
            }
        }
    }
    Ok(clip)
}

impl Clip {
    pub(super) fn prepare(self, root: &mut Value) -> Result<(), Failure> {
        let part = root.pointer_mut(&self.block).expect("located video block");
        let value = part.pointer(self.field).and_then(Value::as_str).unwrap();
        let (mime, data) = if self.field == "/source/data" {
            (part["source"]["media_type"].as_str().unwrap(), value)
        } else {
            data_uri(value).map_err(invalid)?
        };
        let raw = STANDARD
            .decode(data)
            .map_err(|_| invalid("Invalid video base64."))?;
        if raw.len() > SOURCE_BYTES {
            return Err(invalid("Video exceeds the source byte limit."));
        }
        let Some(bytes) = normalize(raw, mime)? else {
            return Ok(());
        };
        let encoded = STANDARD.encode(bytes);
        let value = if self.field == "/source/data" {
            part["source"]["media_type"] = Value::from("video/mp4");
            encoded
        } else {
            format!("data:video/mp4;base64,{encoded}")
        };
        *part.pointer_mut(self.field).unwrap() = Value::String(value);
        part.as_object_mut().unwrap().remove("uuid");
        Ok(())
    }
}

// Only fixed demuxers and private temporary files. No shell or client filename.
fn input_args(cmd: &mut Command, format: &str, src: &Path) {
    cmd.args([
        "-v",
        "error",
        "-max_alloc",
        "134217728",
        "-protocol_whitelist",
        "file",
        "-f",
        format,
        "-threads",
        "1",
        "-max_pixels",
        "16777216",
    ]);
    if format == "mov" {
        cmd.args(["-enable_drefs", "0", "-use_absolute_path", "0"]);
    }
    cmd.arg("-i").arg(src);
}

// The blocking job owns the permit through kill + reap, even on disconnect.
fn run(cmd: &mut Command, stdout: Stdio, deadline: Instant) -> Result<(), Failure> {
    #[cfg(target_os = "linux")]
    {
        use std::os::unix::process::CommandExt;
        // SAFETY: only async-signal-safe setrlimit calls between fork and exec.
        unsafe {
            cmd.pre_exec(|| {
                for (resource, cap) in [
                    (libc::RLIMIT_AS, 1536 << 20),
                    (libc::RLIMIT_FSIZE, (OUTPUT_BYTES + 65536) as libc::rlim_t),
                ] {
                    let mut limit = libc::rlimit {
                        rlim_cur: 0,
                        rlim_max: 0,
                    };
                    if libc::getrlimit(resource, &mut limit) != 0 {
                        return Err(io::Error::last_os_error());
                    }
                    limit.rlim_cur = limit.rlim_cur.min(cap);
                    if libc::setrlimit(resource, &limit) != 0 {
                        return Err(io::Error::last_os_error());
                    }
                }
                Ok(())
            });
        }
    }
    let mut child = cmd
        .env("OPENBLAS_NUM_THREADS", "1")
        .env("OMP_NUM_THREADS", "1")
        .stdin(Stdio::null())
        .stdout(stdout)
        .stderr(Stdio::null())
        .spawn()
        .map_err(unavailable)?;
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                return if status.success() {
                    Ok(())
                } else {
                    Err(invalid(BAD))
                }
            }
            Ok(None) if Instant::now() < deadline => thread::sleep(Duration::from_millis(20)),
            result => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(result
                    .err()
                    .map(unavailable)
                    .unwrap_or_else(|| invalid("Video processing exceeded the time limit.")));
            }
        }
    }
}

struct Stream {
    w: u32,
    h: u32,
    duration: f64,
    frames: u64,
    fps: f64,
    h264: bool,
}
fn probe(src: &Path, format: &str, report: &Path, deadline: Instant) -> Result<Stream, Failure> {
    let mut cmd = Command::new("ffprobe");
    cmd.args([
        "-select_streams",
        "V:0",
        "-show_entries",
        "stream=width,height,codec_name,nb_frames,duration,avg_frame_rate:format=duration",
        "-of",
        "json",
    ]);
    input_args(&mut cmd, format, src);
    run(
        &mut cmd,
        File::create(report).map_err(unavailable)?.into(),
        deadline,
    )?;
    let metadata: Value =
        serde_json::from_reader(File::open(report).map_err(unavailable)?.take(8192))
            .map_err(|_| invalid(BAD))?;
    let s = &metadata["streams"][0];
    let number = |v: &Value| v.as_str()?.parse::<f64>().ok();
    let duration = number(&s["duration"])
        .or_else(|| number(&metadata["format"]["duration"]))
        .unwrap_or(0.0);
    let fps = s["avg_frame_rate"]
        .as_str()
        .and_then(|r| r.split_once('/'))
        .and_then(|(a, b)| Some(a.parse::<f64>().ok()? / b.parse::<f64>().ok()?))
        .unwrap_or(0.0);
    let w = s["width"].as_u64().unwrap_or(0);
    let h = s["height"].as_u64().unwrap_or(0);
    if w == 0
        || h == 0
        || w.max(h) > 8192
        || w * h > MAX_PIXELS
        || !(duration > 0.0 && duration <= 600.0 && fps > 0.0 && fps <= 240.0)
    {
        return Err(invalid("Invalid video dimensions, duration or frame rate."));
    }
    Ok(Stream {
        w: w as u32,
        h: h as u32,
        duration,
        fps,
        frames: s["nb_frames"]
            .as_str()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0),
        h264: s["codec_name"] == "h264",
    })
}

fn normalize(raw: Vec<u8>, mime: &str) -> Result<Option<Vec<u8>>, Failure> {
    let dir = tempfile::tempdir().map_err(unavailable)?;
    let (src, dst, report) = (
        dir.path().join("source"),
        dir.path().join("out.mp4"),
        dir.path().join("probe"),
    );
    let len = raw.len();
    fs::write(&src, raw).map_err(unavailable)?;
    let deadline = Instant::now() + Duration::from_secs(45);
    let format = demuxer(mime).map_err(invalid)?;
    let v = probe(&src, format, &report, deadline)?;
    let (w, h) = (f64::from(v.w), f64::from(v.h));
    let mut cmd = Command::new("ffmpeg");
    cmd.args([
        "-nostdin",
        "-xerror",
        "-abort_on",
        "empty_output",
        "-cpucount",
        "1",
        "-filter_threads",
        "1",
    ]);
    input_args(&mut cmd, format, &src);
    cmd.args(["-map", "0:V:0", "-an", "-sn", "-dn"]);
    if mime == "video/mp4"
        && v.h264
        && len <= OUTPUT_BYTES
        && (1..=96).contains(&v.frames)
        && v.fps <= 8.0
        && w.max(h) <= 1280.0
        && w.min(h) <= 720.0
        && w * h * 3.0 * v.frames as f64 <= RGB_BYTES
    {
        // A plausible header isn't sufficient: reject broken payloads on the fast path too.
        cmd.arg("-progress").arg(&report).args([
            "-nostats",
            "-vsync",
            "0",
            "-frames:v",
            "97",
            "-threads",
            "1",
            "-f",
            "null",
            "-",
        ]);
        run(&mut cmd, Stdio::null(), deadline)?;
        let progress = fs::read_to_string(report).map_err(unavailable)?;
        let decoded = progress
            .lines()
            .rev()
            .find_map(|l| l.strip_prefix("frame="))
            .and_then(|s| s.trim().parse::<u64>().ok());
        if decoded != Some(v.frames) {
            return Err(invalid("Incomplete or inconsistent MP4 frame count."));
        }
        return Ok(None);
    }
    // Even sampling over the whole timeline. Fit both frame count and total decoded RGB.
    let n = (v.duration * v.fps.min(8.0)).floor().clamp(1.0, 96.0);
    let scale = (1280.0 / w.max(h))
        .min(720.0 / w.min(h))
        .min((RGB_BYTES / (3.0 * w * h * n)).sqrt())
        .min(1.0);
    let rate = n / v.duration;
    let vf = format!("setpts=PTS-STARTPTS,fps={rate:.9},scale='max(2,trunc(iw*{scale:.12}/2)*2)':'max(2,trunc(ih*{scale:.12}/2)*2)'");
    cmd.args([
        "-vf",
        &vf,
        "-frames:v",
        &(n as u32 + 1).to_string(),
        "-map_metadata",
        "-1",
        "-map_chapters",
        "-1",
        "-c:v",
        "libx264",
        "-threads",
        "1",
        "-preset",
        "veryfast",
        "-crf",
        "24",
        "-pix_fmt",
        "yuv420p",
        "-fs",
        &OUTPUT_BYTES.to_string(),
        "-movflags",
        "+faststart",
    ])
    .arg(&dst);
    run(&mut cmd, Stdio::null(), deadline)?;
    if fs::metadata(&dst).map_err(unavailable)?.len() > OUTPUT_BYTES as u64 {
        return Err(invalid("Video output exceeds the byte budget."));
    }
    let out = probe(&dst, "mov", &report, deadline)?;
    if out.frames != n as u64
        || !out.h264
        || f64::from(out.w) * f64::from(out.h) * 3.0 * out.frames as f64 > RGB_BYTES
        || (out.duration - v.duration).abs() > 0.005_f64.max(0.01 / rate)
    {
        return Err(invalid(
            "Could not fit the complete video within the processing budget.",
        ));
    }
    Ok(Some(fs::read(dst).map_err(unavailable)?))
}

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::env;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

/// SDK whose Metal compiler builds for the platform cargo is targeting.
fn sdk() -> &'static str {
    match (
        env::var("CARGO_CFG_TARGET_OS").as_deref(),
        env::var("CARGO_CFG_TARGET_ABI").as_deref(),
    ) {
        (Ok("ios"), Ok("sim")) => "iphonesimulator",
        (Ok("ios"), _) => "iphoneos",
        _ => "macosx",
    }
}

const SHADER_FILES: &[&str] = &[
    "projection.metal",
    "ops.metal",
    "vision.metal",
    "dspark.metal",
    "sampling.metal",
    "ane.metal",
    "mpp.metal",
];

const TOOLCHAIN_HINT: &str = "install Xcode (or the Command Line Tools: `xcode-select --install`) \
     and its Metal toolchain (`xcodebuild -downloadComponent MetalToolchain`); \
     DEVELOPER_DIR and SDKROOT select which ones xcrun uses";

fn xcrun() -> Command {
    let mut command = Command::new("xcrun");
    command.args(["-sdk", sdk()]);
    command
}

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let shader_dir = Path::new("../shaders");
    let record_sources = env::var("PROFILE").is_ok_and(|profile| profile == "debug");

    println!("cargo:rerun-if-changed=../shaders/");
    println!("cargo:rerun-if-env-changed=DEVELOPER_DIR");
    println!("cargo:rerun-if-env-changed=SDKROOT");
    println!("cargo:rerun-if-env-changed=IPHONEOS_DEPLOYMENT_TARGET");
    println!("cargo:rerun-if-env-changed=MACOSX_DEPLOYMENT_TARGET");
    // Target the deployment OS (Xcode exports it for apps, .cargo/config.toml for cargo builds);
    // without it the metallib targets the SDK's OS.
    let min_os = match sdk() {
        "iphoneos" => env::var("IPHONEOS_DEPLOYMENT_TARGET")
            .ok()
            .map(|v| format!("-mios-version-min={v}")),
        "macosx" => env::var("MACOSX_DEPLOYMENT_TARGET")
            .ok()
            .map(|v| format!("-mmacosx-version-min={v}")),
        _ => None,
    };

    let air_files = SHADER_FILES
        .iter()
        .map(|shader| {
            let shader_path = shader_dir.join(shader);
            assert!(
                shader_path.is_file(),
                "listed shader {} is missing",
                shader_path.display()
            );
            let air_path = out_dir.join(shader.replace(".metal", ".air"));
            let mut command = xcrun();
            command.args(["metal", "-c", "-std=metal4.1", "-O3", "-fmetal-math-mode=relaxed"]);
            command.args(&min_os);
            if record_sources {
                command.arg("-frecord-sources");
            }
            command
                .arg("-I")
                .arg(shader_dir)
                .arg(&shader_path)
                .arg("-o")
                .arg(&air_path)
                .stdout(Stdio::null())
                .stderr(Stdio::inherit());
            let status = command.status().unwrap_or_else(|e| {
                panic!("cannot run xcrun to compile {shader} ({e}): {TOOLCHAIN_HINT}")
            });
            assert!(
                status.success(),
                "compiling {shader} failed ({status}); see the compiler output above"
            );
            air_path
        })
        .collect::<Vec<_>>();

    let metallib_path = out_dir.join("shaders.metallib");
    let status = xcrun()
        .arg("metallib")
        .args(&air_files)
        .arg("-o")
        .arg(&metallib_path)
        .status()
        .unwrap_or_else(|e| panic!("cannot run xcrun metallib ({e}): {TOOLCHAIN_HINT}"));
    assert!(
        status.success(),
        "linking shaders.metallib failed ({status})"
    );
}

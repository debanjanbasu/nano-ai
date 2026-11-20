#!/usr/bin/env bash
# Build, deploy and benchmark NanoAI on the attached iPhone SE 3 (A15).
# Prerequisite: Xcode > Settings > Accounts re-login (the stored session was
# rejected 2026-11-20), which regenerates the free-provisioning profile.
set -euo pipefail
cd "$(dirname "$0")/.."

UDID=${UDID:-00008110-001A44C63E10401E}

echo "== Rust static lib (aarch64-apple-ios, release) =="
cargo build --release -p nano-ios --target aarch64-apple-ios

echo "== Xcode build =="
xcodebuild -project ios/NanoAI.xcodeproj -scheme NanoAI -configuration Release \
    -destination "id=$UDID" -derivedDataPath target/ios-dd -allowProvisioningUpdates build

echo "== Install =="
xcrun devicectl device install app --device "$UDID" \
    target/ios-dd/Build/Products/Release-iphoneos/NanoAI.app

echo "== Model files (skipped when already present) =="
for f in minicpm-v-4.7-1b-tern4.bin tokenizer.bin; do
    if ! xcrun devicectl device copy from --device "$UDID" --domain-type appDataContainer \
        --domain-identifier com.debanjanbasu.nanoai --source "Documents/$f" \
        --output "/tmp/device_$f" >/dev/null 2>&1; then
        echo "  copying $f"
        xcrun devicectl device copy to --device "$UDID" --domain-type appDataContainer \
            --domain-identifier com.debanjanbasu.nanoai \
            --source "models/minicpm-v-4.7-1b/$f" --destination "Documents/$f"
    else
        echo "  $f already on device"
    fi
done

echo "== Headless benchmark =="
xcrun devicectl device process launch --device "$UDID" --console --terminate-existing \
    com.debanjanbasu.nanoai \
    --prompt "Count from 1 to 40, separated by commas." \
    --prompt "Explain how hash maps work internally, then how B-trees differ, then compare them." \
    --max-tokens 600

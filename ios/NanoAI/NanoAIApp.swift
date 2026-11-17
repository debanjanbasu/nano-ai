import SwiftUI

@main
struct NanoAIApp: App {
    var body: some Scene {
        WindowGroup { ContentView() }
    }
}

/// Model files (`minicpm-v-4.7-1b-tern4.bin`, `tokenizer.json`) and images live in Documents:
/// copy them with Finder file sharing or `xcrun devicectl device copy to`.
private let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0].path

/// Runs one prompt on a background thread and returns the engine's JSON report. Keeps the screen
/// awake meanwhile: a locked (backgrounded) app may not submit GPU work.
@MainActor
private func runEngine(prompt: String, image: String?, maxTokens: UInt32) async -> String {
    UIApplication.shared.isIdleTimerDisabled = true
    defer { UIApplication.shared.isIdleTimerDisabled = false }
    return await Task.detached(priority: .userInitiated) {
        let report = documents.withCString { dir in
            prompt.withCString { text in
                if let image {
                    return image.withCString { nano_ai_run(dir, text, $0, maxTokens) }
                }
                return nano_ai_run(dir, text, nil, maxTokens)
            }
        }
        defer { nano_ai_free(report) }
        return report.map { String(cString: $0) } ?? #"{"error":"no report"}"#
    }.value
}

/// Prefills the prompt up to `typed` on the engine thread ("" opens the engine and pins the
/// system turn), so Run only prefills what was typed since.
private func warmEngine(_ typed: String) async {
    await Task.detached(priority: .utility) {
        _ = documents.withCString { dir in typed.withCString { nano_ai_warm(dir, $0) } }
    }.value
}

struct ContentView: View {
    @State private var prompt = "Explain Gated DeltaNet in two sentences."
    @State private var output = ""
    @State private var running = false
    @State private var warming: Task<Void, Never>?

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            TextField("Prompt", text: $prompt, axis: .vertical)
                .textFieldStyle(.roundedBorder)
                .onChange(of: prompt) { _, typed in
                    // Prefill once typing pauses; Run then only prefills the last few tokens.
                    warming?.cancel()
                    warming = Task {
                        try? await Task.sleep(for: .milliseconds(400))
                        if !Task.isCancelled { await warmEngine(typed) }
                    }
                }
            Button(running ? "Running…" : "Run") {
                running = true
                Task {
                    output = await runEngine(prompt: prompt, image: nil, maxTokens: 512)
                    running = false
                }
            }
            .disabled(running)
            ScrollView {
                Text(output)
                    .font(.system(.footnote, design: .monospaced))
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
        .padding()
        .task {
            await warmEngine("")
            await autorun()
        }
    }

    /// `--prompt TEXT [--prompt TEXT …] [--image FILE] [--max-tokens N]` runs the prompts in order
    /// at launch with one engine, prints one report per line and exits (for
    /// `xcrun devicectl device process launch --console`).
    private func autorun() async {
        let args = ProcessInfo.processInfo.arguments
        func value(_ flag: String) -> String? {
            args.firstIndex(of: flag).flatMap { args.index(after: $0) < args.endIndex ? args[$0 + 1] : nil }
        }
        let prompts = args.indices.dropLast().filter { args[$0] == "--prompt" }.map { args[$0 + 1] }
        guard !prompts.isEmpty else { return }
        for text in prompts {
            let report = await runEngine(
                prompt: text,
                image: value("--image"),
                maxTokens: value("--max-tokens").flatMap(UInt32.init) ?? 512
            )
            print(report)
            fflush(stdout)
        }
        exit(0)
    }
}

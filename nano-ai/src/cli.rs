//! Interactive CLI commands: `chat`, `agents`, their terminal renderer (markdown, code
//! highlighting, LaTeX badges) and the shared flag parser.

use std::path::PathBuf;
use std::time::Instant;

use console::{Style, Term};
use nano_engine::model::{AgentStreamEvent, AsyncEngineHandle, NanoEngine};
use nano_engine::sampler::SamplingParams;
use nano_engine::vision::DownsampleMode;
use termimad::MadSkin;
use syntect::easy::HighlightLines;
use syntect::highlighting::ThemeSet;
use syntect::parsing::SyntaxSet;
use syntect::util::as_24_bit_terminal_escaped;

use super::DEFAULT_MODEL_PATH;

/// Print the CLI usage banner.
pub fn usage() {
    eprintln!("Usage: nano-ai <command> [options]");
    eprintln!();
    eprintln!("Commands:");
    eprintln!("  chat [--image PATH] [--json] <prompt>");
    eprintln!("  agents [--system PROMPT] [--json] <prompt1> <prompt2> ...");
    eprintln!("  serve [--port 8080]");
    eprintln!("  convert");
    eprintln!("  bench");
}

fn configure_skin() -> MadSkin {
    use termimad::crossterm::style::{Attribute, Color};
    let mut skin = MadSkin::default();
    skin.bold.set_fg(Color::Cyan);
    skin.italic.set_fg(Color::Magenta);
    skin.inline_code.set_fg(Color::Yellow);
    skin.code_block.set_fg(Color::Green);
    skin.code_block.left_margin = 2;
    if let Some(h0) = skin.headers.first_mut() {
        h0.set_fg(Color::Cyan);
        h0.add_attr(Attribute::Bold);
    }
    if let Some(h1) = skin.headers.get_mut(1) {
        h1.set_fg(Color::Blue);
        h1.add_attr(Attribute::Bold);
    }
    if let Some(h2) = skin.headers.get_mut(2) {
        h2.set_fg(Color::Green);
        h2.add_attr(Attribute::Bold);
    }
    skin.bullet = termimad::StyledChar::from_fg_char(Color::Cyan, '•');
    skin.quote_mark = termimad::StyledChar::from_fg_char(Color::Magenta, '┃');
    skin
}

fn replace_boxed(input: &str, is_tty: bool) -> String {
    use std::fmt::Write as _;
    let badge_style = Style::new().black().on_green().bold().for_stdout();
    let mut out = String::with_capacity(input.len());
    let mut cursor = input;
    while let Some(pos) = cursor.find("\\boxed{") {
        out.push_str(cursor.get(..pos).unwrap_or_default());
        let rest = cursor.get(pos + 7..).unwrap_or_default();
        let mut depth = 1_usize;
        let mut end_idx = None;
        for (idx, ch) in rest.char_indices() {
            if ch == '{' {
                depth += 1;
            } else if ch == '}' {
                depth -= 1;
                if depth == 0 {
                    end_idx = Some(idx);
                    break;
                }
            }
        }
        if let Some(end) = end_idx {
            let inner = rest.get(..end).unwrap_or_default().trim();
            if is_tty {
                let _ = write!(out, " {} ", badge_style.apply_to(format!(" ✓ {inner} ")));
            } else {
                let _ = write!(out, "[ {inner} ]");
            }
            cursor = rest.get(end + 1..).unwrap_or_default();
        } else {
            out.push_str("\\boxed{");
            cursor = rest;
        }
    }
    out.push_str(cursor);
    out
}
fn replace_latex_macros(input: &str) -> String {
    let mut s = input.replace("</think>", "");
    for macro_prefix in ["\\mathbf{", "\\textbf{", "\\text{", "\\mathrm{"] {
        while let Some(pos) = s.find(macro_prefix) {
            let after = s.get(pos + macro_prefix.len()..).unwrap_or_default();
            if let Some(end) = after.find('}') {
                let inner = after.get(..end).unwrap_or_default().to_owned();
                s.replace_range(pos..=(pos + macro_prefix.len() + end), &format!("**{inner}**"));
            } else {
                break;
            }
        }
    }
    s
}


fn replace_delimited_spans(input: &str, open: &str, close: &str, is_display: bool) -> String {
    use std::fmt::Write as _;
    let mut out = String::with_capacity(input.len());
    let mut cursor = input;
    while let Some(start) = cursor.find(open) {
        out.push_str(cursor.get(..start).unwrap_or_default());
        let after_open = cursor.get(start + open.len()..).unwrap_or_default();
        if let Some(end) = after_open.find(close) {
            let inner = unicodeit::replace(after_open.get(..end).unwrap_or_default().trim());
            if is_display {
                let _ = write!(out, "  `{inner}`  ");
            } else {
                let _ = write!(out, "`{inner}`");
            }
            cursor = after_open.get(end + close.len()..).unwrap_or_default();
        } else {
            out.push_str(after_open);
            cursor = "";
            break;
        }
    }
    out.push_str(cursor);
    out.replace(open, "").replace(close, "")
}

fn replace_latex_symbols_safe(input: &str) -> String {
    if !input.contains('\\') {
        return input.to_owned();
    }
    input
        .replace("\\times", "×")
        .replace("\\div", "÷")
        .replace("\\cdot", "·")
        .replace("\\pm", "±")
        .replace("\\neq", "≠")
        .replace("\\leq", "≤")
        .replace("\\geq", "≥")
        .replace("\\approx", "≈")
        .replace("\\rightarrow", "→")
        .replace("\\to", "→")
}

/// Apply the LaTeX/markdown substitutions to one line (TTY rendering fast path).
#[must_use]
pub fn preprocess_math_and_markdown(line: &str, is_tty: bool) -> String {
    if !line.contains('\\') && !line.contains("</think>") {
        return line.to_owned();
    }
    let step1 = replace_latex_symbols_safe(&replace_latex_macros(&line.replace("\\n", "\n")));
    let step2 = replace_boxed(&step1, is_tty);
    let step3 = replace_delimited_spans(&step2, "\\(", "\\)", false);
    replace_delimited_spans(&step3, "\\[", "\\]", true)
}

struct TerminalStreamer {
    term: Term,
    skin: MadSkin,
    think_style: Style,
    border_style: Style,
    in_thinking: bool,
    started_at: Instant,
    thinking_tokens: usize,
    pending_line: String,
    in_code_fence: bool,
    code_lang: String,
    syntax_set: SyntaxSet,
    theme_set: ThemeSet,
}

impl TerminalStreamer {
    fn new() -> Self {
        let term = Term::stdout();
        let is_tty = term.is_term();
        let border_style = Style::new().cyan().dim().for_stdout();
        if is_tty {
            let _ = term.write_line(&format!(
                "{}",
                border_style.apply_to("╭─ 💭 Reasoning ──────────────────────────────────────────")
            ));
            let _ = term.write_str(&format!("{}", border_style.apply_to("│ ")));
        } else {
            let _ = term.write_str("💭 ");
        }
        Self {
            term,
            skin: configure_skin(),
            think_style: Style::new().dim().italic().for_stdout(),
            border_style,
            in_thinking: true,
            started_at: Instant::now(),
            thinking_tokens: 0,
            pending_line: String::new(),
            in_code_fence: false,
            code_lang: String::new(),
            syntax_set: SyntaxSet::load_defaults_newlines(),
            theme_set: ThemeSet::load_defaults(),
        }
    }

    fn write_thinking_chunk(&self, text: &str) {
        if text.is_empty() {
            return;
        }
        let clean = replace_latex_symbols_safe(text);
        if self.term.is_term() {
            let bar = format!("\n{}", self.border_style.apply_to("│ "));
            let mut first = true;
            for part in clean.split('\n') {
                if !first {
                    let _ = self.term.write_str(&bar);
                }
                first = false;
                if !part.is_empty() {
                    let _ = self
                        .term
                        .write_str(&format!("{}", self.think_style.apply_to(part)));
                }
            }
        } else {
            let _ = self.term.write_str(&clean);
        }
    }

    fn push(&mut self, piece: &str) {
        if self.in_thinking {
            if let Some((before, after)) = piece.split_once("</think>") {
                self.in_thinking = false;
                let clean_before = before.replace("<think>", "");
                if !clean_before.is_empty() {
                    self.write_thinking_chunk(&clean_before);
                }
                let elapsed = self.started_at.elapsed().as_secs_f64();
                let _ = self.term.write_line("");
                if self.term.is_term() {
                    let footer_box = format!(
                        "╰─ ✨ Thought for {elapsed:.1}s · {} tokens ──────────────────",
                        self.thinking_tokens
                    );
                    let _ = self
                        .term
                        .write_line(&format!("{}\n", self.border_style.apply_to(footer_box)));
                } else {
                    let _ = self.term.write_line(&format!(
                        "💭 Thought for {elapsed:.1}s ({} tokens)\n",
                        self.thinking_tokens
                    ));
                }
                let trimmed = after.trim_start_matches('\n');
                if !trimmed.is_empty() {
                    self.emit_answer(trimmed);
                }
            } else {
                self.thinking_tokens += 1;
                let clean = piece.replace("<think>", "");
                self.write_thinking_chunk(&clean);
            }
            return;
        }
        self.emit_answer(piece);
    }
    fn highlight_code_line(&self, line: &str) -> String {
        let syntax = if self.code_lang.is_empty() {
            self.syntax_set.find_syntax_plain_text()
        } else {
            self.syntax_set
                .find_syntax_by_token(&self.code_lang)
                .unwrap_or_else(|| self.syntax_set.find_syntax_plain_text())
        };
        let Some(theme) = self
            .theme_set
            .themes
            .get("base16-ocean.dark")
            .or_else(|| self.theme_set.themes.values().next())
        else {
            return line.to_owned();
        };
        let mut h = HighlightLines::new(syntax, theme);
        h.highlight_line(line, &self.syntax_set).map_or_else(
            |_| line.to_owned(),
            |ranges| format!("{}\x1b[0m", as_24_bit_terminal_escaped(&ranges[..], false)),
        )
    }

    fn render_markdown_line(&mut self, line: &str) {
        let is_tty = self.term.is_term();
        let trimmed = line.trim_start();
        if let Some(after_fence) = trimmed.strip_prefix("```") {
            if self.in_code_fence {
                self.in_code_fence = false;
                self.code_lang.clear();
                if is_tty {
                    let _ = self.term.write_line(&format!(
                        "{}",
                        self.border_style
                            .apply_to("╰────────────────────────────────────────────────────")
                    ));
                } else {
                    let _ = self.term.write_line(line);
                }
            } else {
                self.in_code_fence = true;
                after_fence
                    .split_whitespace()
                    .next()
                    .unwrap_or_default()
                    .clone_into(&mut self.code_lang);
                if is_tty {
                    let label = if self.code_lang.is_empty() {
                        "code".to_owned()
                    } else {
                        self.code_lang.clone()
                    };
                    let _ = self.term.write_line(&format!(
                        "{}",
                        self.border_style
                            .apply_to(format!("╭─ 📝 {label} ──────────────────────────────────────"))
                    ));
                } else {
                    let _ = self.term.write_line(line);
                }
            }
            return;
        }
        if self.in_code_fence {
            if is_tty {
                let bar = self.border_style.apply_to("│ ");
                let highlighted = self.highlight_code_line(line);
                let _ = self.term.write_line(&format!("{bar}{highlighted}"));
            } else {
                let _ = self.term.write_line(line);
            }
            return;
        }
        let formatted = preprocess_math_and_markdown(line, is_tty);
        for subline in formatted.split('\n') {
            if is_tty && !subline.is_empty() {
                let rendered = self.skin.term_text(subline).to_string();
                let _ = self.term.write_str(&rendered);
                if !rendered.ends_with('\n') {
                    let _ = self.term.write_line("");
                }
            } else {
                let _ = self.term.write_line(subline);
            }
        }
    }

    fn emit_answer(&mut self, piece: &str) {
        self.pending_line.push_str(piece);
        while let Some(nl) = self.pending_line.find('\n') {
            let line = self.pending_line.get(..nl).unwrap_or_default().to_owned();
            self.pending_line.drain(..=nl);
            self.render_markdown_line(&line);
        }
    }

    fn finish(&mut self) {
        if self.in_thinking && self.term.is_term() {
            let elapsed = self.started_at.elapsed().as_secs_f64();
            let _ = self.term.write_line("");
            let footer_box = format!(
                "╰─ ✨ Thought for {elapsed:.1}s · {} tokens ──────────────────",
                self.thinking_tokens
            );
            let _ = self
                .term
                .write_line(&format!("{}", self.border_style.apply_to(footer_box)));
        }
        if !self.pending_line.is_empty() {
            let tail = std::mem::take(&mut self.pending_line);
            self.render_markdown_line(&tail);
        }
    }
}

fn render_telemetry_footer(
    device_name: &str,
    ane_enabled: bool,
    report: &nano_engine::model::GenerationReport,
) {
    let err_term = Term::stderr();
    if !err_term.is_term() {
        let footer_style = Style::new().dim().for_stderr();
        let footer = format!(
            "[{} | ANE: {} | cache: {}/{} | prefill: {:.1} tok/s ({:.1} ms) | decode: {:.1} tok/s ({:.1} ms) | encode: {:.1} ms | spec accepted: {}/{}]",
            device_name,
            ane_enabled,
            report.cached_prefix_tokens,
            report.prompt_tokens,
            report.prefill_tok_per_s,
            report.prefill_ms,
            report.decode_tok_per_s,
            report.decode_ms,
            report.decode_encode_ms,
            report.accepted_tokens,
            report.drafted_tokens
        );
        eprintln!("{}", footer_style.apply_to(footer));
        return;
    }

    let sep = Style::new().dim().for_stderr().apply_to("│");
    let chip_badge = Style::new()
        .cyan()
        .bold()
        .for_stderr()
        .apply_to(format!("⚡ {device_name}"));
    let ane_badge = if ane_enabled {
        Style::new().green().bold().for_stderr().apply_to("ANE ● ON")
    } else {
        Style::new().dim().for_stderr().apply_to("ANE ○ OFF")
    };
    let prefill_seg = Style::new().yellow().for_stderr().apply_to(format!(
        "Prefill {:.1} tok/s ({:.1} ms)",
        report.prefill_tok_per_s, report.prefill_ms
    ));

    let filled_bars = ((report.decode_tok_per_s / 70.0).round() as usize).clamp(1, 6);
    let mini_bar = format!(
        "{}{}",
        "▰".repeat(filled_bars),
        "▱".repeat(6 - filled_bars)
    );
    let decode_seg = Style::new().green().bold().for_stderr().apply_to(format!(
        "Decode {:.1} tok/s ({:.1} ms) {mini_bar}",
        report.decode_tok_per_s, report.decode_ms
    ));

    let pct = if report.drafted_tokens > 0 {
        (report.accepted_tokens as f64 / report.drafted_tokens as f64) * 100.0
    } else {
        0.0
    };
    let spec_seg = Style::new().magenta().for_stderr().apply_to(format!(
        "DSpark {}/{} ({pct:.0}%)",
        report.accepted_tokens, report.drafted_tokens
    ));
    let encode_seg = Style::new().dim().for_stderr().apply_to(format!(
        "CPU encode {:.1} ms",
        report.decode_encode_ms
    ));

    if report.cached_prefix_tokens > 0 {
        let cache_seg = Style::new().cyan().bold().for_stderr().apply_to(format!(
            "Cache {}/{}",
            report.cached_prefix_tokens, report.prompt_tokens
        ));
        eprintln!(
            "\n{chip_badge} {sep} {ane_badge} {sep} {cache_seg} {sep} {prefill_seg} {sep} {decode_seg} {sep} {spec_seg} {sep} {encode_seg}"
        );
    } else {
        eprintln!(
            "\n{chip_badge} {sep} {ane_badge} {sep} {prefill_seg} {sep} {decode_seg} {sep} {spec_seg} {sep} {encode_seg}"
        );
    }
}
fn parse_common_flag(
    args: &[String],
    idx: &mut usize,
    model_path: &mut PathBuf,
    json_out: &mut bool,
    params: &mut SamplingParams,
) -> bool {
    let i = *idx;
    match args[i].as_str() {
        "--model" if i + 1 < args.len() => {
            *model_path = PathBuf::from(&args[i + 1]);
            *idx += 2;
            true
        }
        "--json" => {
            *json_out = true;
            *idx += 1;
            true
        }
        "--temperature" if i + 1 < args.len() => {
            if let Ok(t) = args[i + 1].parse::<f32>() {
                params.temperature = t;
                if params.temperature > 0.0 && params.top_k == 1 {
                    params.top_k = 100;
                }
            }
            *idx += 2;
            true
        }
        "--top-p" if i + 1 < args.len() => {
            if let Ok(p) = args[i + 1].parse::<f32>() {
                params.top_p = p;
            }
            *idx += 2;
            true
        }
        "--top-k" if i + 1 < args.len() => {
            if let Ok(k) = args[i + 1].parse::<usize>() {
                params.top_k = k;
            }
            *idx += 2;
            true
        }
        "--repetition-penalty" if i + 1 < args.len() => {
            if let Ok(r) = args[i + 1].parse::<f32>() {
                params.repetition_penalty = r;
            }
            *idx += 2;
            true
        }
        _ => false,
    }
}

/// Run the interactive single-turn `chat` command.
pub async fn run_chat(args: &[String]) -> nano_engine::Result<()> {
    let mut model_path = PathBuf::from(DEFAULT_MODEL_PATH);
    let mut image_path: Option<PathBuf> = None;
    let downsample_mode = DownsampleMode::Mode16x;
    let max_tokens = nano_engine::model::MAX_CONTEXT_TOKENS;
    let mut params = SamplingParams::default();
    let mut json_out = false;
    let mut prompt_parts = Vec::new();

    let mut i = 0;
    while i < args.len() {
        if parse_common_flag(args, &mut i, &mut model_path, &mut json_out, &mut params) {
            continue;
        }
        match args[i].as_str() {
            "--image" if i + 1 < args.len() => {
                image_path = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            other => {
                prompt_parts.push(other.to_owned());
                i += 1;
            }
        }
    }
    let prompt = prompt_parts.join(" ");
    if prompt.is_empty() {
        return Err(nano_engine::Error::InvalidArgument(
            "prompt cannot be empty".into(),
        ));
    }

    // Overlap async image file read with AsyncEngineHandle startup on Apple Silicon
    let img_fut = async {
        if let Some(img_path) = &image_path {
            Ok::<Option<Vec<u8>>, nano_engine::Error>(Some(tokio::fs::read(img_path).await?))
        } else {
            Ok(None)
        }
    };
    let (img_bytes_opt, handle) =
        tokio::try_join!(img_fut, AsyncEngineHandle::open(&model_path))?;

    let (piece_tx, mut piece_rx) = tokio::sync::mpsc::unbounded_channel::<String>();
    let tx_opt = (!json_out).then_some(piece_tx);
    let handle_clone = handle.clone();
    let gen_fut = tokio::spawn(async move {
        if let Some(img_bytes) = img_bytes_opt {
            handle_clone
                .generate_multimodal(
                    img_bytes,
                    prompt,
                    downsample_mode,
                    max_tokens,
                    params,
                    tx_opt,
                )
                .await
        } else {
            let formatted = NanoEngine::format_chat_prompt(&prompt);
            handle_clone
                .generate_text(formatted, max_tokens, params, tx_opt)
                .await
        }
    });

    let mut streamer = (!json_out).then(TerminalStreamer::new);
    while let Some(piece) = piece_rx.recv().await {
        if let Some(s) = &mut streamer {
            s.push(&piece);
        }
    }
    if let Some(s) = &mut streamer {
        s.finish();
    }
    let report = gen_fut
        .await
        .map_err(|e| nano_engine::Error::InvalidArgument(format!("worker join failed: {e}")))??;
    if json_out {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        render_telemetry_footer(handle.device_name(), handle.ane_enabled(), &report);
    }
    Ok(())
}

/// Run the multi-agent `agents` command, streaming per-agent output.
pub async fn run_agents(args: &[String]) -> nano_engine::Result<()> {
    let mut model_path = PathBuf::from(DEFAULT_MODEL_PATH);
    let mut system_prompt = NanoEngine::DEFAULT_SYSTEM_PROMPT.to_owned();
    let max_tokens = nano_engine::model::MAX_CONTEXT_TOKENS;
    let mut params = SamplingParams::default();
    let mut json_out = false;
    let mut agent_prompts = Vec::new();

    let mut idx = 0;
    while idx < args.len() {
        if parse_common_flag(args, &mut idx, &mut model_path, &mut json_out, &mut params) {
            continue;
        }
        match args[idx].as_str() {
            "--system" if idx + 1 < args.len() => {
                system_prompt.clone_from(&args[idx + 1]);
                idx += 2;
            }
            other => {
                agent_prompts.push(other.to_owned());
                idx += 1;
            }
        }
    }
    if agent_prompts.is_empty() {
        return Err(nano_engine::Error::InvalidArgument(
            "at least one agent prompt is required".into(),
        ));
    }

    let handle = AsyncEngineHandle::open(&model_path).await?;
    if json_out {
        let reports = handle
            .run_agents(system_prompt, agent_prompts, max_tokens, params, None)
            .await?;
        println!("{}", serde_json::to_string_pretty(&reports)?);
        return Ok(());
    }

    let total_agents = agent_prompts.len();
    let swarm_start = Instant::now();
    let (ev_tx, mut ev_rx) = tokio::sync::mpsc::unbounded_channel::<AgentStreamEvent>();
    let handle_clone = handle.clone();
    let worker = tokio::spawn(async move {
        handle_clone
            .run_agents(system_prompt, agent_prompts, max_tokens, params, Some(ev_tx))
            .await
    });

    let header_style = Style::new().cyan().bold().for_stdout();
    let mut streamer: Option<TerminalStreamer> = None;
    let mut total_gen_tokens = 0_usize;
    let mut total_cached_tokens = 0_usize;
    let mut total_decode_ms = 0.0_f64;

    while let Some(ev) = ev_rx.recv().await {
        match ev {
            AgentStreamEvent::Header {
                agent_idx,
                total_agents: total,
            } => {
                println!(
                    "\n{}",
                    header_style.apply_to(format!(
                        "╭─ 🤖 Agent {}/{total} ──────────────────────────────────────────",
                        agent_idx + 1
                    ))
                );
                streamer = Some(TerminalStreamer::new());
            }
            AgentStreamEvent::Piece(piece) => {
                if let Some(s) = &mut streamer {
                    s.push(&piece);
                }
            }
            AgentStreamEvent::AgentDone {
                device_name,
                ane_enabled,
                report,
            } => {
                if let Some(mut s) = streamer.take() {
                    s.finish();
                }
                total_gen_tokens += report.generated_tokens;
                total_cached_tokens += report.cached_prefix_tokens;
                total_decode_ms += report.decode_ms;
                render_telemetry_footer(&device_name, ane_enabled, &report);
            }
        }
    }

    let _ = worker
        .await
        .map_err(|e| nano_engine::Error::InvalidArgument(format!("worker join failed: {e}")))??;

    let wall_s = swarm_start.elapsed().as_secs_f64();
    let avg_decode_tok_s = if total_decode_ms > 0.0 {
        (total_gen_tokens as f64) / (total_decode_ms / 1000.0)
    } else {
        0.0
    };
    let summary_style = Style::new().green().bold().for_stderr();
    eprintln!(
        "\n{}",
        summary_style.apply_to(format!(
            "╰─ 🚀 Swarm complete: {total_agents} agents · {total_gen_tokens} generated tokens · {total_cached_tokens} cached prefix tokens · {avg_decode_tok_s:.1} decode tok/s ({wall_s:.2}s total)"
        ))
    );
    Ok(())
}

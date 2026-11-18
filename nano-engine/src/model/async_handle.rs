//! Tokio worker + thread-safe handle around the single-threaded [`NanoEngine`]: the engine owns
//! Metal objects that cannot cross threads, so all requests go through one worker's command
//! queue while callers hold cloneable, `Send` handles.

use std::path::Path;

use super::NanoEngine;
use super::{GenerationReport, ScoredPrompt, ScorePrompt};
use crate::sampler::SamplingParams;
use crate::vision::DownsampleMode;
use crate::Error;

/// Streaming event emitted per agent during `AsyncEngineHandle::run_agents`.
pub enum AgentStreamEvent {
    Header {
        agent_idx: usize,
        total_agents: usize,
    },
    Piece(String),
    AgentDone {
        device_name: String,
        ane_enabled: bool,
        report: Box<GenerationReport>,
    },
}

enum AsyncEngineCommand {
    ChatText {
        prompt_text: String,
        max_new_tokens: usize,
        params: SamplingParams,
        piece_tx: Option<tokio::sync::mpsc::UnboundedSender<String>>,
        reply: tokio::sync::oneshot::Sender<crate::Result<GenerationReport>>,
    },
    ChatIds {
        prompt_ids: Vec<u32>,
        max_new_tokens: usize,
        params: SamplingParams,
        piece_tx: Option<tokio::sync::mpsc::UnboundedSender<String>>,
        reply: tokio::sync::oneshot::Sender<crate::Result<GenerationReport>>,
    },
    Multimodal {
        image_bytes: Vec<u8>,
        prompt: String,
        mode: DownsampleMode,
        max_new_tokens: usize,
        params: SamplingParams,
        piece_tx: Option<tokio::sync::mpsc::UnboundedSender<String>>,
        reply: tokio::sync::oneshot::Sender<crate::Result<GenerationReport>>,
    },
    WarmPrefix {
        prefix_text: String,
        reply: tokio::sync::oneshot::Sender<crate::Result<usize>>,
    },
    Score {
        prompt: ScorePrompt,
        reply: tokio::sync::oneshot::Sender<crate::Result<ScoredPrompt>>,
    },
    Agents {
        system_prompt: String,
        agent_prompts: Vec<String>,
        max_new_tokens: usize,
        params: SamplingParams,
        event_tx: Option<tokio::sync::mpsc::UnboundedSender<AgentStreamEvent>>,
        reply: tokio::sync::oneshot::Sender<crate::Result<Vec<GenerationReport>>>,
    },
}

/// Thread-safe `Clone + Send + Sync` Tokio async handle to a dedicated `NanoEngine` worker.
///
/// Runs a single-threaded Tokio runtime that awaits non-blocking GPU completions and shares `PrefixCache`.
#[derive(Clone)]
pub struct AsyncEngineHandle {
    tx: tokio::sync::mpsc::UnboundedSender<AsyncEngineCommand>,
    device_name: String,
    ane_status: serde_json::Value,
}

impl AsyncEngineHandle {
    /// Open `NanoEngine` on a dedicated async worker thread and return a cloneable `AsyncEngineHandle`.
    pub async fn open(model_path: &Path) -> crate::Result<Self> {
        let path_buf = model_path.to_path_buf();
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<AsyncEngineCommand>();
        let (init_tx, init_rx) =
            tokio::sync::oneshot::channel::<crate::Result<(String, serde_json::Value)>>();
        std::thread::Builder::new()
            .name("nano-engine-tokio-worker".into())
            .spawn(move || {
                let rt = match tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                {
                    Ok(r) => r,
                    Err(err) => {
                        let _ = init_tx.send(Err(Error::Io(err)));
                        return;
                    }
                };
                let mut engine = match NanoEngine::open(&path_buf) {
                    Ok(eng) => {
                        let _ = init_tx.send(Ok((eng.device_name(), eng.ane_status())));
                        eng
                    }
                    Err(err) => {
                        let _ = init_tx.send(Err(err));
                        return;
                    }
                };
                rt.block_on(async move {
                    while let Some(cmd) = rx.recv().await {
                        match cmd {
                            AsyncEngineCommand::ChatText {
                                prompt_text,
                                max_new_tokens,
                                params,
                                piece_tx,
                                reply,
                            } => {
                                let res = match engine.encode_text(&prompt_text) {
                                    Ok(ids) => {
                                        engine
                                            .generate_ids_streaming_async_stoppable(
                                                &ids,
                                                max_new_tokens,
                                                params,
                                                |piece| {
                                                    if let Some(tx) = &piece_tx {
                                                        let _ = tx.send(piece.to_owned());
                                                    }
                                                },
                                                || piece_tx.as_ref().is_some_and(tokio::sync::mpsc::UnboundedSender::is_closed),
                                            )
                                            .await
                                    }
                                    Err(e) => Err(e),
                                };
                                let _ = reply.send(res);
                            }
                            AsyncEngineCommand::ChatIds {
                                prompt_ids,
                                max_new_tokens,
                                params,
                                piece_tx,
                                reply,
                            } => {
                                let res = engine
                                    .generate_ids_streaming_async_stoppable(
                                        &prompt_ids,
                                        max_new_tokens,
                                        params,
                                        |piece| {
                                            if let Some(tx) = &piece_tx {
                                                let _ = tx.send(piece.to_owned());
                                            }
                                        },
                                        || piece_tx.as_ref().is_some_and(tokio::sync::mpsc::UnboundedSender::is_closed),
                                    )
                                    .await;
                                let _ = reply.send(res);
                            }
                            AsyncEngineCommand::Multimodal {
                                image_bytes,
                                prompt,
                                mode,
                                max_new_tokens,
                                params,
                                piece_tx,
                                reply,
                            } => {
                                let res = engine
                                    .generate_multimodal_streaming_async_stoppable(
                                        &image_bytes,
                                        &prompt,
                                        mode,
                                        max_new_tokens,
                                        params,
                                        |piece| {
                                            if let Some(tx) = &piece_tx {
                                                let _ = tx.send(piece.to_owned());
                                            }
                                        },
                                        || piece_tx.as_ref().is_some_and(tokio::sync::mpsc::UnboundedSender::is_closed),
                                    )
                                    .await;
                                let _ = reply.send(res);
                            }
                            AsyncEngineCommand::WarmPrefix { prefix_text, reply } => {
                                let _ = reply.send(engine.warm_prefix_cache(&prefix_text));
                            }
                            AsyncEngineCommand::Score { prompt, reply } => {
                                let _ = reply.send(engine.score_prompt(prompt));
                            }
                            AsyncEngineCommand::Agents {
                                system_prompt,
                                agent_prompts,
                                max_new_tokens,
                                params,
                                event_tx,
                                reply,
                            } => {
                                let res = async {
                                    if agent_prompts.len() > 1 {
                                        let shared_prefix = format!(
                                            "<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n"
                                        );
                                        let _ = engine.warm_prefix_cache(&shared_prefix);
                                    }
                                    let total_agents = agent_prompts.len();
                                    let dev_name = engine.device_name();
                                    let ane_enabled =
                                        engine.ane_status()["enabled"].as_bool().unwrap_or(false);
                                    let mut reports = Vec::with_capacity(total_agents);
                                    for (agent_idx, agent_prompt) in
                                        agent_prompts.iter().enumerate()
                                    {
                                        if let Some(tx) = &event_tx {
                                            let _ = tx.send(AgentStreamEvent::Header {
                                                agent_idx,
                                                total_agents,
                                            });
                                        }
                                        let formatted = NanoEngine::format_chat_prompt_with_system(
                                            &system_prompt,
                                            agent_prompt,
                                        );
                                        let ids = engine.encode_text(&formatted)?;
                                        let rep = engine
                                            .generate_ids_streaming_async(
                                                &ids,
                                                max_new_tokens,
                                                params.clone(),
                                                |piece| {
                                                    if let Some(tx) = &event_tx {
                                                        let _ = tx.send(AgentStreamEvent::Piece(
                                                            piece.to_owned(),
                                                        ));
                                                    }
                                                },
                                            )
                                            .await?;
                                        if let Some(tx) = &event_tx {
                                            let _ = tx.send(AgentStreamEvent::AgentDone {
                                                device_name: dev_name.clone(),
                                                ane_enabled,
                                                report: Box::new(rep.clone()),
                                            });
                                        }
                                        reports.push(rep);
                                    }
                                    Ok(reports)
                                }
                                .await;
                                let _ = reply.send(res);
                            }
                        }
                    }
                });
            })
            .map_err(Error::Io)?;
        let (device_name, ane_status) = init_rx
            .await
            .map_err(|_| Error::InvalidArgument("worker init dropped".into()))??;
        Ok(Self {
            tx,
            device_name,
            ane_status,
        })
    }

    #[must_use]
    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    #[must_use]
    pub const fn ane_status(&self) -> &serde_json::Value {
        &self.ane_status
    }

    #[must_use]
    pub fn ane_enabled(&self) -> bool {
        self.ane_status["enabled"].as_bool().unwrap_or(false)
    }

    /// Asynchronously pre-prefill and pin `prefix_text` in the shared GPU `PrefixCache`.
    pub async fn warm_prefix_cache(&self, prefix_text: String) -> crate::Result<usize> {
        let (reply_tx, reply_rx) = tokio::sync::oneshot::channel();
        self.tx
            .send(AsyncEngineCommand::WarmPrefix {
                prefix_text,
                reply: reply_tx,
            })
            .map_err(|_| Error::InvalidArgument("AsyncEngineHandle worker closed".into()))?;
        reply_rx
            .await
            .map_err(|_| Error::InvalidArgument("AsyncEngineHandle reply dropped".into()))?
    }

    /// Asynchronously teacher-force `prompt` (see `NanoEngine::score_tokens`).
    pub async fn score_prompt(&self, prompt: ScorePrompt) -> crate::Result<ScoredPrompt> {
        let (reply_tx, reply_rx) = tokio::sync::oneshot::channel();
        self.tx
            .send(AsyncEngineCommand::Score {
                prompt,
                reply: reply_tx,
            })
            .map_err(|_| Error::InvalidArgument("AsyncEngineHandle worker closed".into()))?;
        reply_rx
            .await
            .map_err(|_| Error::InvalidArgument("AsyncEngineHandle reply dropped".into()))?
    }

    /// Asynchronously encode and generate a completion from `prompt_text`, streaming pieces to `piece_tx` if provided.
    pub async fn generate_text(
        &self,
        prompt_text: String,
        max_new_tokens: usize,
        params: SamplingParams,
        piece_tx: Option<tokio::sync::mpsc::UnboundedSender<String>>,
    ) -> crate::Result<GenerationReport> {
        let (reply_tx, reply_rx) = tokio::sync::oneshot::channel();
        self.tx
            .send(AsyncEngineCommand::ChatText {
                prompt_text,
                max_new_tokens,
                params,
                piece_tx,
                reply: reply_tx,
            })
            .map_err(|_| Error::InvalidArgument("AsyncEngineHandle worker closed".into()))?;
        reply_rx
            .await
            .map_err(|_| Error::InvalidArgument("AsyncEngineHandle reply dropped".into()))?
    }

    /// Asynchronously generate a completion from `prompt_ids`, streaming pieces to `piece_tx` if provided.
    pub async fn generate_ids(
        &self,
        prompt_ids: Vec<u32>,
        max_new_tokens: usize,
        params: SamplingParams,
        piece_tx: Option<tokio::sync::mpsc::UnboundedSender<String>>,
    ) -> crate::Result<GenerationReport> {
        let (reply_tx, reply_rx) = tokio::sync::oneshot::channel();
        self.tx
            .send(AsyncEngineCommand::ChatIds {
                prompt_ids,
                max_new_tokens,
                params,
                piece_tx,
                reply: reply_tx,
            })
            .map_err(|_| Error::InvalidArgument("AsyncEngineHandle worker closed".into()))?;
        reply_rx
            .await
            .map_err(|_| Error::InvalidArgument("AsyncEngineHandle reply dropped".into()))?
    }

    /// Asynchronously generate a multimodal completion from `image_bytes` + `prompt`.
    pub async fn generate_multimodal(
        &self,
        image_bytes: Vec<u8>,
        prompt: String,
        mode: DownsampleMode,
        max_new_tokens: usize,
        params: SamplingParams,
        piece_tx: Option<tokio::sync::mpsc::UnboundedSender<String>>,
    ) -> crate::Result<GenerationReport> {
        let (reply_tx, reply_rx) = tokio::sync::oneshot::channel();
        self.tx
            .send(AsyncEngineCommand::Multimodal {
                image_bytes,
                prompt,
                mode,
                max_new_tokens,
                params,
                piece_tx,
                reply: reply_tx,
            })
            .map_err(|_| Error::InvalidArgument("AsyncEngineHandle worker closed".into()))?;
        reply_rx
            .await
            .map_err(|_| Error::InvalidArgument("AsyncEngineHandle reply dropped".into()))?
    }

    /// Asynchronously execute a batch of collaborating agents sharing `system_prompt` and `PrefixCache`.
    pub async fn run_agents(
        &self,
        system_prompt: String,
        agent_prompts: Vec<String>,
        max_new_tokens: usize,
        params: SamplingParams,
        event_tx: Option<tokio::sync::mpsc::UnboundedSender<AgentStreamEvent>>,
    ) -> crate::Result<Vec<GenerationReport>> {
        let (reply_tx, reply_rx) = tokio::sync::oneshot::channel();
        self.tx
            .send(AsyncEngineCommand::Agents {
                system_prompt,
                agent_prompts,
                max_new_tokens,
                params,
                event_tx,
                reply: reply_tx,
            })
            .map_err(|_| Error::InvalidArgument("AsyncEngineHandle worker closed".into()))?;
        reply_rx
            .await
            .map_err(|_| Error::InvalidArgument("AsyncEngineHandle reply dropped".into()))?
    }
}


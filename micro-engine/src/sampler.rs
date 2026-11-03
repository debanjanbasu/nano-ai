//! Sampler and verification logic for `micro-ai`.

use micro_metal::sampling::TopKCandidate;

#[derive(Debug, Clone)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub min_p: f32,
    pub repetition_penalty: f32,
    pub eos_tokens: Vec<u32>,
    pub seed: u64,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_k: 1,
            top_p: 0.95,
            min_p: 0.0,
            repetition_penalty: 1.0,
            eos_tokens: vec![248_044, 248_045, 248_046],
            seed: 0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SamplingResult {
    pub token_id: u32,
    pub is_eos: bool,
}

pub struct Sampler {
    params: SamplingParams,
    rng_state: u64,
}

impl Sampler {
    #[must_use]
    pub const fn new(params: SamplingParams) -> Self {
        Self {
            rng_state: params.seed ^ 0x9E37_79B9_7F4A_7C15,
            params,
        }
    }

    #[must_use]
    pub const fn is_greedy(&self) -> bool {
        self.params.temperature <= 0.0 || self.params.top_k == 1
    }

    #[must_use]
    pub fn active_top_k(&self) -> usize {
        if self.is_greedy() {
            1
        } else if self.params.top_k == 0 {
            micro_metal::sampling::MAX_TOP_K
        } else {
            self.params.top_k.clamp(1, micro_metal::sampling::MAX_TOP_K)
        }
    }

    #[must_use]
    pub const fn repetition_penalty(&self) -> f32 {
        self.params.repetition_penalty
    }

    #[must_use]
    pub fn is_eos(&self, token_id: u32) -> bool {
        self.params.eos_tokens.contains(&token_id)
    }

    fn next_f32(&mut self) -> f32 {
        self.rng_state = self.rng_state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.rng_state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        ((z >> 40) as u32 as f32) / ((1_u32 << 24) as f32)
    }

    pub fn select_from_candidates(&mut self, candidates: &[TopKCandidate]) -> SamplingResult {
        if candidates.is_empty() {
            return SamplingResult {
                token_id: 0,
                is_eos: true,
            };
        }
        if self.is_greedy() {
            let best = candidates[0].token_id;
            return SamplingResult {
                token_id: best,
                is_eos: self.is_eos(best),
            };
        }
        let k = self.active_top_k().min(candidates.len()).max(1);
        let slice = &candidates[..k];
        let inv_temp = 1.0 / self.params.temperature.max(1e-5);
        let max_logit = slice[0].logit;
        let mut probs: Vec<(u32, f32)> = slice
            .iter()
            .map(|c| (c.token_id, ((c.logit - max_logit) * inv_temp).exp()))
            .collect();
        let sum: f32 = probs.iter().map(|p| p.1).sum();
        if sum > 0.0 {
            for p in &mut probs {
                p.1 /= sum;
            }
        }
        // top_p / min_p filter
        let min_prob = probs[0].1 * self.params.min_p;
        let mut cum = 0.0_f32;
        let mut kept = 0_usize;
        for &(_, pr) in &probs {
            if kept > 0 && (cum >= self.params.top_p || pr < min_prob) {
                break;
            }
            cum += pr;
            kept += 1;
        }
        let r = self.next_f32() * cum;
        let mut acc = 0.0_f32;
        let mut chosen = probs[0].0;
        for &(id, pr) in &probs[..kept] {
            acc += pr;
            if r <= acc {
                chosen = id;
                break;
            }
        }
        SamplingResult {
            token_id: chosen,
            is_eos: self.is_eos(chosen),
        }
    }

}

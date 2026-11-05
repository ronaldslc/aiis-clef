use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum QuestionType {
    Noul,
    Choice,
    Score,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Question {
    #[serde(rename = "type")]
    pub kind: QuestionType,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub criteria: Option<Value>,
}

impl Question {
    pub fn criteria_is_usable(&self) -> bool {
        if self.kind == QuestionType::Noul {
            return true;
        }
        match &self.criteria {
            Some(Value::Object(m)) => !m.is_empty(),
            Some(Value::Array(a)) => !a.is_empty(),
            _ => false,
        }
    }
}

fn deserialize_questions<'de, D>(deserializer: D) -> Result<Vec<(String, Question)>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    struct QuestionsVisitor;

    impl<'de> serde::de::Visitor<'de> for QuestionsVisitor {
        type Value = Vec<(String, Question)>;

        fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("a map of question id to question")
        }

        fn visit_map<A>(self, mut access: A) -> Result<Self::Value, A::Error>
        where
            A: serde::de::MapAccess<'de>,
        {
            let mut out = Vec::with_capacity(access.size_hint().unwrap_or(0));
            while let Some(entry) = access.next_entry::<String, Question>()? {
                out.push(entry);
            }
            Ok(out)
        }
    }

    deserializer.deserialize_map(QuestionsVisitor)
}

#[derive(Debug, Clone, Deserialize)]
pub struct SystemOneRequest {
    pub state: String,

    #[serde(deserialize_with = "deserialize_questions")]
    pub questions: Vec<(String, Question)>,

    #[serde(default)]
    pub images: Option<Value>,
    #[serde(default)]
    pub videos: Option<Value>,
    #[serde(default)]
    pub model: Option<String>,
}

impl SystemOneRequest {
    pub fn validate(&self) -> Result<(), String> {
        if self.questions.is_empty() {
            return Err("at least one question is required".into());
        }
        for (id, q) in &self.questions {
            if !q.criteria_is_usable() {
                return Err(format!("{id}: criteria must not be empty"));
            }
        }

        if self.images.is_some() || self.videos.is_some() {
            return Err("media input is not supported by this backend".into());
        }
        Ok(())
    }

    pub fn model_name(&self) -> String {
        self.model
            .clone()
            .or_else(|| std::env::var("CLEF_MODEL_NAME").ok())
            .unwrap_or_else(|| "clef-flash".into())
    }
}

fn option_ids(kind: QuestionType, criteria: Option<&Value>) -> Vec<String> {
    match kind {
        QuestionType::Noul => vec!["true".into(), "false".into()],
        QuestionType::Choice => match criteria {
            Some(Value::Object(m)) => {
                let mut keys: Vec<String> = m.keys().cloned().collect();
                keys.sort();
                keys
            }
            Some(Value::Array(a)) => a.iter().map(|v| value_as_key(v)).collect(),
            _ => Vec::new(),
        },
        QuestionType::Score => (0..criteria.map(value_len).unwrap_or(0)).map(|i| i.to_string()).collect(),
    }
}

fn value_len(v: &Value) -> usize {
    match v {
        Value::Object(m) => m.len(),
        Value::Array(a) => a.len(),
        _ => 0,
    }
}

fn value_as_key(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

fn round4(x: f32) -> f64 {
    ((x as f64) * 10_000.0).round() / 10_000.0
}

pub fn systemone_answer(question: &Question, probabilities: &Probabilities) -> Value {
    let ids = option_ids(question.kind, question.criteria.as_ref());
    let get = |id: &str| probabilities.get(id).copied().unwrap_or(0.0);

    match question.kind {
        QuestionType::Noul => serde_json::json!({
            "type": "noul",
            "noul": round4(get("true")),
        }),
        QuestionType::Choice => {
            let best = ids
                .iter()
                .max_by(|a, b| get(a).partial_cmp(&get(b)).unwrap_or(std::cmp::Ordering::Equal))
                .cloned()
                .unwrap_or_default();
            let probs: serde_json::Map<String, Value> =
                ids.iter().map(|id| (id.clone(), Value::from(round4(get(id))))).collect();
            serde_json::json!({
                "type": "choice",
                "choice": best,
                "confidence": round4(get(&best)),
                "probabilities": Value::Object(probs),
            })
        }
        QuestionType::Score => {
            let mut score = 0.0f64;
            for (index, id) in ids.iter().enumerate() {
                score += index as f64 * get(id) as f64;
            }
            let confidence = ids.iter().map(|id| get(id)).fold(0.0f32, f32::max);
            let probs: serde_json::Map<String, Value> =
                ids.iter().map(|id| (id.clone(), Value::from(round4(get(id))))).collect();
            let legend: serde_json::Map<String, Value> = ids
                .iter()
                .enumerate()
                .map(|(index, id)| {
                    let label = match question.criteria.as_ref() {
                        Some(Value::Object(m)) => m.get(id).cloned(),
                        Some(Value::Array(a)) => a.get(index).cloned(),
                        _ => None,
                    };
                    (id.clone(), label.unwrap_or(Value::Null))
                })
                .collect();
            serde_json::json!({
                "type": "score",
                "score": (score * 10_000.0).round() / 10_000.0,
                "confidence": round4(confidence),
                "legend": Value::Object(legend),
                "probabilities": Value::Object(probs),
            })
        }
    }
}

pub type Probabilities = BTreeMap<String, f32>;

#[derive(Debug, Clone, Serialize)]
pub struct Usage {
    pub input_tokens: usize,
    pub output_tokens: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct SystemOneResponse {
    pub model: String,
    pub answers: BTreeMap<String, Value>,
    pub usage: Usage,
    pub latency_s: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct Health {
    pub ok: bool,
    pub model: String,
    pub version: String,
    pub device: String,
    pub backend: String,
    pub quant: String,
    pub checkpoint: String,
    pub max_batch: usize,
    pub max_tokens: usize,
    pub graphs: usize,
    pub graphs_chunked: usize,
    pub captures_pending: usize,
    pub head_graphs: usize,
    pub chunk_tokens: usize,
    pub att_keys: usize,
    pub stream_priority: i32,
    pub kv_tokens: usize,
    pub kv_mib: usize,
    pub kv_slots: usize,
    pub chunked_passes: usize,
    pub chunked_width_max: usize,
    pub chunked_passes_sharing: usize,
    pub graph_pool_mib: usize,
    pub refused_over_ceiling: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn health_serialises_the_graph_and_the_cache() {
        let h = Health {
            ok: true,
            model: "clef-flash".into(),
            version: "test".into(),
            device: "RTX".into(),
            backend: "rust-cuda".into(),
            quant: "nvfp4".into(),
            checkpoint: "clef-flash-nvfp4".into(),
            max_batch: 4,
            max_tokens: 32768,
            graphs: 66,
            graphs_chunked: 33,
            kv_slots: 1,
            chunked_passes: 0,
            chunked_width_max: 0,
            chunked_passes_sharing: 0,
            graph_pool_mib: 0,
            captures_pending: 2,
            head_graphs: 1,
            chunk_tokens: 2048,
            att_keys: 4,
            stream_priority: -5,
            kv_tokens: 32768,
            kv_mib: 1074,
            refused_over_ceiling: 3,
        };
        let v = serde_json::to_value(&h).expect("serialise");
        for key in ["graphs", "graphs_chunked", "captures_pending", "head_graphs", "chunk_tokens",

                    "att_keys",
                    "kv_tokens", "kv_mib", "refused_over_ceiling",

                    "kv_slots", "chunked_passes", "chunked_width_max", "chunked_passes_sharing",

                    "checkpoint"] {
            assert!(v.get(key).is_some(), "/health must report `{key}`: {v}");
        }
        assert_eq!(v["graphs_chunked"], 33);
        assert_eq!(v["kv_tokens"], 32768);
        assert_eq!(v["kv_slots"], 1, "the default that turns batching off, and it must be visible");
    }

    fn score_q() -> Question {
        Question {
            kind: QuestionType::Score,
            instructions: None,
            criteria: Some(json!({
                "0": "Does not answer.",
                "1": "Related but does not answer.",
                "2": "Answers part.",
                "3": "Directly answers.",
            })),
        }
    }

    #[test]
    fn a_score_is_the_expected_level() {
        let probs: Probabilities = [("0", 0.1f32), ("1", 0.2), ("2", 0.3), ("3", 0.4)]
            .into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect();
        let answer = systemone_answer(&score_q(), &probs);
        assert_eq!(answer["type"], json!("score"));
        assert_eq!(answer["score"], json!(2.0));
        assert_eq!(answer["confidence"], json!(0.4));
        assert_eq!(answer["legend"]["3"], json!("Directly answers."));
    }

    #[test]
    fn a_score_spans_the_full_rubric() {
        let lowest: Probabilities = [("0", 1.0f32)].into_iter().map(|(k, v)| (k.to_string(), v)).collect();
        assert_eq!(systemone_answer(&score_q(), &lowest)["score"], json!(0.0));
        let highest: Probabilities = [("3", 1.0f32)].into_iter().map(|(k, v)| (k.to_string(), v)).collect();
        assert_eq!(systemone_answer(&score_q(), &highest)["score"], json!(3.0));
    }

    #[test]
    fn a_choice_picks_the_most_probable_option() {
        let q = Question {
            kind: QuestionType::Choice,
            instructions: None,
            criteria: Some(json!(["alpha", "beta", "gamma"])),
        };
        let probs: Probabilities = [("alpha", 0.2f32), ("beta", 0.7), ("gamma", 0.1)]
            .into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect();
        let answer = systemone_answer(&q, &probs);
        assert_eq!(answer["choice"], json!("beta"));
        assert_eq!(answer["confidence"], json!(0.7));
        assert_eq!(answer["probabilities"]["alpha"], json!(0.2));
    }

    #[test]
    fn a_noul_reads_the_true_option() {
        let q = Question { kind: QuestionType::Noul, instructions: None, criteria: None };
        let probs: Probabilities =
            [("true", 0.25f32), ("false", 0.75)].into_iter().map(|(k, v)| (k.to_string(), v)).collect();
        let answer = systemone_answer(&q, &probs);
        assert_eq!(answer["type"], json!("noul"));
        assert_eq!(answer["noul"], json!(0.25));
    }

    #[test]
    fn a_missing_option_is_zero_not_a_panic() {
        let probs: Probabilities = [("3", 1.0f32)].into_iter().map(|(k, v)| (k.to_string(), v)).collect();
        let answer = systemone_answer(&score_q(), &probs);
        assert_eq!(answer["probabilities"]["0"], json!(0.0));
    }

    #[test]
    fn validation_mirrors_the_server_contract() {
        let ok = SystemOneRequest {
            state: "text".into(),
            questions: [("q".to_string(), score_q())].into_iter().collect(),
            images: None,
            videos: None,
            model: None,
        };
        assert!(ok.validate().is_ok());

        let mut empty = ok.clone();
        empty.questions.clear();
        assert!(empty.validate().is_err());

        let mut no_criteria = ok.clone();
        no_criteria.questions = vec![(
            "q".into(),
            Question { kind: QuestionType::Score, instructions: None, criteria: None },
        )];
        assert!(no_criteria.validate().is_err());

        let mut noul = ok.clone();
        noul.questions = vec![(
            "q".into(),
            Question { kind: QuestionType::Noul, instructions: None, criteria: None },
        )];
        assert!(noul.validate().is_ok());

        let mut media = ok.clone();
        media.images = Some(json!(["x"]));
        assert!(media.validate().is_err());
    }
}

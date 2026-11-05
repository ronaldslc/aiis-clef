use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use anyhow::{bail, Result};
use serde_json::Value;
use tokenizers::Tokenizer;

use crate::types::{Question, QuestionType, SystemOneRequest};

pub const SYSTEM_PROMPT: &str = "Read the complete state and schema. Decide every field jointly. \
Each answer must be exactly one of that field's allowed options.";

const NOUL_TRUE: &str = "The proposition is true or the answer is yes.";
const NOUL_FALSE: &str = "The proposition is false or the answer is no.";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncodedQuestion {
    pub question_id: String,
    pub question_type: QuestionType,
    pub question_span: (usize, usize),
    pub option_spans: Vec<(usize, usize)>,
    pub option_ids: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct Encoded {
    pub input_ids: Vec<u32>,
    pub questions: Vec<EncodedQuestion>,
}

impl Encoded {
    pub fn len(&self) -> usize {
        self.input_ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.input_ids.is_empty()
    }
}

pub fn render(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        other => serde_json::to_string(&sort_keys(other)).unwrap_or_default(),
    }
}

fn sort_keys(value: &Value) -> Value {
    match value {
        Value::Object(map) => {
            let mut entries: Vec<(&String, &Value)> = map.iter().collect();
            entries.sort_by(|a, b| a.0.cmp(b.0));
            let mut out = serde_json::Map::with_capacity(entries.len());
            for (key, value) in entries {
                out.insert(key.clone(), sort_keys(value));
            }
            Value::Object(out)
        }
        Value::Array(items) => Value::Array(items.iter().map(sort_keys).collect()),
        other => other.clone(),
    }
}

pub fn question_options(question: &Question) -> Vec<(String, Option<Value>)> {
    match question.kind {
        QuestionType::Noul => {
            let mut pairs: Vec<(String, Option<Value>)> = vec![
                ("true".into(), Some(Value::String(NOUL_TRUE.into()))),
                ("false".into(), Some(Value::String(NOUL_FALSE.into()))),
            ];
            if let Some(Value::Object(overrides)) = &question.criteria {
                for (id, description) in pairs.iter_mut() {
                    if let Some(v) = overrides.get(id) {
                        *description = Some(v.clone());
                    }
                }
            }
            pairs
        }
        QuestionType::Choice => match &question.criteria {
            Some(Value::Object(m)) => {
                let mut keys: Vec<String> = m.keys().cloned().collect();
                keys.sort();
                keys.into_iter().map(|k| (k, None)).collect()
            }
            _ => Vec::new(),
        },
        QuestionType::Score => match &question.criteria {
            Some(Value::Object(m)) => {
                let mut keys: Vec<String> = m.keys().cloned().collect();
                keys.sort();
                keys.into_iter()
                    .enumerate()
                    .map(|(index, key)| (index.to_string(), Some(Value::String(key))))
                    .collect()
            }
            Some(Value::Array(a)) => a
                .iter()
                .enumerate()
                .map(|(index, v)| (index.to_string(), Some(v.clone())))
                .collect(),
            _ => Vec::new(),
        },
    }
}

fn tokens(tokenizer: &Tokenizer, text: &str) -> Result<Vec<u32>> {
    let encoding = tokenizer
        .encode(text, false)
        .map_err(|e| anyhow::anyhow!("tokenizer failed: {e}"))?;
    Ok(encoding.get_ids().to_vec())
}

pub struct Encoder {
    tokenizer: Tokenizer,
    cache: Mutex<HashMap<String, Arc<[u32]>>>,
}

impl Encoder {
    pub fn load(model_dir: &std::path::Path) -> Result<Self> {
        let path = model_dir.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&path)
            .map_err(|e| anyhow::anyhow!("cannot load {}: {e}", path.display()))?;
        Ok(Self { tokenizer, cache: Mutex::new(HashMap::new()) })
    }

    pub fn cache_size(&self) -> usize {
        self.cache.lock().map(|c| c.len()).unwrap_or(0)
    }

    fn tokens(&self, text: &str) -> Result<Arc<[u32]>> {
        if let Some(hit) = self.cache.lock().ok().and_then(|c| c.get(text).cloned()) {
            return Ok(hit);
        }
        let ids: Arc<[u32]> = tokens(&self.tokenizer, text)?.into();
        if let Ok(mut cache) = self.cache.lock() {
            cache.insert(text.to_string(), Arc::clone(&ids));
        }
        Ok(ids)
    }

    pub fn encode(&self, request: &SystemOneRequest) -> Result<Encoded> {
        encode_with(request, &|text| self.tokens(text))
    }
}

pub fn encode(tokenizer: &Tokenizer, request: &SystemOneRequest) -> Result<Encoded> {
    encode_with(request, &|text| tokens(tokenizer, text).map(Arc::from))
}

fn encode_with(
    request: &SystemOneRequest,
    tok: &dyn Fn(&str) -> Result<Arc<[u32]>>,
) -> Result<Encoded> {
    let mut schema_ids = tok("\n\nSCHEMA FIELDS:\n")?.to_vec();
    let mut questions = Vec::with_capacity(request.questions.len());
    for (index, (question_id, question)) in request.questions.iter().enumerate() {
        schema_ids.extend_from_slice(&tok(&format!(
            "\nFIELD {}\nID: {question_id}\nTYPE: {}\nINSTRUCTION: ",
            index + 1,
            type_name(question.kind)
        ))?);
        let question_start = schema_ids.len();

        let instruction = match question.instructions.as_deref() {
            Some(s) if !s.is_empty() => s.to_string(),
            _ => question_id.clone(),
        };
        schema_ids.extend_from_slice(&tok(&instruction)?);
        let question_end = schema_ids.len();

        schema_ids.extend_from_slice(&tok("\nALLOWED OPTIONS:\n")?);
        let mut option_spans = Vec::new();
        let mut option_ids = Vec::new();
        for (option_index, (option_id, description)) in
            question_options(question).into_iter().enumerate()
        {
            schema_ids.extend_from_slice(&tok(&format!("OPTION {}: ", option_index + 1))?);
            let option_start = schema_ids.len();
            let mut semantics = serde_json::Map::new();
            semantics.insert("option_id".into(), Value::String(option_id.clone()));
            if let Some(description) = description {
                semantics.insert("description".into(), description);
            }
            schema_ids.extend_from_slice(&tok(&render(&Value::Object(semantics)))?);
            option_spans.push((option_start, schema_ids.len()));
            option_ids.push(option_id);
            schema_ids.extend_from_slice(&tok("\n")?);
        }
        schema_ids.extend_from_slice(&tok("END FIELD\n")?);

        questions.push(EncodedQuestion {
            question_id: question_id.clone(),
            question_type: question.kind,
            question_span: (question_start, question_end),
            option_spans,
            option_ids,
        });
    }

    let mut input_ids = tok(&format!(
        "<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\nSTATE:\n"
    ))?
    .to_vec();
    let suffix_ids = tok(

        "\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:",
    )?;
    let state_ids = tok(&render(&Value::String(request.state.clone())))?;

    if input_ids.is_empty() || questions.is_empty() {
        bail!("record produced no model input or questions");
    }

    let schema_offset = input_ids.len() + state_ids.len();
    for question in questions.iter_mut() {
        question.question_span = (
            question.question_span.0 + schema_offset,
            question.question_span.1 + schema_offset,
        );
        for span in question.option_spans.iter_mut() {
            *span = (span.0 + schema_offset, span.1 + schema_offset);
        }
    }

    input_ids.extend_from_slice(&state_ids);
    input_ids.extend_from_slice(&schema_ids);
    input_ids.extend_from_slice(&suffix_ids);
    Ok(Encoded {
        input_ids,
        questions,
    })
}

fn type_name(kind: QuestionType) -> &'static str {
    match kind {
        QuestionType::Noul => "noul",
        QuestionType::Choice => "choice",
        QuestionType::Score => "score",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::QuestionType;
    use serde_json::json;

    fn score_question() -> Question {
        Question {
            kind: QuestionType::Score,
            instructions: Some("How well does the item answer?".into()),
            criteria: Some(json!({
                "0": "Does not answer.",
                "1": "Related but does not answer.",
                "2": "Answers part.",
                "3": "Directly answers.",
            })),
        }
    }

    #[test]
    fn render_matches_the_reference_json_rules() {
        assert_eq!(render(&json!("plain")), "plain");

        assert_eq!(render(&json!({"b": 1, "a": 2})), r#"{"a":2,"b":1}"#);

        assert_eq!(render(&json!({"k": "café"})), r#"{"k":"café"}"#);
    }

    #[test]
    fn score_options_carry_the_key_as_the_description() {
        let options = question_options(&score_question());
        assert_eq!(options.len(), 4);
        let ids: Vec<&str> = options.iter().map(|(id, _)| id.as_str()).collect();
        assert_eq!(ids, ["0", "1", "2", "3"]);

        assert_eq!(options[0].1, Some(json!("0")));
        assert_eq!(options[3].1, Some(json!("3")));
    }

    #[test]
    fn noul_always_offers_true_and_false() {
        let plain = Question { kind: QuestionType::Noul, instructions: None, criteria: None };
        let options = question_options(&plain);
        assert_eq!(options[0].0, "true");
        assert_eq!(options[1].0, "false");
        assert_eq!(options[0].1, Some(json!(NOUL_TRUE)));

        let overridden = Question {
            kind: QuestionType::Noul,
            instructions: None,
            criteria: Some(json!({"true": "custom"})),
        };
        let options = question_options(&overridden);
        assert_eq!(options[0].0, "true", "the id stays true; only the description is replaced");
        assert_eq!(options[0].1, Some(json!("custom")));
        assert_eq!(options[1].1, Some(json!(NOUL_FALSE)), "the untouched default survives");
    }

    #[test]
    fn choice_uses_the_criteria_keys() {
        let q = Question {
            kind: QuestionType::Choice,
            instructions: None,
            criteria: Some(json!({"gamma": "g", "alpha": "a", "beta": "b"})),
        };
        let ids: Vec<String> = question_options(&q).into_iter().map(|(id, _)| id).collect();
        assert_eq!(ids, ["alpha", "beta", "gamma"]);
    }

    #[test]
    fn encoding_never_truncates_the_state() {
        let Some(tokenizer) = fixture_tokenizer() else { return };
        let long_state = "word ".repeat(5000);
        let request = SystemOneRequest {
            state: long_state.clone(),
            questions: [("q".to_string(), score_question())].into_iter().collect(),
            images: None,
            videos: None,
            model: None,
        };
        let encoded = encode(&tokenizer, &request).expect("encode");

        let state_only = tokens(&tokenizer, &long_state).expect("tokens").len();
        assert!(encoded.len() > state_only, "the state was cut");
        assert!(state_only > 4000, "the test state is not long enough to prove anything");
    }

    fn fixture_tokenizer() -> Option<Tokenizer> {
        let dir = std::env::var("CLEF_MODEL_DIR").unwrap_or_else(|_| "/models/clef-flash-nvfp4".into());
        let path = std::path::Path::new(&dir).join("tokenizer.json");
        if !path.exists() {
            eprintln!("SKIPPED: no tokenizer at {} — set CLEF_MODEL_DIR to run", path.display());
            return None;
        }
        Tokenizer::from_file(&path).ok()
    }

    fn fixture() -> Option<Value> {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/eval/clef_encode_fixture.json");
        let text = std::fs::read_to_string(path).ok()?;
        serde_json::from_str(&text).ok()
    }

    #[test]
    fn matches_the_reference_encoder_exactly() {
        let (Some(tokenizer), Some(fixture)) = (fixture_tokenizer(), fixture()) else {
            eprintln!("SKIPPED: needs both the tokenizer and tests/eval/clef_encode_fixture.json");
            return;
        };
        for case in fixture["cases"].as_array().expect("cases") {
            let name = case["name"].as_str().expect("name");

            let request: SystemOneRequest =
                serde_json::from_str(case["request_json"].as_str().expect("request_json"))
                    .expect("request");

            let encoded = encode(&tokenizer, &request).expect("encode");

            assert_eq!(
                encoded.len(),
                case["input_tokens"].as_u64().expect("input_tokens") as usize,
                "{name}: token count differs from the reference"
            );

            if let Some(expected) = case["input_ids"].as_array() {
                let expected: Vec<u32> = expected.iter().map(|v| v.as_u64().unwrap() as u32).collect();
                if encoded.input_ids != expected {
                    let at = encoded
                        .input_ids
                        .iter()
                        .zip(&expected)
                        .position(|(a, b)| a != b)
                        .unwrap_or(0);
                    panic!(
                        "{name}: token ids differ, first at {at} — got {:?} want {:?}",
                        &encoded.input_ids[at.saturating_sub(4)..(at + 4).min(encoded.len())],
                        &expected[at.saturating_sub(4)..(at + 4).min(expected.len())]
                    );
                }
            }

            let expected_questions = case["questions"].as_array().expect("questions");
            assert_eq!(encoded.questions.len(), expected_questions.len(), "{name}: question count");
            for (got, want) in encoded.questions.iter().zip(expected_questions) {
                assert_eq!(got.question_id, want["id"].as_str().unwrap(), "{name}: question id");
                let span = want["span"].as_array().unwrap();
                assert_eq!(
                    got.question_span,
                    (span[0].as_u64().unwrap() as usize, span[1].as_u64().unwrap() as usize),
                    "{name}: instruction span for {}",
                    got.question_id
                );
                let want_options = want["options"].as_array().unwrap();
                assert_eq!(got.option_ids.len(), want_options.len(), "{name}: option count");
                for (i, option) in want_options.iter().enumerate() {
                    assert_eq!(got.option_ids[i], option["id"].as_str().unwrap(), "{name}: option id");
                    let os = option["span"].as_array().unwrap();
                    assert_eq!(
                        got.option_spans[i],
                        (os[0].as_u64().unwrap() as usize, os[1].as_u64().unwrap() as usize),
                        "{name}: option span {i} for {}",
                        got.question_id
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod perf {
    use super::*;
    use crate::types::QuestionType;
    use serde_json::json;
    use std::time::Instant;

    fn realistic_request() -> SystemOneRequest {
        let state = format!(
            "User question: how does a caller's display name get resolved?\n\
             Items under review (untrusted source text):\n\n{}",
            "item:\nid: docs/auth.md#2\ntitle: Auth\nlocation: docs/auth.md\n\
             text: The session resolves the caller's display name at write time and records the \
             transition trail alongside it.\n\n"
                .repeat(8)
        );
        let questions: Vec<(String, Question)> = (0..8)
            .map(|i| {
                (
                    format!("rel_{i}"),
                    Question {
                        kind: QuestionType::Score,
                        instructions: Some(format!(
                            "How well does the text of item {i} answer the user question?"
                        )),
                        criteria: Some(json!({
                            "0": "Does not answer.",
                            "1": "Related but does not answer.",
                            "2": "Answers part.",
                            "3": "Directly answers.",
                        })),
                    },
                )
            })
            .chain(std::iter::once((
                "doc_0".to_string(),
                Question {
                    kind: QuestionType::Score,
                    instructions: Some(
                        "How well does the text of items 0 to 7 together answer?".into(),
                    ),
                    criteria: Some(json!({
                        "0": "Does not answer.", "1": "Related but does not answer.",
                        "2": "Answers part.", "3": "Directly answers.",
                    })),
                },
            )))
            .collect();
        SystemOneRequest { state, questions, images: None, videos: None, model: None }
    }

    #[test]
    #[ignore = "timing test; run explicitly with --release --ignored"]
    fn encoder_cost_breakdown() {
        let dir = std::env::var("CLEF_MODEL_DIR").unwrap_or_else(|_| "/models/clef-flash-nvfp4".into());
        let tokenizer = match Tokenizer::from_file(std::path::Path::new(&dir).join("tokenizer.json")) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("SKIPPED: no tokenizer in {dir} ({e})");
                return;
            }
        };
        let request = realistic_request();
        let iterations = 200;

        let _ = encode(&tokenizer, &request).expect("warmup");

        let mut samples = Vec::with_capacity(iterations);
        let mut tokens = 0usize;
        for _ in 0..iterations {
            let started = Instant::now();
            let encoded = encode(&tokenizer, &request).expect("encode");
            samples.push(started.elapsed());
            tokens = encoded.len();
        }
        samples.sort();
        let p50 = samples[iterations / 2];
        let p90 = samples[iterations * 9 / 10];
        let total: std::time::Duration = samples.iter().sum();
        let mean = total / iterations as u32;

        let raw: String = std::iter::once(request.state.clone())
            .chain(request.questions.iter().map(|(id, q)| {
                format!("{id}{}", q.instructions.clone().unwrap_or_default())
            }))
            .collect();
        let t0 = Instant::now();
        for _ in 0..iterations {
            let _ = tokenizer.encode(raw.as_str(), false).expect("raw");
        }
        let raw_p50 = t0.elapsed() / iterations as u32;

        let encoder = Encoder::load(std::path::Path::new(&dir)).expect("encoder");
        let _ = encoder.encode(&request).expect("warmup");
        let mut cached = Vec::with_capacity(iterations);
        for _ in 0..iterations {
            let started = Instant::now();
            let _ = encoder.encode(&request).expect("encode");
            cached.push(started.elapsed());
        }
        cached.sort();
        let cached_p50 = cached[iterations / 2];
        let cached_mean = cached.iter().sum::<std::time::Duration>() / iterations as u32;
        let distinct = encoder.cache_size();

        println!("  tokens per request : {tokens}");
        println!("  encode p50 (no memo): {:?}", p50);
        println!("  encode p50 (memo)   : {:?}  mean {:?}", cached_p50, cached_mean);
        println!("  memo speedup        : {:.2}x", p50.as_secs_f64() / cached_p50.as_secs_f64());
        println!("  distinct fragments  : {distinct} memoised");
        println!("  encode p50         : {:?}  p90 {:?}  mean {:?}", p50, p90, mean);
        println!("  throughput         : {:.0} requests/s   {:.1} Mtok/s",
            1.0 / mean.as_secs_f64(),
            tokens as f64 / mean.as_secs_f64() / 1e6);
        println!("  raw encode of the same text, p50 : {:?}  ({:.0}% of the above)",
            raw_p50,
            100.0 * raw_p50.as_secs_f64() / mean.as_secs_f64());
    }

    #[test]
    #[ignore = "timing test; run explicitly with --release --ignored"]
    fn encoder_stays_within_budget() {
        let dir = std::env::var("CLEF_MODEL_DIR").unwrap_or_else(|_| "/models/clef-flash-nvfp4".into());
        let tokenizer = match Tokenizer::from_file(std::path::Path::new(&dir).join("tokenizer.json")) {
            Ok(t) => t,
            Err(_) => {
                eprintln!("SKIPPED: no tokenizer in {dir}");
                return;
            }
        };
        let request = realistic_request();
        let _ = encode(&tokenizer, &request).expect("warmup");
        let started = Instant::now();
        let rounds = 50;
        for _ in 0..rounds {
            let _ = encode(&tokenizer, &request).expect("encode");
        }
        let per_request = started.elapsed() / rounds;

        assert!(
            per_request < std::time::Duration::from_millis(20),
            "encode took {per_request:?} per request, over the 20 ms budget"
        );
    }
}

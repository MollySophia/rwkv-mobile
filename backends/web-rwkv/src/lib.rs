use std::{
    ffi::{c_char, CStr},
    path::Path,
    sync::{Arc, RwLock},
};

use anyhow::Result;
use half::f16;
use itertools::Itertools;
use memmap2::Mmap;
use safetensors::SafeTensors;
use tokio::fs::File;
use web_rwkv::{
    context::{Context, ContextBuilder, InstanceExt},
    runtime::{
        infer::{InferInput, InferInputBatch, InferOption, InferOutput},
        loader::Loader,
        model::{
            Build, ContextAutoLimits, ModelBuilder, ModelInfo, ModelRuntime, ModelVersion, Quant,
            State,
        },
        softmax::softmax_one,
        v4, v5, v6, JobRuntime,
    },
    wgpu,
};

static RUNTIME: RwLock<Option<Runtime>> = RwLock::new(None);

#[derive(Clone)]
struct Runtime {
    runtime: JobRuntime<InferInput, InferOutput>,
    state: Arc<dyn State + Sync + Send + 'static>,
    context: Context,
    tokio: Arc<tokio::runtime::Runtime>,
}

async fn create_context(info: &ModelInfo) -> Result<Context> {
    let instance = wgpu::Instance::default();
    let adapter = instance
        .adapter(wgpu::PowerPreference::HighPerformance)
        .await?;
    let context = ContextBuilder::new(adapter)
        .auto_limits(info)
        .build()
        .await?;
    Ok(context)
}

fn load_runtime(
    model: impl AsRef<Path>,
    quant: usize,
    quant_nf4: usize,
    rescale: Option<usize>,
) -> Result<Runtime> {
    let tokio = Arc::new(tokio::runtime::Runtime::new()?);
    let _tokio = tokio.clone();

    _tokio.block_on(async move {
        let file = File::open(model).await?;
        let data = unsafe { Mmap::map(&file)? };

        let model = SafeTensors::deserialize(&data)?;
        let info = Loader::info(&model)?;
        log::info!("{:#?}", info);

        let context = create_context(&info).await?;
        log::info!("{:#?}", context.adapter.get_info());

        let quant = (0..quant)
            .map(|layer| (layer, Quant::Int8))
            .chain((0..quant_nf4).map(|layer| (layer, Quant::NF4)))
            .collect();

        let builder = ModelBuilder::new(&context, model).quant(quant);
        let builder = match rescale {
            Some(rescale) => builder.rescale(rescale),
            None => builder,
        };
        let runtime = match info.version {
            ModelVersion::V4 => {
                let model = Build::<v4::Model>::build(builder).await?;
                let builder = v4::ModelRuntime::<f16>::new(model, 1);
                let state = Arc::new(builder.state());
                let runtime = JobRuntime::new(builder).await;
                Runtime {
                    runtime,
                    state,
                    context,
                    tokio,
                }
            }
            ModelVersion::V5 => {
                let model = Build::<v5::Model>::build(builder).await?;
                let builder = v5::ModelRuntime::<f16>::new(model, 1);
                let state = Arc::new(builder.state());
                let runtime = JobRuntime::new(builder).await;
                Runtime {
                    runtime,
                    state,
                    context,
                    tokio,
                }
            }
            ModelVersion::V6 => {
                let model = Build::<v6::Model>::build(builder).await?;
                let builder = v6::ModelRuntime::<f16>::new(model, 1);
                let state = Arc::new(builder.state());
                let runtime = JobRuntime::new(builder).await;
                Runtime {
                    runtime,
                    state,
                    context,
                    tokio,
                }
            }
        };
        Ok(runtime)
    })
}

/// Initialize logger and RNG. Call this once before everything.
#[no_mangle]
pub extern "C" fn web_rwkv_init(seed: u64) {
    let _ = simple_logger::SimpleLogger::new()
        .with_level(log::LevelFilter::Warn)
        .with_module_level("web_rwkv", log::LevelFilter::Info)
        .with_module_level("web_rwkv_ffi", log::LevelFilter::Info)
        .init();
    fastrand::seed(seed);
}

/// Set the RNG seed.
#[no_mangle]
pub extern "C" fn web_rwkv_seed(seed: u64) {
    fastrand::seed(seed);
}

/// Load a runtime.
///
/// # Safety
///
/// The caller must ensure that `model` is valid.
#[no_mangle]
pub unsafe extern "C" fn web_rwkv_load(model: *const c_char, quant: usize, quant_nf4: usize) -> i32 {
    let model = unsafe { CStr::from_ptr(model).to_string_lossy().to_string() };
    match load_runtime(model, quant, quant_nf4, None) {
        Ok(runtime) => {
            let mut rt = RUNTIME.write().unwrap();
            rt.replace(runtime);
            return 0;
        }
        Err(err) => {
            log::error!("{err}");
            return -1;
        }
    }
}

/// Load a runtime with `rescale` layers specified.
///
/// # Safety
///
/// The caller must ensure that `model` is valid.
#[no_mangle]
pub unsafe extern "C" fn web_rwkv_load_with_rescale(
    model: *const c_char,
    quant: usize,
    quant_nf4: usize,
    rescale: usize,
) -> i32 {
    let model = unsafe { CStr::from_ptr(model).to_string_lossy().to_string() };
    match load_runtime(model, quant, quant_nf4, Some(rescale)) {
        Ok(runtime) => {
            let mut rt = RUNTIME.write().unwrap();
            rt.replace(runtime);
            return 0;
        }
        Err(err) => {
            log::error!("{err}");
            return -1;
        }
    }
}

/// Clear the model state.
#[no_mangle]
pub extern "C" fn web_rwkv_clear_state() {
    let runtime = {
        let runtime = RUNTIME.read().unwrap();
        let Some(runtime) = runtime.clone() else {
            log::error!("runtime not loaded");
            return;
        };
        runtime
    };
    let tensor = runtime.state.init();
    let _ = runtime.state.load(tensor, 0);
}

/// Get the model state.
// #[no_mangle]
// pub extern "C" fn get_state(output: *mut f32) -> i32 {
//     let runtime = {
//         let runtime = RUNTIME.read().unwrap();
//         let Some(runtime) = runtime.clone() else {
//             log::error!("runtime not loaded");
//             return std::ptr::null_mut();
//         };
//         runtime
//     };
//     if output.is_null() {
//         log::error!("output buffer cannot be null");
//         return -1;
//     }


// }

/// Generate the next token prediction given the input tokens and a sampler.
///
/// # Safety
///
/// The caller must ensure that `tokens` is valid and `len` does not exceed the actual length of `tokens`.
#[no_mangle]
pub unsafe extern "C" fn web_rwkv_infer(tokens: *const u16, len: usize, sampler: Sampler) -> u16 {
    let runtime = {
        let runtime = RUNTIME.read().unwrap();
        let Some(runtime) = runtime.clone() else {
            log::error!("runtime not loaded");
            return 0;
        };
        runtime
    };

    let tokens: &[u16] = unsafe { std::slice::from_raw_parts(tokens, len) };
    if tokens.is_empty() {
        log::error!("input cannot be empty");
        return 0;
    }

    let tokio = runtime.tokio.clone();
    tokio.block_on(async move {
        let context = &runtime.context;
        let mut inference = Some(InferInput::new(
            vec![InferInputBatch {
                tokens: tokens.to_vec(),
                option: InferOption::Last,
            }],
            128,
        ));
        let output = loop {
            let input = inference.take().unwrap();
            let (input, InferOutput(output)) = runtime.runtime.infer(input).await;
            let output = output[0].0.clone();
            inference.replace(input);

            if output.size() > 0 {
                let output = softmax_one(context, output).await.expect("softmax failed");
                break output.to_vec();
            }
        };
        sampler.sample(&output)
    })
}

/// Generate the next token probabilities given the input tokens.
///
/// # Safety
///
/// The caller must ensure that `tokens` is valid and `len` does not exceed the actual length of `tokens`.
#[no_mangle]
pub unsafe extern "C" fn web_rwkv_infer_logits(tokens: *const u16, len: usize, logits: *mut f32, logits_len: usize) -> i32 {
    let runtime = {
        let runtime = RUNTIME.read().unwrap();
        let Some(runtime) = runtime.clone() else {
            log::error!("runtime not loaded");
            return -1;
        };
        runtime
    };

    let tokens: &[u16] = unsafe { std::slice::from_raw_parts(tokens, len) };
    if tokens.is_empty() {
        log::error!("input cannot be empty");
        return -1;
    }

    if logits.is_null() {
        log::error!("output buffer cannot be null");
        return -1;
    }

    let tokio = runtime.tokio.clone();
    tokio.block_on(async move {
        let mut inference = Some(InferInput::new(
            vec![InferInputBatch {
                tokens: tokens.to_vec(),
                option: InferOption::Last,
            }],
            128,
        ));
        let output = loop {
            let input = inference.take().unwrap();
            let (input, InferOutput(output)) = runtime.runtime.infer(input).await;
            let output = output[0].0.clone();
            inference.replace(input);

            if output.size() > 0 {
                break output;
            }
        };

        if output.size() != (logits_len * 4) {
            log::error!("output buffer size mismatch");
            log::error!("expected: {}", logits_len * 4);
            log::error!("actual: {}", output.size());
            return -1;
        }

        std::ptr::copy_nonoverlapping(output.as_ptr(), logits, logits_len);
        return 0;
    })
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Sampler {
    pub temp: f32,
    pub top_p: f32,
    pub top_k: usize,
}

impl Default for Sampler {
    fn default() -> Self {
        Self {
            temp: 1.0,
            top_p: 0.5,
            top_k: 128,
        }
    }
}

impl Sampler {
    pub fn sample(&self, probs: &[f32]) -> u16 {
        let sorted: Vec<_> = probs
            .iter()
            .copied()
            .enumerate()
            .sorted_unstable_by(|(_, x), (_, y)| x.total_cmp(y).reverse())
            .take(self.top_k.max(1))
            .scan((0, 0.0, 0.0), |(_, cum, _), (id, x)| {
                if *cum > self.top_p {
                    None
                } else {
                    *cum += x;
                    Some((id, *cum, x))
                }
            })
            .map(|(id, _, x)| (id, x.powf(1.0 / self.temp)))
            .collect();

        let sum: f32 = sorted.iter().map(|(_, x)| x).sum();
        let sorted: Vec<_> = sorted
            .into_iter()
            .map(|(id, x)| (id, x / sum))
            .scan((0, 0.0), |(_, cum), (id, x)| {
                *cum += x;
                Some((id, *cum))
            })
            .collect();

        let rand = fastrand::f32();
        let token = sorted
            .into_iter()
            .find_or_first(|&(_, cum)| rand <= cum)
            .map(|(id, _)| id)
            .unwrap_or_default();
        token as u16
    }
}

//! Weight-only quantization of the model's `Linear` layers.
//!
//! VoxCPM2's weights are 4.37 GB of BF16, which the `wgpu` backend upcasts
//! to **8.74 GB** of F32 because WGSL has no BF16 type. That is the single
//! biggest barrier to running this anywhere: it demands a 12 GB card.
//! Quantizing the `Linear` weights to 8-bit takes GPU residency to about
//! **2.5 GB** and, because the matmuls move a quarter of the bytes, also
//! runs faster.
//!
//! Nothing about the model has to change. A burn `Tensor` can hold a
//! quantized primitive, and `matmul` dispatches to `q_matmul`, which
//! dequantizes inside the kernel rather than materializing a float copy.
//! So this is purely a transform over module parameters.
//!
//! # What is deliberately left alone
//!
//! * **Embeddings.** `q_gather` is `unimplemented!()` on the cubecl
//!   backend, so a quantized embedding panics at inference. At
//!   73448 x 2048 it is also 301 MB, which is the floor this approach
//!   cannot go below.
//! * **Convolutions.** The AudioVAE is `Conv1d`/`ConvTranspose1d`, and
//!   there is no quantized convolution kernel.
//! * **Norm and scale parameters.** 1-D and negligible.
//! * **Anything whose innermost dimension is not a multiple of
//!   `32 / bits`.** [`QuantStore::PackedU32`] packs that many values into
//!   each `u32` along the last axis and panics with "Can't store in u32"
//!   otherwise. VoxCPM2 really has such a layer — the stop head projects
//!   to 2 classes.
//!
//! # Precision
//!
//! Measured against a float matmul on `[1,2048] x [2048,2048]`
//! (`examples/quant_probe.rs`): Q8 gives ~0.4 % relative L2 error, Q4
//! ~7 %. Q4 additionally does not currently run on this model — see
//! [`QuantPreset`].

use burn::module::{Module, ModuleMapper, ModuleVisitor, Param};
use burn::prelude::*;
use burn::tensor::DType;
use burn::tensor::quantization::{QuantLevel, QuantScheme, QuantStore, QuantValue};

/// A quantization setting that can be named from a config file or a URL.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantPreset {
    /// 8-bit symmetric, per tensor. The only preset verified to run this
    /// model end to end.
    Q8,
    /// 4-bit symmetric, per tensor.
    ///
    /// Quantization itself succeeds and would put the checkpoint at
    /// ~1.45 GB, but inference currently fails: `cubek-matmul`'s `auto`
    /// strategy tries cooperative-matrix (unavailable on WebGPU), falls
    /// back to `SimpleUnit` with a tile selection that needs 96 KB of
    /// shared memory against a 48 KB hardware limit, and unwraps the
    /// error rather than trying a smaller tile.
    Q4,
    /// 4-bit symmetric with per-block scales. Same blocker as [`Self::Q4`].
    Q4Block(u8),
}

impl QuantPreset {
    /// Parse a preset name: `q8`, `q4`, `q4b32`, `q4b64`, `q4b128`.
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "q8" => Some(Self::Q8),
            "q4" => Some(Self::Q4),
            "q4b32" => Some(Self::Q4Block(32)),
            "q4b64" => Some(Self::Q4Block(64)),
            "q4b128" => Some(Self::Q4Block(128)),
            _ => None,
        }
    }

    /// The burn scheme this preset stands for.
    ///
    /// `PackedU32` throughout: WGSL has no 8-bit or 4-bit type, so
    /// quantized values have to be packed into `u32`.
    pub fn scheme(self) -> QuantScheme {
        let base = QuantScheme::default().with_store(QuantStore::PackedU32(0));
        match self {
            Self::Q8 => base.with_value(QuantValue::Q8S),
            Self::Q4 => base.with_value(QuantValue::Q4S),
            Self::Q4Block(n) => base
                .with_value(QuantValue::Q4S)
                .with_level(QuantLevel::block([n])),
        }
    }

    /// Bits per weight.
    pub fn bits(self) -> usize {
        self.scheme().value.size_bits()
    }

    /// Whether this preset is known to run inference on this model.
    pub fn is_supported(self) -> bool {
        matches!(self, Self::Q8)
    }
}

/// What one quantization pass converted.
///
/// Deliberately minimal: a pass only knows what *it* changed, and the
/// streaming loader runs one per batch. For the state of the whole model,
/// use [`survey`] — summing per-pass counters double-counts everything a
/// pass declined to touch, which is most of the model most of the time.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct QuantStats {
    /// Tensors converted by this pass.
    pub quantized: usize,
    /// Parameters converted by this pass.
    pub quantized_elems: usize,
    /// `Linear` weights big enough to be worth converting but whose last
    /// dimension `PackedU32` cannot pack.
    pub unpackable: usize,
}

impl QuantStats {
    /// Add another pass's conversions.
    pub fn add(&mut self, other: QuantStats) {
        self.quantized += other.quantized;
        self.quantized_elems += other.quantized_elems;
        self.unpackable += other.unpackable;
    }
}

/// The measured state of a model's parameters.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct QuantSurvey {
    /// Parameters stored quantized.
    pub quantized_params: usize,
    /// Bytes those occupy, including their scales.
    pub quantized_bytes: usize,
    /// Parameters still in floating point.
    pub float_params: usize,
    /// Bytes those occupy.
    pub float_bytes: usize,
    /// Number of quantized tensors.
    pub quantized_tensors: usize,
    /// Number of float tensors.
    pub float_tensors: usize,
}

impl QuantSurvey {
    /// Total parameter bytes resident on the device.
    pub fn total_bytes(&self) -> usize {
        self.quantized_bytes + self.float_bytes
    }

    /// Total in MiB.
    pub fn total_mib(&self) -> f64 {
        self.total_bytes() as f64 / 1048576.0
    }
}

struct Surveyor {
    survey: QuantSurvey,
}

impl<B: Backend> ModuleVisitor<B> for Surveyor {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
        let tensor = param.val();
        let elems = tensor.shape().num_elements();
        match tensor.dtype() {
            DType::QFloat(scheme) => {
                self.survey.quantized_tensors += 1;
                self.survey.quantized_params += elems;
                let value_bits = scheme.value.size_bits();
                // Plus the scales: one per tensor, or one per block.
                let scales = match scheme.level {
                    QuantLevel::Tensor => 1,
                    QuantLevel::Block(b) => {
                        let block: usize = b.num_elements();
                        elems.div_ceil(block.max(1))
                    }
                };
                let scale_bytes = match scheme.param {
                    burn::tensor::quantization::QuantParam::F32 => 4,
                    _ => 2,
                };
                self.survey.quantized_bytes += elems * value_bits / 8 + scales * scale_bytes;
            }
            other => {
                self.survey.float_tensors += 1;
                self.survey.float_params += elems;
                self.survey.float_bytes += elems * dtype_bytes(other);
            }
        }
    }
}

fn dtype_bytes(dt: DType) -> usize {
    match dt {
        DType::F64 | DType::I64 | DType::U64 => 8,
        DType::F32 | DType::Flex32 | DType::I32 | DType::U32 => 4,
        DType::F16 | DType::BF16 | DType::I16 | DType::U16 => 2,
        _ => 1,
    }
}

/// Measure how a model's parameters are actually stored, in one pass.
///
/// This is the number to report: it counts each parameter exactly once and
/// reads the real dtype off each tensor, rather than trusting a tally kept
/// by whatever did the converting.
pub fn survey<B: Backend, M: Module<B>>(model: &M) -> QuantSurvey {
    let mut v = Surveyor {
        survey: QuantSurvey::default(),
    };
    model.visit(&mut v);
    v.survey
}

struct LinearQuantizer {
    scheme: QuantScheme,
    min_elems: usize,
    /// Container types from `enter_module`, so `Struct:Linear` can be
    /// targeted exactly instead of inferred from tensor shapes.
    stack: Vec<&'static str>,
    stats: QuantStats,
}

impl<B: Backend> ModuleMapper<B> for LinearQuantizer {
    fn enter_module(&mut self, _name: &str, container_type: &str) {
        // Only this one matters, and keeping a &'static str avoids an
        // allocation per module on a tree walked once per load batch.
        self.stack.push(if container_type == "Struct:Linear" {
            "Struct:Linear"
        } else {
            ""
        });
    }

    fn exit_module(&mut self, _name: &str, _container_type: &str) {
        self.stack.pop();
    }

    fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
        let in_linear = self.stack.iter().any(|c| *c == "Struct:Linear");
        let (id, tensor, mapper) = param.consume();

        // Idempotent: the streaming loader runs this after every batch, so
        // most tensors it sees have already been converted.
        if matches!(tensor.dtype(), DType::QFloat(_)) {
            return Param::from_mapped_value(id, tensor, mapper);
        }

        let dims = tensor.shape().dims;
        let elems = tensor.shape().num_elements();
        let per_u32 = 32 / self.scheme.value.size_bits();
        let packable = dims.last().is_some_and(|last| last % per_u32 == 0);
        let eligible = D == 2 && in_linear && elems >= self.min_elems;

        if eligible && packable {
            self.stats.quantized += 1;
            self.stats.quantized_elems += elems;
            let q = tensor.quantize_dynamic(&self.scheme);
            Param::from_mapped_value(id, q, mapper)
        } else {
            if eligible && !packable {
                self.stats.unpackable += 1;
            }
            Param::from_mapped_value(id, tensor, mapper)
        }
    }
}

/// Default size floor: below this the per-tensor scale overhead and the
/// kernel-dispatch change are not worth it.
pub const DEFAULT_MIN_ELEMS: usize = 65536;

/// Quantize every eligible `Linear` weight in `model`, in place.
///
/// Safe to call repeatedly — already-quantized parameters are left alone
/// and counted in [`QuantStats::already`]. That is what lets the streaming
/// loader convert each batch as it lands, instead of holding the whole
/// model in floating point first and only then shrinking it, which would
/// peak at the full 8.74 GB and defeat the purpose.
pub fn quantize_linears<B: Backend, M: Module<B>>(
    model: M,
    preset: QuantPreset,
    min_elems: usize,
) -> (M, QuantStats) {
    let mut mapper = LinearQuantizer {
        scheme: preset.scheme(),
        min_elems,
        stack: Vec::new(),
        stats: QuantStats::default(),
    };
    let model = model.map(&mut mapper);
    (model, mapper.stats)
}

// ---------------------------------------------------------------------------
// Pre-quantized checkpoints
// ---------------------------------------------------------------------------
//
// Quantizing in the browser shrinks GPU residency but not the download: the
// 4.37 GB still has to arrive before anything can be quantized. Quantizing
// *offline* and shipping the result shrinks both.
//
// The format is plain safetensors so the existing Range-streaming loader
// works unchanged. Quantized weights cannot be typed natively — safetensors
// has no 4- or 8-bit quantized dtype — so each is stored as a flat `U8`
// blob holding exactly what `q_into_data` produced (packed values with the
// scales appended), and a manifest in `__metadata__` records the logical
// shape and the scheme. `q_from_data` consumes that layout directly, so
// loading never materializes a float copy.
//
// Names in the file are final module paths: the checkpoint is produced by
// loading the upstream weights normally — remapped, fused, and transposed
// by `PyTorchToBurnAdapter` — and then exporting. So it must be applied
// **without** that adapter, or the Linear weights get transposed twice.

/// What a pre-quantized checkpoint declares about itself.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct QuantManifest {
    /// Format version. Bumped if the layout changes.
    pub version: u32,
    /// The scheme every quantized tensor in the file uses.
    pub scheme: QuantScheme,
    /// Logical shape of each quantized tensor, keyed by module path. The
    /// safetensors entry itself is a flat `U8` blob, so the real shape has
    /// to be recorded here.
    pub quantized: std::collections::BTreeMap<String, Vec<usize>>,
}

/// The `__metadata__` key the manifest lives under.
pub const MANIFEST_KEY: &str = "voxcpm_quant";

/// Current format version.
pub const MANIFEST_VERSION: u32 = 1;

impl QuantManifest {
    /// Parse a manifest out of a safetensors `__metadata__` map.
    ///
    /// Returns `Ok(None)` for an ordinary, unquantized checkpoint.
    pub fn from_metadata(
        metadata: &std::collections::HashMap<String, String>,
    ) -> crate::Result<Option<Self>> {
        let Some(raw) = metadata.get(MANIFEST_KEY) else {
            return Ok(None);
        };
        let manifest: Self = serde_json::from_str(raw)?;
        if manifest.version != MANIFEST_VERSION {
            return Err(crate::Error::Unsupported(format!(
                "pre-quantized checkpoint is format version {} but this build \
                 understands {MANIFEST_VERSION}",
                manifest.version
            )));
        }
        Ok(Some(manifest))
    }

    /// Serialize for storing in `__metadata__`.
    pub fn to_json(&self) -> crate::Result<String> {
        Ok(serde_json::to_string(self)?)
    }

    /// Bytes a quantized tensor of this shape occupies in the file:
    /// packed values plus its scales.
    pub fn blob_len(&self, shape: &[usize]) -> usize {
        let elems: usize = shape.iter().product();
        let value_bits = self.scheme.value.size_bits();
        let scales = match self.scheme.level {
            QuantLevel::Tensor => 1,
            QuantLevel::Block(b) => {
                let block = b.num_elements().max(1);
                elems.div_ceil(block)
            }
        };
        let scale_bytes = match self.scheme.param {
            burn::tensor::quantization::QuantParam::F32 => 4,
            _ => 2,
        };
        elems * value_bits / 8 + scales * scale_bytes
    }
}

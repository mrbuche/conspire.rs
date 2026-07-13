mod gemm;
mod ldl;
mod lu;

pub use ldl::CscLdl;
pub use lu::CscLu;

#[cfg(target_arch = "x86_64")]
fn simd() -> bool {
    std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
}

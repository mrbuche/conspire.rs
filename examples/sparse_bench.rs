//! Times the sparse factorizations that run on the SIMD kernels, on a 3D
//! Laplacian, and prints one markdown table row per run.
//!
//! ```text
//! cargo run --release --features math --example sparse_bench [grid] [runs]
//! CONSPIRE_SIMD=scalar cargo run --release --features math --example sparse_bench
//! ```
use conspire::math::{Vector, sparse::CscMatrix};
use std::{hint::black_box, time::Instant};

fn laplacian(grid: usize) -> CscMatrix {
    let index = |i: usize, j: usize, k: usize| (i * grid + j) * grid + k;
    let mut pattern = Vec::new();
    (0..grid).for_each(|i| {
        (0..grid).for_each(|j| {
            (0..grid).for_each(|k| {
                let row = index(i, j, k);
                pattern.push((row, row));
                if i + 1 < grid {
                    pattern.push((row, index(i + 1, j, k)));
                    pattern.push((index(i + 1, j, k), row));
                }
                if j + 1 < grid {
                    pattern.push((row, index(i, j + 1, k)));
                    pattern.push((index(i, j + 1, k), row));
                }
                if k + 1 < grid {
                    pattern.push((row, index(i, j, k + 1)));
                    pattern.push((index(i, j, k + 1), row));
                }
            })
        })
    });
    let n = grid * grid * grid;
    let mut matrix = CscMatrix::from_pattern(n, n, pattern);
    matrix.fill(|i, j| if i == j { 6.5 } else { -1.0 });
    matrix
}

/// Median wall time of `runs` calls, in milliseconds, after one warm-up call.
fn median_ms(runs: usize, mut work: impl FnMut()) -> f64 {
    work();
    let mut times: Vec<f64> = (0..runs)
        .map(|_| {
            let start = Instant::now();
            work();
            start.elapsed().as_secs_f64() * 1e3
        })
        .collect();
    times.sort_by(f64::total_cmp);
    times[runs / 2]
}

fn backend() -> String {
    let arch = std::env::consts::ARCH;
    if std::env::var_os("CONSPIRE_SIMD").is_some_and(|value| value == "scalar") {
        format!("{arch} scalar (forced)")
    } else if cfg!(feature = "nightly") {
        format!("{arch} std::simd")
    } else if cfg!(target_arch = "x86_64") {
        format!("{arch} avx2 (if detected)")
    } else {
        format!("{arch} scalar")
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let grid: usize = args.next().map_or(24, |arg| arg.parse().expect("grid"));
    let runs: usize = args.next().map_or(5, |arg| arg.parse().expect("runs"));
    let matrix = laplacian(grid);
    let b: Vector = (0..matrix.height())
        .map(|i| (i % 13) as f64 - 6.0)
        .collect();
    let mut lu = matrix.lu_symbolic().expect("lu_symbolic");
    let lu_ms = median_ms(runs, || lu.refactor(black_box(&matrix)).expect("lu"));
    let lu_solve_ms = median_ms(runs, || {
        black_box(lu.solve(black_box(&b)));
    });
    let mut ldl = matrix.ldl_symbolic().expect("ldl_symbolic");
    let ldl_ms = median_ms(runs, || ldl.refactor(black_box(&matrix)).expect("ldl"));
    let ldl_solve_ms = median_ms(runs, || {
        black_box(ldl.solve(black_box(&b)));
    });
    println!(
        "| {} | {} | {:.1} | {:.2} | {:.1} | {:.2} |",
        backend(),
        matrix.height(),
        lu_ms,
        lu_solve_ms,
        ldl_ms,
        ldl_solve_ms
    );
}

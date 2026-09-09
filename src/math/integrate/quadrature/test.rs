use super::{gauss_hermite, gauss_laguerre, gauss_legendre};

/// A Gaussian integral evaluated three ways: directly on the line (Hermite),
/// on the half-line after folding (Laguerre), and on a truncated interval
/// (Legendre). All should agree with sqrt(pi).
#[test]
fn gaussian_three_ways() {
    let target = std::f64::consts::PI.sqrt();

    let (nodes, weights) = gauss_hermite(20);
    let hermite: f64 = nodes.iter().zip(&weights).map(|(_, &w)| w).sum();
    assert!((hermite - target).abs() < 1e-13);

    // int_0^inf x^{-1/2} e^{-x} dx = Gamma(1/2) = sqrt(pi)
    let (_, weights) = gauss_laguerre(24, -0.5);
    let laguerre: f64 = weights.iter().sum();
    assert!((laguerre - target).abs() < 1e-11);

    // int_{-L}^{L} e^{-x^2} dx -> sqrt(pi) for L large enough
    let l = 8.0;
    let (nodes, weights) = gauss_legendre(64);
    let legendre: f64 = nodes
        .iter()
        .zip(&weights)
        .map(|(&x, &w)| w * l * (-(l * x).powi(2)).exp())
        .sum();
    assert!((legendre - target).abs() < 1e-10, "{legendre}");
}

use super::sphere_product;

fn integrate(nodes: &[([f64; 3], f64)], f: impl Fn([f64; 3]) -> f64) -> f64 {
    nodes.iter().map(|&(u, w)| w * f(u)).sum()
}

const FOUR_PI: f64 = 4.0 * std::f64::consts::PI;

#[test]
fn weights_sum_to_four_pi() {
    let nodes = sphere_product(16, 32);
    assert!((integrate(&nodes, |_| 1.0) - FOUR_PI).abs() < 1e-12);
}

#[test]
fn nodes_are_unit_vectors() {
    for (u, w) in sphere_product(8, 16) {
        assert!((u[0] * u[0] + u[1] * u[1] + u[2] * u[2] - 1.0).abs() < 1e-13);
        assert!(w > 0.0);
    }
}

#[test]
fn low_order_monomials() {
    // int x_i^2 dOmega = 4pi/3 ;  int x_i^2 x_j^2 dOmega = 4pi/15 (i != j) ;
    // int x_i^4 dOmega = 4pi/5 ;  odd monomials -> 0
    let nodes = sphere_product(12, 24);
    for i in 0..3 {
        assert!((integrate(&nodes, |u| u[i].powi(2)) - FOUR_PI / 3.0).abs() < 1e-11);
        assert!((integrate(&nodes, |u| u[i].powi(4)) - FOUR_PI / 5.0).abs() < 1e-11);
        assert!(integrate(&nodes, |u| u[i].powi(3)).abs() < 1e-11);
        for j in i + 1..3 {
            assert!(
                (integrate(&nodes, |u| u[i].powi(2) * u[j].powi(2)) - FOUR_PI / 15.0).abs() < 1e-11
            );
        }
    }
}

#[test]
fn recovers_the_gaussian_tensor_identity() {
    // with s = sum u_i^2 / b_i (i.e. u . M . u for M = diag(1/b_i)),
    //   int u_i^2 s^{-5/2} dOmega = (4pi/3) sqrt(b_1 b_2 b_3) b_i;
    // normalizing prod(b) = 1 leaves (4pi/3) b_i.
    let b = [0.5_f64, 1.0, 2.0];
    let scale = (b[0] * b[1] * b[2]).cbrt();
    let b = b.map(|x| x / scale); // force det = 1
    let nodes = sphere_product(48, 96);
    for i in 0..3 {
        let got = integrate(&nodes, |u| {
            let s = u[0].powi(2) / b[0] + u[1].powi(2) / b[1] + u[2].powi(2) / b[2];
            u[i].powi(2) * s.powf(-2.5)
        });
        let exact = FOUR_PI / 3.0 * b[i];
        assert!(
            (got - exact).abs() < 1e-4 * exact,
            "i={i}: {got} vs {exact}"
        );
    }
}

use crate::physics::molecular::single_chain::{Ensemble, FreelyRotatingChain, MonteCarlo};

#[test]
fn bar() {
    const N: usize = 5;
    let model = FreelyRotatingChain {
        link_angle: 0.6,
        link_length: 1.0,
        number_of_links: N as u8,
        ensemble: Ensemble::Isometric,
    };
    let (gamma, g) =
        MonteCarlo::nondimensional_radial_distribution::<N>(&model, 333, 10_000_000, 1);
        // MonteCarlo::nondimensional_radial_distribution::<N>(&model, 1_000, 10_000_000, 4);
    gamma
        .into_iter()
        .zip(g)
        .for_each(|(gamma_i, g_i)| println!("[{gamma_i}, {g_i}],"))
}

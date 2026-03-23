use crate::{
    math::Scalar,
    physics::{
        BOLTZMANN_CONSTANT, ROOM_TEMPERATURE,
        molecular::single_chain::{Ensemble, ExtensibleFreelyRotatingChain, MonteCarloExtensible},
    },
};

const STIFFNESS: Scalar = 5.0 * BOLTZMANN_CONSTANT * ROOM_TEMPERATURE;

#[test]
fn monte_carlo() {
    let model = ExtensibleFreelyRotatingChain {
        link_angle: 0.4363323129985824,
        link_length: 1.0,
        link_stiffness: STIFFNESS,
        number_of_links: 3,
        ensemble: Ensemble::Isometric(ROOM_TEMPERATURE),
    };
    let (gamma, g) =
        MonteCarloExtensible::nondimensional_radial_distribution(&model, 333, 1_000_000, 4, 3.0);
    gamma
        .into_iter()
        .zip(g)
        .for_each(|(gamma_i, g_i)| println!("[{gamma_i}, {g_i}],"))
}

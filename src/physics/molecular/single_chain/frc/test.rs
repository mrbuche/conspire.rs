use crate::physics::{
    ROOM_TEMPERATURE,
    molecular::single_chain::{Ensemble, FreelyRotatingChain, MonteCarloInextensible},
};

#[test]
fn monte_carlo() {
    const N: usize = 8;
    let model = FreelyRotatingChain {
        link_angle: 0.4363323129985824,
        link_length: 1.0,
        number_of_links: N as u8,
        ensemble: Ensemble::Isometric(ROOM_TEMPERATURE),
    };
    let (gamma, g) =
        MonteCarloInextensible::nondimensional_radial_distribution(&model, 333, 10_000_000, 4);
    gamma
        .into_iter()
        .zip(g)
        .for_each(|(gamma_i, g_i)| println!("[{gamma_i}, {g_i}],"))
}

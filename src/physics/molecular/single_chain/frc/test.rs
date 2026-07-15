use crate::{
    math::{
        Rank2, Scalar, Tensor,
        assert::{AssertionError, assert_eq},
    },
    physics::{
        ROOM_TEMPERATURE,
        molecular::single_chain::{
            Ensemble, FreelyRotatingChain, MonteCarlo, MonteCarloInextensible,
        },
    },
};

const MODEL: FreelyRotatingChain = FreelyRotatingChain {
    link_angle: 0.4363323129985824,
    link_length: 1.0,
    number_of_links: 8,
    ensemble: Ensemble::Isometric(ROOM_TEMPERATURE),
};

#[test]
fn monte_carlo() {
    let (gamma, g) =
        MonteCarloInextensible::nondimensional_radial_distribution(&MODEL, 0.0, 333, 10_000, 1);
    gamma
        .into_iter()
        .zip(g)
        .for_each(|(gamma_i, g_i)| println!("[{gamma_i}, {g_i}],"));
}

#[test]
fn cosine_moments() -> Result<(), AssertionError> {
    let (cos, coscos, cos2, _) = MODEL.cosine_moments(3.3, 10_000, 1);
    let gamma_z = cos.iter().sum::<Scalar>() / MODEL.number_of_links as Scalar;
    assert!(gamma_z > 0.0 && gamma_z < 1.0);
    assert_eq(&coscos.transpose(), &coscos)?;
    coscos
        .into_iter()
        .zip(cos2)
        .enumerate()
        .try_for_each(|(i, (coscos_i, cos2_i))| assert_eq(&coscos_i[i], &cos2_i))
}

use crate::{
    EPSILON,
    math::{
        Scalar,
        test::{TestError, assert_eq_from_fd},
    },
    physics::{
        ROOM_TEMPERATURE,
        molecular::single_chain::{Ensemble, FreelyJointedChain, Thermodynamics},
    },
};

const NUM: usize = 333;

#[test]
fn monte_carlo() {
    use crate::physics::molecular::single_chain::MonteCarloInextensible;
    let model = FreelyJointedChain {
        link_length: 19.0,
        number_of_links: 5,
        ensemble: Ensemble::Isometric(ROOM_TEMPERATURE),
    };
    let (gamma, g) =
        MonteCarloInextensible::nondimensional_radial_distribution(&model, 333, 10_000, 1);
    gamma
        .into_iter()
        .zip(g)
        .for_each(|(gamma_i, g_i)| println!("[{gamma_i}, {g_i}],"))
}

#[test]
fn monte_carlo_cosines() {
    use crate::physics::molecular::single_chain::MonteCarloInextensible;
    let model = FreelyJointedChain {
        link_length: 1.0,
        number_of_links: 5,
        ensemble: Ensemble::Isotensional(ROOM_TEMPERATURE),
    };
    let eta = 3.3;
    println!(
        "{}",
        Thermodynamics::nondimensional_extension(&model, eta).unwrap()
    );
    let cosines = model.cosine_powers(eta, 2, 10_000, 1);
    println!("{:?}", cosines);
    let gamma = MonteCarloInextensible::nondimensional_extension(&model, eta, 10_000, 1);
    println!("{:?}", gamma);
}

#[test]
fn finite_difference() -> Result<(), TestError> {
    [
        // Ensemble::Isometric(ROOM_TEMPERATURE), // a bit finnicky and should maybe do separately
        Ensemble::Isotensional(ROOM_TEMPERATURE),
    ]
    .into_iter()
    .try_for_each(|ensemble| {
        (3..16).into_iter().try_for_each(|number_of_links| {
            let model = FreelyJointedChain {
                link_length: 1.0,
                number_of_links,
                ensemble,
            };
            (30..NUM)
                .map(|k| k as Scalar / NUM as Scalar)
                .into_iter()
                .try_for_each(|mut nondimensional_extension| {
                    println!("{}", nondimensional_extension);
                    let nondimensional_force =
                        model.nondimensional_force(nondimensional_extension)?;
                    let nondimensional_stiffness =
                        model.nondimensional_stiffness(nondimensional_extension)?;
                    nondimensional_extension += 0.5 * EPSILON;
                    let mut finite_difference_1 =
                        model.nondimensional_helmholtz_free_energy(nondimensional_extension)?;
                    let mut finite_difference_2 =
                        model.nondimensional_force(nondimensional_extension)?;
                    nondimensional_extension -= EPSILON;
                    finite_difference_1 -=
                        model.nondimensional_helmholtz_free_energy(nondimensional_extension)?;
                    finite_difference_2 -= model.nondimensional_force(nondimensional_extension)?;
                    assert_eq_from_fd(
                        &nondimensional_force,
                        &(finite_difference_1 / number_of_links as Scalar / EPSILON),
                    )?;
                    assert_eq_from_fd(&nondimensional_stiffness, &(finite_difference_2 / EPSILON))
                })
        })
    })
}

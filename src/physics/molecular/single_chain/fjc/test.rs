use crate::{
    EPSILON,
    math::{
        Scalar,
        test::{TestError, assert_eq_from_fd},
    },
    physics::{
        ROOM_TEMPERATURE,
        molecular::single_chain::{Ensemble, FreelyJointedChain, MonteCarlo, Thermodynamics},
    },
};

const NUM: usize = 333;

#[test]
fn monte_carlo() {
    const N: usize = 5;
    let model = FreelyJointedChain {
        link_length: 1.0,
        number_of_links: N as u8,
        ensemble: Ensemble::Isometric(ROOM_TEMPERATURE),
    };
    let (gamma, g) = MonteCarlo::nondimensional_radial_distribution::<N>(&model, 333, 1_000_000, 4);
    gamma
        .into_iter()
        .zip(g)
        .for_each(|(gamma_i, g_i)| println!("[{gamma_i}, {g_i}],"))
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

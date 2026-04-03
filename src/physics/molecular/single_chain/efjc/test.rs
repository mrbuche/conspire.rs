use crate::{
    EPSILON,
    math::{
        Scalar,
        test::{TestError, assert_eq_from_fd},
    },
    physics::{
        BOLTZMANN_CONSTANT, ROOM_TEMPERATURE,
        molecular::single_chain::{Ensemble, ExtensibleFreelyJointedChain, Thermodynamics},
    },
};

const STIFFNESS: Scalar = 5.0 * BOLTZMANN_CONSTANT * ROOM_TEMPERATURE;
const NUM: usize = 333;

#[test]
fn monte_carlo() {
    use crate::physics::molecular::single_chain::MonteCarloExtensible;
    let model = ExtensibleFreelyJointedChain {
        link_length: 1.0,
        link_stiffness: STIFFNESS,
        number_of_links: 5,
        ensemble: Ensemble::Isometric(ROOM_TEMPERATURE),
    };
    let (gamma, g) =
        MonteCarloExtensible::nondimensional_radial_distribution(&model, 0.0, 333, 10_000, 1, 3.0);
    gamma
        .into_iter()
        .zip(g)
        .for_each(|(gamma_i, g_i)| println!("[{gamma_i}, {g_i}],"))
}

#[test]
fn finite_difference() -> Result<(), TestError> {
    let link_stiffness = 1e3;
    [Ensemble::Isotensional(ROOM_TEMPERATURE)]
        .into_iter()
        .try_for_each(|ensemble| {
            (3..16).into_iter().try_for_each(|number_of_links| {
                let model = ExtensibleFreelyJointedChain {
                    link_length: 1.0,
                    link_stiffness,
                    number_of_links,
                    ensemble,
                };
                (10..NUM)
                    .map(|k| k as Scalar / NUM as Scalar * 0.6 * link_stiffness)
                    .into_iter()
                    .try_for_each(|mut nondimensional_force| {
                        nondimensional_force += 0.5 * EPSILON;
                        let mut finite_difference_3 = -model
                            .nondimensional_gibbs_free_energy_per_link(nondimensional_force)?;
                        // let mut finite_difference_4 =
                        //     model.nondimensional_extension(nondimensional_force)?;
                        nondimensional_force -= EPSILON;
                        finite_difference_3 -= -model
                            .nondimensional_gibbs_free_energy_per_link(nondimensional_force)?;
                        // finite_difference_4 -=
                        //     model.nondimensional_extension(nondimensional_force)?;
                        nondimensional_force += 0.5 * EPSILON;
                        let nondimensional_extension =
                            model.nondimensional_extension(nondimensional_force)?;
                        // let nondimensional_compliance =
                        //     model.nondimensional_compliance(nondimensional_force)?;
                        assert_eq_from_fd(
                            &nondimensional_extension,
                            &(finite_difference_3 / EPSILON),
                        )
                        // assert_eq_from_fd(
                        //     &nondimensional_compliance,
                        //     &(finite_difference_4 / EPSILON),
                        // )
                    })
            })
        })
}

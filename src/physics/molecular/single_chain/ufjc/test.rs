use crate::{
    EPSILON,
    math::{
        Scalar,
        test::{TestError, assert_eq_from_fd},
    },
    physics::{
        molecular::{
            potential::{Harmonic, Morse},
            single_chain::{ArbitraryPotentialFreelyJointedChain, Ensemble, Thermodynamics},
        },
        {BOLTZMANN_CONSTANT, ROOM_TEMPERATURE},
    },
};

const NUM: usize = 333;

#[test]
fn finite_difference() -> Result<(), TestError> {
    let e = 1e5;
    let a = 1.0;
    let x0 = 1.0;
    let eta_max = 0.5 * a * x0 * e / BOLTZMANN_CONSTANT / ROOM_TEMPERATURE;
    [Ensemble::Isotensional(ROOM_TEMPERATURE)]
        .into_iter()
        .try_for_each(|ensemble| {
            (3..16).into_iter().try_for_each(|number_of_links| {
                let model = ArbitraryPotentialFreelyJointedChain {
                    link_potential: Harmonic {
                        rest_length: x0,
                        stiffness: e,
                    },
                    number_of_links,
                    ensemble,
                };
                (1..NUM)
                    .map(|k| k as Scalar / NUM as Scalar * eta_max)
                    .into_iter()
                    .try_for_each(|mut nondimensional_force| {
                        nondimensional_force += 0.5 * EPSILON;
                        let mut finite_difference_3 = -model
                            .nondimensional_gibbs_free_energy_per_link(nondimensional_force)?;
                        let mut finite_difference_4 =
                            model.nondimensional_extension(nondimensional_force)?;
                        nondimensional_force -= EPSILON;
                        finite_difference_3 -= -model
                            .nondimensional_gibbs_free_energy_per_link(nondimensional_force)?;
                        finite_difference_4 -=
                            model.nondimensional_extension(nondimensional_force)?;
                        nondimensional_force += 0.5 * EPSILON;
                        let nondimensional_extension =
                            model.nondimensional_extension(nondimensional_force)?;
                        let nondimensional_compliance =
                            model.nondimensional_compliance(nondimensional_force)?;
                        assert_eq_from_fd(
                            &nondimensional_extension,
                            &(finite_difference_3 / EPSILON),
                        )?;
                        assert_eq_from_fd(
                            &nondimensional_compliance,
                            &(finite_difference_4 / EPSILON),
                        )
                    })
            })
        })?;
    [Ensemble::Isotensional(ROOM_TEMPERATURE)]
        .into_iter()
        .try_for_each(|ensemble| {
            (3..16).into_iter().try_for_each(|number_of_links| {
                let model = ArbitraryPotentialFreelyJointedChain {
                    link_potential: Morse {
                        rest_length: x0,
                        depth: e,
                        parameter: a,
                    },
                    number_of_links,
                    ensemble,
                };
                (1..NUM)
                    .map(|k| k as Scalar / NUM as Scalar * eta_max)
                    .into_iter()
                    .try_for_each(|mut nondimensional_force| {
                        nondimensional_force += 0.5 * EPSILON;
                        let mut finite_difference_3 = -model
                            .nondimensional_gibbs_free_energy_per_link(nondimensional_force)?;
                        let mut finite_difference_4 =
                            model.nondimensional_extension(nondimensional_force)?;
                        nondimensional_force -= EPSILON;
                        finite_difference_3 -= -model
                            .nondimensional_gibbs_free_energy_per_link(nondimensional_force)?;
                        finite_difference_4 -=
                            model.nondimensional_extension(nondimensional_force)?;
                        nondimensional_force += 0.5 * EPSILON;
                        let nondimensional_extension =
                            model.nondimensional_extension(nondimensional_force)?;
                        let nondimensional_compliance =
                            model.nondimensional_compliance(nondimensional_force)?;
                        assert_eq_from_fd(
                            &nondimensional_extension,
                            &(finite_difference_3 / EPSILON),
                        )?;
                        assert_eq_from_fd(
                            &nondimensional_compliance,
                            &(finite_difference_4 / EPSILON),
                        )
                    })
            })
        })
}

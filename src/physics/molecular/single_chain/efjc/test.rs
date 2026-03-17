use crate::{
    EPSILON,
    math::{
        Scalar,
        test::{TestError, assert_eq_from_fd},
    },
    physics::{
        ROOM_TEMPERATURE,
        molecular::single_chain::{Ensemble, ExtensibleFreelyJointedChain, Thermodynamics},
    },
};

const NUM: usize = 1000;

#[test]
fn finite_difference() -> Result<(), TestError> {
    [Ensemble::Isotensional]
        .into_iter()
        .try_for_each(|ensemble| {
            (3..16).into_iter().try_for_each(|number_of_links| {
                let model = ExtensibleFreelyJointedChain {
                    link_length: 1.0,
                    link_stiffness: 1000.0,
                    number_of_links,
                    ensemble,
                };
                (0..NUM)
                    .map(|k| k as Scalar / NUM as Scalar * 10.0)
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

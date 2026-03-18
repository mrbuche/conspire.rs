use crate::{
    EPSILON,
    math::{
        Scalar,
        test::{TestError, assert_eq_from_fd},
    },
    physics::molecular::single_chain::{Ensemble, IdealChain, Thermodynamics},
};

const NUM: usize = 333;

#[test]
fn finite_difference() -> Result<(), TestError> {
    [Ensemble::Isometric, Ensemble::Isotensional]
        .into_iter()
        .try_for_each(|ensemble| {
            (3..16).into_iter().try_for_each(|number_of_links| {
                let model = IdealChain {
                    link_length: 1.0,
                    number_of_links,
                    ensemble,
                };
                (0..NUM)
                    .map(|k| k as Scalar / NUM as Scalar)
                    .into_iter()
                    .try_for_each(|mut nondimensional_extension| {
                        let mut nondimensional_force =
                            model.nondimensional_force(nondimensional_extension)?;
                        let nondimensional_stiffness =
                            model.nondimensional_stiffness(nondimensional_extension)?;
                        nondimensional_extension += 0.5 * EPSILON;
                        nondimensional_force += 0.5 * EPSILON;
                        let mut finite_difference_1 = model
                            .nondimensional_helmholtz_free_energy_per_link(
                                nondimensional_extension,
                            )?;
                        let mut finite_difference_2 =
                            model.nondimensional_force(nondimensional_extension)?;
                        let mut finite_difference_3 = -model
                            .nondimensional_gibbs_free_energy_per_link(nondimensional_force)?;
                        let mut finite_difference_4 =
                            model.nondimensional_extension(nondimensional_force)?;
                        nondimensional_extension -= EPSILON;
                        nondimensional_force -= EPSILON;
                        finite_difference_1 -= model
                            .nondimensional_helmholtz_free_energy_per_link(
                                nondimensional_extension,
                            )?;
                        finite_difference_2 -=
                            model.nondimensional_force(nondimensional_extension)?;
                        finite_difference_3 -= -model
                            .nondimensional_gibbs_free_energy_per_link(nondimensional_force)?;
                        finite_difference_4 -=
                            model.nondimensional_extension(nondimensional_force)?;
                        nondimensional_force += 0.5 * EPSILON;
                        let nondimensional_extension =
                            model.nondimensional_extension(nondimensional_force)?;
                        let nondimensional_compliance =
                            model.nondimensional_compliance(nondimensional_force)?;
                        assert_eq_from_fd(&nondimensional_force, &(finite_difference_1 / EPSILON))?;
                        assert_eq_from_fd(
                            &nondimensional_stiffness,
                            &(finite_difference_2 / EPSILON),
                        )?;
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

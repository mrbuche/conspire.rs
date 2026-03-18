use crate::{
    EPSILON,
    math::{
        Scalar,
        test::{TestError, assert_eq_from_fd},
    },
    physics::{
        ROOM_TEMPERATURE,
        molecular::{
            potential::{Harmonic, Potential},
            single_chain::{Ensemble, Foo, Thermodynamics},
        },
    },
};

const NUM: usize = 333;

#[test]
fn finite_difference() -> Result<(), TestError> {
    let e = 1.2;
    let a = 1.1;
    let x0 = 1.5;
    let x_max = x0 + 2.0_f64.ln() / a;
    [Ensemble::Isotensional]
        .into_iter()
        .try_for_each(|ensemble| {
            (3..16).into_iter().try_for_each(|number_of_links| {
                let model = Foo {
                    link_potential: Harmonic {
                        rest_length: x0,
                        stiffness: e,
                    },
                    number_of_links,
                    ensemble,
                };
                (30..NUM)
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
    todo!("Morse");
    Ok(())
}

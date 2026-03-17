use crate::{
    EPSILON,
    math::{
        Scalar,
        test::{TestError, assert_eq_from_fd},
    },
    physics::molecular::single_chain::{Ensemble, Foo, MonteCarlo, Thermodynamics},
};

const NUM: usize = 1000;

#[test]
fn monte_carlo() {
    const N: usize = 5;
    let model = Foo {
        link_length: 1.0,
        number_of_links: N as u8,
        well_width: 0.3,
        ensemble: Ensemble::Isometric,
    };
    let (gamma, g) = MonteCarlo::nondimensional_radial_distribution::<N>(&model, 333, 1_000_000, 4);
    gamma
        .into_iter()
        .zip(g)
        .for_each(|(gamma_i, g_i)| println!("[{gamma_i}, {g_i}],"))
}

#[test]
fn finite_difference() -> Result<(), TestError> {
    [Ensemble::Isotensional]
        .into_iter()
        .try_for_each(|ensemble| {
            (3..16).into_iter().try_for_each(|number_of_links| {
                let model = Foo {
                    link_length: 1.0,
                    number_of_links,
                    well_width: 0.3,
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
        })
}

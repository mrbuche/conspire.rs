use crate::{
    EPSILON,
    math::{
        Scalar,
        test::{TestError, assert_eq_from_fd},
    },
    physics::molecular::single_chain::{Ensemble, FreelyJointedChain, Thermodynamics},
};

const NUM: usize = 1000;

#[test]
fn foo() {
    let fjc = FreelyJointedChain {
        link_length: 1.0,
        number_of_links: 6,
        ensemble: Ensemble::Isometric,
    };
    let force = fjc.nondimensional_force(-0.8);
    println!("{:?}", force);
    let h = fjc.nondimensional_helmholtz_free_energy(1e-5);
    println!("{:?}", h);
    let g = crate::physics::molecular::single_chain::Isometric::nondimensional_radial_distribution(
        &fjc, 0.2,
    );
    println!("{:?}", g);
    let p =
        crate::physics::molecular::single_chain::Isometric::nondimensional_spherical_distribution(
            &fjc, 0.01,
        );
    println!("{:?}", p)
}

#[test]
fn finite_difference() -> Result<(), TestError> {
    [Ensemble::Isometric, Ensemble::Isotensional]
        .into_iter()
        .try_for_each(|ensemble| {
            (3..16).into_iter().try_for_each(|number_of_links| {
                let model = FreelyJointedChain {
                    link_length: 1.0,
                    number_of_links,
                    ensemble,
                };
                (0..NUM)
                    .map(|k| k as Scalar / NUM as Scalar)
                    .into_iter()
                    .try_for_each(|mut nondimensional_extension| {
                        let nondimensional_force =
                            model.nondimensional_force(nondimensional_extension)?;
                        nondimensional_extension += 0.5 * EPSILON;
                        let mut finite_difference =
                            model.nondimensional_helmholtz_free_energy(nondimensional_extension)?;
                        nondimensional_extension -= EPSILON;
                        finite_difference -=
                            model.nondimensional_helmholtz_free_energy(nondimensional_extension)?;
                        assert_eq_from_fd(
                            &nondimensional_force,
                            &(finite_difference / number_of_links as Scalar / EPSILON),
                        )
                    })
            })
        })
}

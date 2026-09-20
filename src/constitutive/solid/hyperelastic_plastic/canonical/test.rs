use crate::{
    EPSILON,
    constitutive::{
        canonical::Canonical,
        fluid::plastic::{PlasticFlow, RateIndependentPlastic},
        solid::{
            elastic_plastic::ElasticPlasticOrViscoplastic,
            hyperelastic::{Hencky, NeoHookean, SaintVenantKirchhoff},
            hyperelastic_plastic::HyperelasticPlastic,
        },
    },
    math::{
        Rank2, Tensor,
        assert::{AssertionError, perturbation},
    },
    mechanics::{DeformationGradient, DeformationGradientPlastic},
    units::Stress,
};

macro_rules! test_canonical {
    ($elastic:ident) => {
        use super::*;
        fn model() -> Canonical<$elastic, PlasticFlow> {
            Canonical::from((
                $elastic {
                    bulk_modulus: Stress::pascals(13.0),
                    shear_modulus: Stress::pascals(3.0),
                },
                PlasticFlow {
                    yield_stress: Stress::pascals(2.0),
                    hardening_slope: Stress::pascals(1.0),
                },
            ))
        }
        fn deformation_gradient_p() -> DeformationGradientPlastic {
            DeformationGradientPlastic::from([
                [1.2, 0.1, 0.05],
                [0.0, 0.5, 0.2],
                [0.0, 0.0, 1.0 / 0.6],
            ])
        }
        fn deformation_gradient() -> DeformationGradient {
            DeformationGradient::from([
                [1.31924942, 0.36431217, 0.41764434],
                [0.09959341, 1.08409741, 0.48320137],
                [0.21114106, 0.16675104, 1.18146028],
            ])
        }
        #[test]
        fn stress_is_energy_gradient() -> Result<(), AssertionError> {
            let model = model();
            let (f, f_p) = (deformation_gradient(), deformation_gradient_p());
            let stress = model.first_piola_kirchhoff_stress(&f, &f_p)?;
            for i in 0..3 {
                for j in 0..3 {
                    let mut plus = f.clone();
                    plus[i][j] += perturbation(0.5 * EPSILON);
                    let mut minus = f.clone();
                    minus[i][j] -= perturbation(0.5 * EPSILON);
                    let fd = (model.helmholtz_free_energy_density(&plus, &f_p)?
                        - model.helmholtz_free_energy_density(&minus, &f_p)?)
                    .value()
                        / EPSILON;
                    let exact = stress[i][j].value();
                    assert!(
                        (fd - exact).abs() <= 1e-5 * exact.abs().max(1.0),
                        "dPsi/dF[{i}][{j}]: fd {fd} vs P {exact}",
                    );
                }
            }
            Ok(())
        }
        #[test]
        fn plastic_flow_releases_mandel_power() -> Result<(), AssertionError> {
            let model = model();
            let (f, f_p) = (deformation_gradient(), deformation_gradient_p());
            let deviatoric = model.mandel_stress(&f, &f_p)?.deviatoric();
            let direction = {
                let direction = model.flow_direction(&deviatoric)?;
                (&direction + direction.transpose()) * 0.5
            };
            let step = 1e-6;
            let energy = |gamma: f64| -> Result<f64, AssertionError> {
                let f_p_gamma = (&direction * gamma).expm().unwrap() * &f_p;
                Ok(model.helmholtz_free_energy_density(&f, &f_p_gamma)?.value())
            };
            let fd = (energy(step)? - energy(-step)?) / (2.0 * step);
            let exact = -deviatoric.norm().value();
            assert!(
                (fd - exact).abs() <= 1e-5 * exact.abs().max(1.0),
                "da/dgamma: fd {fd} vs -|M'| {exact}",
            );
            Ok(())
        }
    };
}

mod neo_hookean {
    test_canonical!(NeoHookean);
}

mod hencky {
    test_canonical!(Hencky);
}

mod saint_venant_kirchhoff {
    test_canonical!(SaintVenantKirchhoff);
}

mod direction_diagnostic {
    use super::*;
    use crate::{
        constitutive::{
            ConstitutiveError, fluid::plastic::PlasticStateVariables,
            solid::elastic_plastic::ElasticPlastic,
        },
        math::{Quantity, SquareMatrix, TensorArray, Vector},
        mechanics::FlowDirectionPlastic,
    };

    type Plastic = (DeformationGradientPlastic, f64);

    fn direction<M: ElasticPlastic>(
        model: &M,
        f: &DeformationGradient,
        f_p: &DeformationGradientPlastic,
    ) -> Result<FlowDirectionPlastic, ConstitutiveError> {
        let deviatoric = model.mandel_stress(f, f_p)?.deviatoric();
        let direction = model.flow_direction(&deviatoric)?;
        Ok((&direction + direction.transpose()) * 0.5)
    }

    /// Return map whose flow direction is re-evaluated at the updated plastic
    /// gradient `sweeps` times; `sweeps = 0` freezes it at the trial state, exactly as
    /// `ElasticPlastic::return_map` does. Sweeps stop early once the plastic gradient
    /// changes by less than a tolerance; the number of passes taken and the last
    /// change are returned alongside the state. The direction update is damped, and
    /// its fixed point is the fully implicit end-of-step direction.
    fn return_map_swept<M: ElasticPlastic>(
        model: &M,
        f: &DeformationGradient,
        state: &Plastic,
        sweeps: usize,
    ) -> Result<(Plastic, usize, f64), ConstitutiveError> {
        let (f_p_n, strain_n) = state;
        let deviatoric = model.mandel_stress(f, f_p_n)?.deviatoric();
        if model
            .yield_function(&deviatoric, Quantity::new(*strain_n))?
            .value()
            <= 0.0
        {
            return Ok((state.clone(), 0, 0.0));
        }
        let mut n = direction(model, f, f_p_n)?;
        let mut result = (f_p_n.clone(), *strain_n);
        let (mut passes, mut change) = (0, f64::INFINITY);
        for _ in 0..=sweeps {
            let f_p_at = |gamma: f64| (&n * gamma).expm().unwrap() * f_p_n;
            let residual = |gamma: f64| -> Result<f64, ConstitutiveError> {
                let deviatoric = model.mandel_stress(f, &f_p_at(gamma))?.deviatoric();
                Ok(model
                    .yield_function(&deviatoric, Quantity::new(strain_n + gamma))?
                    .value())
            };
            let (mut lower, mut upper) = (0.0, 1e-3);
            while residual(upper)? > 0.0 {
                upper *= 2.0;
                if upper > 16.0 {
                    return Err(ConstitutiveError::custom(
                        "Failed to bracket the plastic multiplier.",
                        model,
                    ));
                }
            }
            for _ in 0..200 {
                let middle = 0.5 * (lower + upper);
                if residual(middle)? > 0.0 {
                    lower = middle
                } else {
                    upper = middle
                }
            }
            let gamma = 0.5 * (lower + upper);
            let f_p = f_p_at(gamma);
            let updated = direction(model, f, &f_p)?;
            let mixed = (&n + updated) * 0.5;
            n = &mixed / mixed.norm().value();
            passes += 1;
            if passes > 1 {
                change = (&f_p - &result.0).norm().value();
            }
            result = (f_p, strain_n + gamma);
            if change < 1e-13 {
                break;
            }
        }
        Ok((result, passes - 1, change))
    }

    use crate::constitutive::solid::elastic_plastic::{Entries4, entries_4, matrix_3};
    use std::array::from_fn;

    const SIZE: usize = 10;
    const ZERO: Matrix3 = [[0.0; 3]; 3];
    const IDENTITY: Matrix3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

    type Matrix3 = [[f64; 3]; 3];
    type Unknowns = [f64; SIZE];

    #[derive(Clone, Copy)]
    enum Guess {
        Frozen,
        Cold,
    }

    #[derive(Clone, Copy)]
    enum Jacobian {
        FiniteDifference,
        Analytic,
    }

    fn mul(a: &Matrix3, b: &Matrix3) -> Matrix3 {
        from_fn(|i| from_fn(|j| (0..3).map(|k| a[i][k] * b[k][j]).sum()))
    }

    fn transpose(a: &Matrix3) -> Matrix3 {
        from_fn(|i| from_fn(|j| a[j][i]))
    }

    fn add(a: &Matrix3, b: &Matrix3, scale: f64) -> Matrix3 {
        from_fn(|i| from_fn(|j| a[i][j] + scale * b[i][j]))
    }

    fn trace(a: &Matrix3) -> f64 {
        a[0][0] + a[1][1] + a[2][2]
    }

    fn determinant(a: &Matrix3) -> f64 {
        a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
            - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
            + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])
    }

    fn contract(c: &Entries4, x: &Matrix3) -> Matrix3 {
        from_fn(|i| {
            from_fn(|j| {
                (0..3)
                    .map(|k| (0..3).map(|l| c[i][j][k][l] * x[k][l]).sum::<f64>())
                    .sum()
            })
        })
    }

    fn basis(a: usize, b: usize) -> Matrix3 {
        from_fn(|i| from_fn(|j| if i == a && j == b { 1.0 } else { 0.0 }))
    }

    fn increment(x: &Unknowns) -> FlowDirectionPlastic {
        FlowDirectionPlastic::from(from_fn::<_, 3, _>(|i| from_fn::<_, 3, _>(|j| x[3 * i + j])))
    }

    fn plastic_gradient<M: ElasticPlastic>(
        model: &M,
        f_p_n: &DeformationGradientPlastic,
        x: &Unknowns,
    ) -> Result<DeformationGradientPlastic, ConstitutiveError> {
        Ok(increment(x)
            .expm()
            .map_err(|error| ConstitutiveError::custom(format!("{error:?}"), model))?
            * f_p_n)
    }

    /// The elastic-plastic stress and its tangent at fixed `Fp`, from which the
    /// directional derivatives of the first Piola-Kirchhoff and Mandel stresses along
    /// any `(dF, dFp)` follow. The model's Mandel stress carries `det F`, whereas
    /// `Fe^T P Fp^T` carries `det Fe`; the two differ by `det Fp`, which is what the
    /// last step of `mandel_derivative` accounts for.
    struct Linearization {
        tangent: Entries4,
        stress: Matrix3,
        f_p: Matrix3,
        f_p_inverse: Matrix3,
        f_e: Matrix3,
        mandel: Matrix3,
    }

    impl Linearization {
        fn new<M: ElasticPlastic>(
            model: &M,
            f: &DeformationGradient,
            f_p: &DeformationGradientPlastic,
        ) -> Result<Self, ConstitutiveError> {
            let f_p_inverse = f_p.inverse();
            let stress = matrix_3(&model.first_piola_kirchhoff_stress(f, f_p)?);
            let f_e = matrix_3(&(f * &f_p_inverse));
            let f_p_matrix = matrix_3(f_p);
            let mandel = mul(&mul(&transpose(&f_e), &stress), &transpose(&f_p_matrix));
            Ok(Self {
                tangent: entries_4(&model.first_piola_kirchhoff_tangent_stiffness(f, f_p)?),
                stress,
                f_p: f_p_matrix,
                f_p_inverse: matrix_3(&f_p_inverse),
                f_e,
                mandel,
            })
        }

        fn stress_derivative(&self, d_f: &Matrix3, d_f_p: &Matrix3) -> Matrix3 {
            let by_f = contract(&self.tangent, d_f);
            let by_f_p = contract(&self.tangent, &mul(&self.f_e, d_f_p));
            let correction = mul(
                &mul(&self.stress, &transpose(d_f_p)),
                &transpose(&self.f_p_inverse),
            );
            from_fn(|i| from_fn(|j| by_f[i][j] - by_f_p[i][j] - correction[i][j]))
        }

        fn mandel_derivative(&self, d_f: &Matrix3, d_f_p: &Matrix3) -> Matrix3 {
            let d_f_e = add(
                &mul(d_f, &self.f_p_inverse),
                &mul(&mul(&self.f_e, d_f_p), &self.f_p_inverse),
                -1.0,
            );
            let d_p = self.stress_derivative(d_f, d_f_p);
            let (f_e_t, f_p_t) = (transpose(&self.f_e), transpose(&self.f_p));
            let d_m = add(
                &add(
                    &mul(&mul(&transpose(&d_f_e), &self.stress), &f_p_t),
                    &mul(&mul(&f_e_t, &d_p), &f_p_t),
                    1.0,
                ),
                &mul(&mul(&f_e_t, &self.stress), &transpose(d_f_p)),
                1.0,
            );
            let d_m = add(&d_m, &self.mandel, trace(&mul(&self.f_p_inverse, d_f_p)));
            let determinant = determinant(&self.f_p);
            from_fn(|i| from_fn(|j| determinant * d_m[i][j]))
        }
    }

    /// Everything the Jacobian and the consistent tangent need at a given iterate.
    struct Sensitivities {
        linearization: Linearization,
        unit: Matrix3,
        symmetric: Matrix3,
        magnitude: f64,
        slopes: [[Matrix3; 3]; 3],
    }

    impl Sensitivities {
        fn new<M: ElasticPlastic>(
            model: &M,
            f: &DeformationGradient,
            f_p_n: &DeformationGradientPlastic,
            x: &Unknowns,
        ) -> Result<Self, ConstitutiveError> {
            let f_p = plastic_gradient(model, f_p_n, x)?;
            let deviatoric = model.mandel_stress(f, &f_p)?.deviatoric();
            let magnitude = deviatoric.norm().value();
            let deviatoric = matrix_3(&deviatoric);
            let unit: Matrix3 = from_fn(|i| from_fn(|j| deviatoric[i][j] / magnitude));
            let symmetric = add(&add(&ZERO, &unit, 0.5), &transpose(&unit), 0.5);
            let exponential_slope = entries_4(
                &increment(x)
                    .dexpm()
                    .map_err(|error| ConstitutiveError::custom(format!("{error:?}"), model))?,
            );
            let f_p_n = matrix_3(f_p_n);
            let slopes = from_fn(|a| {
                from_fn(|b| {
                    from_fn(|i| {
                        from_fn(|j| {
                            (0..3)
                                .map(|k| exponential_slope[i][k][a][b] * f_p_n[k][j])
                                .sum()
                        })
                    })
                })
            });
            Ok(Self {
                linearization: Linearization::new(model, f, &f_p)?,
                unit,
                symmetric,
                magnitude,
                slopes,
            })
        }

        /// The slope of the symmetrized flow direction and of `|M'|` along a Mandel
        /// stress increment.
        fn direction_slope(&self, d_m: &Matrix3) -> (Matrix3, f64) {
            let deviatoric = add(d_m, &IDENTITY, -trace(d_m) / 3.0);
            let d_magnitude = (0..3)
                .map(|i| {
                    (0..3)
                        .map(|j| self.unit[i][j] * deviatoric[i][j])
                        .sum::<f64>()
                })
                .sum();
            let d_unit: Matrix3 = from_fn(|i| {
                from_fn(|j| (deviatoric[i][j] - self.unit[i][j] * d_magnitude) / self.magnitude)
            });
            (
                add(&add(&ZERO, &d_unit, 0.5), &transpose(&d_unit), 0.5),
                d_magnitude,
            )
        }
    }

    fn residual<M: ElasticPlastic>(
        model: &M,
        f: &DeformationGradient,
        f_p_n: &DeformationGradientPlastic,
        strain_n: f64,
        x: &Unknowns,
    ) -> Result<Unknowns, ConstitutiveError> {
        let deviatoric = model
            .mandel_stress(f, &plastic_gradient(model, f_p_n, x)?)?
            .deviatoric();
        let n = {
            let direction = model.flow_direction(&deviatoric)?;
            (&direction + direction.transpose()) * 0.5
        };
        let mut r = [0.0; SIZE];
        (0..3).for_each(|i| {
            (0..3).for_each(|j| r[3 * i + j] = x[3 * i + j] - x[9] * n[i][j].value())
        });
        r[9] = model
            .yield_function(&deviatoric, Quantity::new(strain_n + x[9]))?
            .value();
        Ok(r)
    }

    fn finite_difference_jacobian<M: ElasticPlastic>(
        model: &M,
        f: &DeformationGradient,
        f_p_n: &DeformationGradientPlastic,
        strain_n: f64,
        x: &Unknowns,
    ) -> Result<[[f64; SIZE]; SIZE], ConstitutiveError> {
        let mut jacobian = [[0.0; SIZE]; SIZE];
        for column in 0..SIZE {
            let h = 1e-7;
            let (mut plus, mut minus) = (*x, *x);
            plus[column] += h;
            minus[column] -= h;
            let r_plus = residual(model, f, f_p_n, strain_n, &plus)?;
            let r_minus = residual(model, f, f_p_n, strain_n, &minus)?;
            (0..SIZE)
                .for_each(|row| jacobian[row][column] = (r_plus[row] - r_minus[row]) / (2.0 * h));
        }
        Ok(jacobian)
    }

    fn analytic_jacobian(
        sensitivities: &Sensitivities,
        gamma: f64,
        hardening_slope: f64,
    ) -> [[f64; SIZE]; SIZE] {
        let mut jacobian = [[0.0; SIZE]; SIZE];
        for a in 0..3 {
            for b in 0..3 {
                let d_m = sensitivities
                    .linearization
                    .mandel_derivative(&ZERO, &sensitivities.slopes[a][b]);
                let (d_symmetric, d_magnitude) = sensitivities.direction_slope(&d_m);
                for i in 0..3 {
                    for j in 0..3 {
                        jacobian[3 * i + j][3 * a + b] =
                            f64::from(u8::from(i == a && j == b)) - gamma * d_symmetric[i][j];
                    }
                }
                jacobian[9][3 * a + b] = d_magnitude;
            }
        }
        for i in 0..3 {
            for j in 0..3 {
                jacobian[3 * i + j][9] = -sensitivities.symmetric[i][j];
            }
        }
        jacobian[9][9] = -hardening_slope;
        jacobian
    }

    /// Coupled Newton solve for the fully implicit step. Unknowns are the plastic
    /// increment `dE = dgamma N` (nine components, so symmetry and tracelessness are
    /// enforced by the residual rather than a basis) and `dgamma`; the residuals are
    /// ```text
    /// dE - dgamma N(M'(F, exp(dE) Fp_n)) = 0,   f(M', eps_n + dgamma) = 0.
    /// ```
    /// Returns `None` for an elastic step.
    fn newton_solve<M: ElasticPlastic>(
        model: &M,
        f: &DeformationGradient,
        state: &Plastic,
        guess: Guess,
        jacobian: Jacobian,
    ) -> Result<Option<(Unknowns, usize, f64)>, ConstitutiveError> {
        let (f_p_n, strain_n) = state;
        let deviatoric = model.mandel_stress(f, f_p_n)?.deviatoric();
        if model
            .yield_function(&deviatoric, Quantity::new(*strain_n))?
            .value()
            <= 0.0
        {
            return Ok(None);
        }
        let failure =
            |error: &dyn std::fmt::Debug| ConstitutiveError::custom(format!("{error:?}"), model);
        let scale = model.initial_yield_stress().value().max(1.0);
        let size = |r: &Unknowns| {
            r[..9]
                .iter()
                .fold(r[9].abs() / scale, |norm, value| norm.max(value.abs()))
        };
        let n0 = direction(model, f, f_p_n)?;
        let gamma0 = match guess {
            Guess::Frozen => return_map_swept(model, f, state, 0)?.0.1 - strain_n,
            Guess::Cold => 1e-3,
        };
        let mut x = [0.0; SIZE];
        (0..3).for_each(|i| (0..3).for_each(|j| x[3 * i + j] = gamma0 * n0[i][j].value()));
        x[9] = gamma0;
        let mut r = residual(model, f, f_p_n, *strain_n, &x)?;
        for iteration in 0..=30 {
            if size(&r) < 1e-12 {
                return Ok(Some((x, iteration, size(&r))));
            }
            let matrix = match jacobian {
                Jacobian::FiniteDifference => {
                    finite_difference_jacobian(model, f, f_p_n, *strain_n, &x)?
                }
                Jacobian::Analytic => analytic_jacobian(
                    &Sensitivities::new(model, f, f_p_n, &x)?,
                    x[9],
                    model.hardening_slope().value(),
                ),
            };
            let mut rhs = Vector::zero(SIZE);
            (0..SIZE).for_each(|row| rhs[row] = -r[row]);
            let step = SquareMatrix::from(matrix)
                .solve_lu(&rhs)
                .map_err(|error| failure(&error))?;
            (0..SIZE).for_each(|row| x[row] += step[row]);
            r = residual(model, f, f_p_n, *strain_n, &x)?;
        }
        Err(ConstitutiveError::custom(
            format!("Newton did not converge, |R| = {:.3e}", size(&r)),
            model,
        ))
    }

    fn return_map_newton<M: ElasticPlastic>(
        model: &M,
        f: &DeformationGradient,
        state: &Plastic,
        guess: Guess,
        jacobian: Jacobian,
    ) -> Result<(Plastic, usize, f64), ConstitutiveError> {
        Ok(match newton_solve(model, f, state, guess, jacobian)? {
            None => (state.clone(), 0, 0.0),
            Some((x, iterations, residual)) => (
                (plastic_gradient(model, &state.0, &x)?, state.1 + x[9]),
                iterations,
                residual,
            ),
        })
    }

    /// The consistent tangent `dP/dF` of the converged coupled solve, by the implicit
    /// function theorem: `dx/dF = -J^-1 dR/dF`, then
    /// `dP/dF = dP/dF|Fp + dP/dFp : dFp/dE : dE/dF`.
    fn consistent_tangent<M: ElasticPlastic>(
        model: &M,
        f: &DeformationGradient,
        state: &Plastic,
    ) -> Result<Entries4, ConstitutiveError> {
        let Some((x, ..)) = newton_solve(model, f, state, Guess::Cold, Jacobian::Analytic)? else {
            return Ok(Linearization::new(model, f, &state.0)?.tangent);
        };
        let sensitivities = Sensitivities::new(model, f, &state.0, &x)?;
        let lu = SquareMatrix::from(analytic_jacobian(
            &sensitivities,
            x[9],
            model.hardening_slope().value(),
        ))
        .factorize_lu()
        .map_err(|error| ConstitutiveError::custom(format!("{error:?}"), model))?;
        let columns: [[Matrix3; 3]; 3] = from_fn(|k| {
            from_fn(|l| {
                let d_f = basis(k, l);
                let d_m = sensitivities.linearization.mandel_derivative(&d_f, &ZERO);
                let (d_symmetric, d_magnitude) = sensitivities.direction_slope(&d_m);
                let mut rhs = Vector::zero(SIZE);
                (0..3).for_each(|i| (0..3).for_each(|j| rhs[3 * i + j] = x[9] * d_symmetric[i][j]));
                rhs[9] = -d_magnitude;
                let z = lu.solve(&rhs);
                let d_f_p: Matrix3 = from_fn(|i| {
                    from_fn(|j| {
                        (0..3)
                            .map(|a| {
                                (0..3)
                                    .map(|b| z[3 * a + b] * sensitivities.slopes[a][b][i][j])
                                    .sum::<f64>()
                            })
                            .sum()
                    })
                });
                sensitivities.linearization.stress_derivative(&d_f, &d_f_p)
            })
        });
        Ok(from_fn(|i| {
            from_fn(|j| from_fn(|k| from_fn(|l| columns[k][l][i][j])))
        }))
    }

    fn state(f_p: &DeformationGradientPlastic, strain: f64) -> PlasticStateVariables {
        (f_p.clone(), Quantity::new(strain)).into()
    }

    fn stretch_shear(s: f64) -> DeformationGradient {
        DeformationGradient::from([
            [1.0 + s, 0.7 * s, 0.2 * s],
            [0.0, 1.0 - 0.2 * s, 0.4 * s],
            [0.0, 0.0, 1.0 + 0.3 * s],
        ])
    }

    fn report<M: ElasticPlastic>(
        name: &str,
        model: &M,
        scenario: &str,
        cases: &[(f64, DeformationGradient, Plastic)],
    ) -> Result<(), ConstitutiveError> {
        println!("\n{name} / {scenario}");
        println!(
            "{:>7} {:>10} {:>12} {:>12} {:>12} {:>7} {:>10}",
            "step", "dgamma", "err Fp", "err eps_p", "err P", "sweeps", "last dFp"
        );
        for (step, f, start) in cases {
            let implicit = return_map_swept(model, f, start, 200);
            let frozen = match return_map_swept(model, f, start, 0) {
                Ok((frozen, ..)) => frozen,
                Err(_) => {
                    println!(
                        "{step:>7.3} frozen-direction map fails to bracket; implicit reference {}",
                        if implicit.is_ok() {
                            "succeeds"
                        } else {
                            "fails too"
                        }
                    );
                    continue;
                }
            };
            let library = model.return_map(f, &state(&start.0, start.1))?;
            let (library_f_p, &library_strain): (&DeformationGradientPlastic, &Quantity) =
                (&library).into();
            assert!(
                (library_f_p - &frozen.0).norm().value() < 1e-8
                    && (library_strain.value() - frozen.1).abs() < 1e-8,
                "sweep 0 must reproduce return_map"
            );
            let (converged, used, last) = match implicit {
                Ok(result) => result,
                Err(error) => {
                    println!("{step:>7.3} implicit reference failed: {error}");
                    continue;
                }
            };
            let stress =
                |f_p: &DeformationGradientPlastic| model.first_piola_kirchhoff_stress(f, f_p);
            let (p_frozen, p_converged) = (stress(&frozen.0)?, stress(&converged.0)?);
            println!(
                "{:>7.3} {:>10.3e} {:>12.3e} {:>12.3e} {:>12.3e} {:>7} {:>10.1e}",
                step,
                converged.1 - start.1,
                (&frozen.0 - &converged.0).norm().value() / converged.0.norm().value(),
                (frozen.1 - converged.1).abs() / (converged.1 - start.1).abs().max(1e-300),
                (&p_frozen - &p_converged).norm().value() / p_converged.norm().value().max(1e-300),
                used,
                last,
            );
        }
        Ok(())
    }

    type Case = (f64, DeformationGradient, Plastic);

    fn scenario_cases<M: ElasticPlastic>(
        model: &M,
    ) -> Result<Vec<(&'static str, Vec<Case>)>, ConstitutiveError> {
        let steps = [0.05, 0.1, 0.2, 0.4, 0.8, 1.6];
        let identity = (DeformationGradientPlastic::identity(), 0.0);
        let virgin: Vec<_> = steps
            .iter()
            .map(|&s| (s, stretch_shear(s), identity.clone()))
            .collect();
        let pre_load = [[1.8, 0.56, 0.16], [0.0, 0.84, 0.32], [0.0, 0.0, 1.24]];
        let mut pre_state = identity.clone();
        for k in 1..=64 {
            pre_state =
                return_map_swept(model, &stretch_shear(0.8 * k as f64 / 64.0), &pre_state, 30)?.0;
        }
        let rotated: Vec<_> = steps
            .iter()
            .map(|&s| {
                let shear = [[1.0, 0.0, 0.0], [0.6 * s, 1.0, 0.0], [0.0, -0.5 * s, 1.0]];
                let product: [[f64; 3]; 3] = std::array::from_fn(|i| {
                    std::array::from_fn(|j| (0..3).map(|k| pre_load[i][k] * shear[k][j]).sum())
                });
                (s, DeformationGradient::from(product), pre_state.clone())
            })
            .collect();
        Ok(vec![
            ("from virgin state, proportional", virgin),
            ("from pre-loaded state, rotated loading", rotated),
        ])
    }

    fn scenarios<M: ElasticPlastic>(name: &str, model: &M) -> Result<(), ConstitutiveError> {
        scenario_cases(model)?
            .iter()
            .try_for_each(|(scenario, cases)| report(name, model, scenario, cases))
    }

    #[test]
    #[ignore = "diagnostic: run with --ignored --nocapture"]
    fn frozen_vs_implicit_flow_direction() -> Result<(), ConstitutiveError> {
        macro_rules! run {
            ($elastic:ident) => {
                scenarios(
                    stringify!($elastic),
                    &Canonical::from((
                        $elastic {
                            bulk_modulus: Stress::pascals(13.0),
                            shear_modulus: Stress::pascals(3.0),
                        },
                        PlasticFlow {
                            yield_stress: Stress::pascals(2.0),
                            hardening_slope: Stress::pascals(1.0),
                        },
                    )),
                )?
            };
        }
        run!(Hencky);
        run!(NeoHookean);
        run!(SaintVenantKirchhoff);
        Ok(())
    }

    fn report_newton<M: ElasticPlastic>(name: &str, model: &M) -> Result<(), ConstitutiveError> {
        for (scenario, cases) in scenario_cases(model)? {
            println!("\n{name} / {scenario}");
            println!(
                "{:>7} {:>10} {:>12} {:>12} {:>7} {:>7} {:>10}",
                "step", "dgamma", "err Fp", "err eps_p", "iters", "cold", "|R|"
            );
            for (step, f, start) in &cases {
                let reference = match return_map_swept(model, f, start, 200) {
                    Ok((reference, ..)) => reference,
                    Err(_) => {
                        let describe = |guess| {
                            return_map_newton(model, f, start, guess, Jacobian::Analytic)
                                .map_or_else(
                                    |error| format!("fails ({error})").replace('\n', " "),
                                    |(_, iterations, residual)| {
                                        format!("converges in {iterations} (|R| {residual:.1e})")
                                    },
                                )
                        };
                        println!(
                            "{step:>7.3} sweep reference fails; Newton frozen-start {}, cold-start {}",
                            describe(Guess::Frozen),
                            describe(Guess::Cold),
                        );
                        continue;
                    }
                };
                let cold = return_map_newton(model, f, start, Guess::Cold, Jacobian::Analytic)
                    .map_or_else(
                        |_| "fail".to_string(),
                        |(_, iterations, _)| iterations.to_string(),
                    );
                match return_map_newton(model, f, start, Guess::Frozen, Jacobian::Analytic) {
                    Ok((newton, iterations, residual)) => println!(
                        "{:>7.3} {:>10.3e} {:>12.3e} {:>12.3e} {:>7} {:>7} {:>10.1e}",
                        step,
                        reference.1 - start.1,
                        (&newton.0 - &reference.0).norm().value() / reference.0.norm().value(),
                        (newton.1 - reference.1).abs() / (reference.1 - start.1).abs().max(1e-300),
                        iterations,
                        cold,
                        residual,
                    ),
                    Err(error) => println!("{step:>7.3} Newton failed: {error}"),
                }
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "diagnostic: run with --ignored --nocapture"]
    fn coupled_newton_vs_reference() -> Result<(), ConstitutiveError> {
        macro_rules! run {
            ($elastic:ident) => {
                report_newton(
                    stringify!($elastic),
                    &Canonical::from((
                        $elastic {
                            bulk_modulus: Stress::pascals(13.0),
                            shear_modulus: Stress::pascals(3.0),
                        },
                        PlasticFlow {
                            yield_stress: Stress::pascals(2.0),
                            hardening_slope: Stress::pascals(1.0),
                        },
                    )),
                )?
            };
        }
        run!(Hencky);
        run!(NeoHookean);
        run!(SaintVenantKirchhoff);
        Ok(())
    }

    macro_rules! for_each_model {
        ($check:ident) => {
            macro_rules! run {
                ($elastic:ident) => {
                    $check(
                        stringify!($elastic),
                        &Canonical::from((
                            $elastic {
                                bulk_modulus: Stress::pascals(13.0),
                                shear_modulus: Stress::pascals(3.0),
                            },
                            PlasticFlow {
                                yield_stress: Stress::pascals(2.0),
                                hardening_slope: Stress::pascals(1.0),
                            },
                        )),
                    )?
                };
            }
            run!(Hencky);
            run!(NeoHookean);
            run!(SaintVenantKirchhoff);
        };
    }

    fn check_jacobian<M: ElasticPlastic>(name: &str, model: &M) -> Result<(), ConstitutiveError> {
        let f = stretch_shear(0.4);
        let state = (DeformationGradientPlastic::identity(), 0.0);
        let n0 = direction(model, &f, &state.0)?;
        let skew = [[0.2, 0.5, -0.3], [0.1, -0.4, 0.6], [0.2, 0.0, 0.2]];
        let mut x = [0.0; SIZE];
        (0..3).for_each(|i| {
            (0..3).for_each(|j| x[3 * i + j] = 0.02 * n0[i][j].value() + 0.004 * skew[i][j])
        });
        x[9] = 0.02;
        let analytic = analytic_jacobian(
            &Sensitivities::new(model, &f, &state.0, &x)?,
            x[9],
            model.hardening_slope().value(),
        );
        let finite_difference = finite_difference_jacobian(model, &f, &state.0, state.1, &x)?;
        let (analytic_solution, ..) =
            return_map_newton(model, &f, &state, Guess::Cold, Jacobian::Analytic)?;
        let (finite_difference_solution, ..) =
            return_map_newton(model, &f, &state, Guess::Cold, Jacobian::FiniteDifference)?;
        assert!(
            (&analytic_solution.0 - &finite_difference_solution.0)
                .norm()
                .value()
                < 1e-10,
            "{name}: Newton with either Jacobian must reach the same solution",
        );
        for row in 0..SIZE {
            for column in 0..SIZE {
                let (a, b) = (analytic[row][column], finite_difference[row][column]);
                assert!(
                    (a - b).abs() <= 1e-6 * (1.0 + a.abs()),
                    "{name}: J[{row}][{column}] analytic {a} vs finite difference {b}",
                );
            }
        }
        Ok(())
    }

    fn check_tangent<M: ElasticPlastic>(name: &str, model: &M) -> Result<(), ConstitutiveError> {
        let identity = (DeformationGradientPlastic::identity(), 0.0);
        let pre_state = return_map_newton(
            model,
            &stretch_shear(0.5),
            &identity,
            Guess::Cold,
            Jacobian::Analytic,
        )?
        .0;
        let rotated = {
            let pre_load = matrix_3(&stretch_shear(0.5));
            let shear = [[1.0, 0.0, 0.0], [0.18, 1.0, 0.0], [0.0, -0.15, 1.0]];
            DeformationGradient::from(mul(&pre_load, &shear))
        };
        let cases = [
            (stretch_shear(0.3), identity.clone()),
            (stretch_shear(0.6), identity.clone()),
            (stretch_shear(0.9), identity.clone()),
            (rotated, pre_state),
        ];
        let stress_after = |f: &DeformationGradient, state: &Plastic| {
            let (f_p, ..) = return_map_newton(model, f, state, Guess::Cold, Jacobian::Analytic)?.0;
            Ok::<_, ConstitutiveError>(matrix_3(&model.first_piola_kirchhoff_stress(f, &f_p)?))
        };
        for (case, (f, state)) in cases.iter().enumerate() {
            let tangent = consistent_tangent(model, f, state)?;
            let h = 1e-4;
            let (mut worst, mut scale) = (0.0_f64, 0.0_f64);
            for k in 0..3 {
                for l in 0..3 {
                    let (mut plus, mut minus) = (f.clone(), f.clone());
                    plus[k][l] += crate::math::assert::perturbation(h);
                    minus[k][l] -= crate::math::assert::perturbation(h);
                    let (p_plus, p_minus) =
                        (stress_after(&plus, state)?, stress_after(&minus, state)?);
                    for i in 0..3 {
                        for j in 0..3 {
                            let finite_difference = (p_plus[i][j] - p_minus[i][j]) / (2.0 * h);
                            let a = tangent[i][j][k][l];
                            worst = worst.max((a - finite_difference).abs() / (1.0 + a.abs()));
                            scale = scale.max(a.abs());
                        }
                    }
                }
            }
            let versus_frozen = model
                .consistent_tangent_stiffness(f, &self::state(&state.0, state.1))
                .map_or_else(
                    |_| "n/a (frozen return map fails)".to_string(),
                    |(frozen, _)| {
                        let frozen = entries_4(&frozen);
                        let difference = (0..81)
                            .map(|n| {
                                let (i, j, k, l) = (n / 27, n / 9 % 3, n / 3 % 3, n % 3);
                                (tangent[i][j][k][l] - frozen[i][j][k][l]).abs()
                            })
                            .fold(0.0_f64, f64::max);
                        format!("{difference:.2e}")
                    },
                );
            println!(
                "{name} case {case}: |tangent| max {scale:.3e}, worst rel err vs finite \
                 difference {worst:.2e}, max |implicit - frozen library tangent| {versus_frozen}"
            );
            assert!(
                worst < 1e-5,
                "{name} case {case}: tangent error {worst:.3e}"
            );
        }
        Ok(())
    }

    #[test]
    fn analytic_jacobian_matches_finite_difference() -> Result<(), ConstitutiveError> {
        for_each_model!(check_jacobian);
        Ok(())
    }

    #[test]
    fn consistent_tangent_matches_finite_difference() -> Result<(), ConstitutiveError> {
        for_each_model!(check_tangent);
        Ok(())
    }

    fn microseconds<T>(mut call: impl FnMut() -> Result<T, ConstitutiveError>) -> Option<f64> {
        const REPETITIONS: u32 = 300;
        call().ok()?;
        let start = std::time::Instant::now();
        for _ in 0..REPETITIONS {
            std::hint::black_box(call().ok()?);
        }
        Some(start.elapsed().as_secs_f64() * 1e6 / f64::from(REPETITIONS))
    }

    fn cost<M: ElasticPlastic>(name: &str, model: &M) -> Result<(), ConstitutiveError> {
        let identity = (DeformationGradientPlastic::identity(), 0.0);
        let pre_state = return_map_newton(
            model,
            &stretch_shear(0.5),
            &identity,
            Guess::Cold,
            Jacobian::Analytic,
        )?
        .0;
        let rotated = {
            let pre_load = matrix_3(&stretch_shear(0.5));
            let shear = [[1.0, 0.0, 0.0], [0.05, 1.0, 0.0], [0.0, -0.04, 1.0]];
            DeformationGradient::from(mul(&pre_load, &shear))
        };
        let cases = [
            ("virgin, step 0.4", stretch_shear(0.4), identity.clone()),
            ("virgin, step 0.8", stretch_shear(0.8), identity),
            ("pre-loaded, small rotated step", rotated, pre_state),
        ];
        println!("\n{name} (microseconds per call)");
        println!(
            "{:<32} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9}",
            "case", "frozen", "newton", "newton FD", "tan lib", "tan impl", "iters"
        );
        let show = |value: Option<f64>| value.map_or("n/a".to_string(), |v| format!("{v:.1}"));
        for (label, f, state) in &cases {
            let frozen = microseconds(|| model.return_map(f, &self::state(&state.0, state.1)));
            let newton = microseconds(|| {
                return_map_newton(model, f, state, Guess::Cold, Jacobian::Analytic)
            });
            let newton_fd = microseconds(|| {
                return_map_newton(model, f, state, Guess::Cold, Jacobian::FiniteDifference)
            });
            let tangent_library = microseconds(|| {
                model.consistent_tangent_stiffness(f, &self::state(&state.0, state.1))
            });
            let tangent = microseconds(|| consistent_tangent(model, f, state));
            let iterations = return_map_newton(model, f, state, Guess::Cold, Jacobian::Analytic)
                .map_or(0, |(_, iterations, _)| iterations);
            println!(
                "{label:<32} {:>9} {:>9} {:>9} {:>9} {:>9} {iterations:>9}",
                show(frozen),
                show(newton),
                show(newton_fd),
                show(tangent_library),
                show(tangent),
            );
        }
        Ok(())
    }

    #[test]
    #[ignore = "benchmark: cargo test --profile release-dev -F constitutive --lib cost_frozen -- --ignored --nocapture"]
    fn cost_frozen_vs_coupled_newton() -> Result<(), ConstitutiveError> {
        for_each_model!(cost);
        Ok(())
    }
}

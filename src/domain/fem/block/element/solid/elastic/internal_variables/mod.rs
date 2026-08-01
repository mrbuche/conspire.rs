use crate::{
    ABS_TOL,
    constitutive::{ConstitutiveError, solid::elastic::internal_variables::ElasticIV},
    fem::block::element::{
        Element, ElementNodalCoordinates, FiniteElement, FiniteElementError, GradientVectors,
        solid::{ElementNodalForcesSolid, ElementNodalStiffnessesSolid, SolidFiniteElement},
    },
    math::{
        ContractSecondFourthWithFirst, HessianBlock, Jacobian, Scalar, ScalarList, Solution,
        SquareMatrix, Tensor, TensorList, Vector,
    },
    mechanics::{
        DeformationGradient, FirstPiolaKirchhoffStress, FirstPiolaKirchhoffStressList,
        FirstPiolaKirchhoffTangentStiffness, FirstPiolaKirchhoffTangentStiffnessList,
    },
};

/// The most Newton steps an integration point is given.
const MAX_STEPS: usize = 25;

/// The factor by which a step is shortened when it does not decrease.
const CUT_BACK: Scalar = 5e-1;

/// The shortest step worth trying before giving up.
const MIN_FRACTION: Scalar = 1e-8;

/// The indices of the internal variables that are free to move.
fn free_indices<C, V, T1, T2, T3>(constitutive_model: &C, size: usize) -> Vec<usize>
where
    C: ElasticIV<V, T1, T2, T3>,
{
    let mut free = vec![true; size];
    constitutive_model
        .internal_variables_fixed()
        .iter()
        .for_each(|&index| free[index] = false);
    (0..size).filter(|&index| free[index]).collect()
}

/// The internal variables held at each integration point of an element.
pub type InternalVariables<const G: usize, V> = TensorList<V, G>;

pub trait ElasticIVFiniteElement<
    C,
    const G: usize,
    const M: usize,
    const N: usize,
    const P: usize,
    V,
    T1,
    T2,
    T3,
> where
    C: ElasticIV<V, T1, T2, T3>,
    Self: SolidFiniteElement<G, M, N, P>,
    T1: HessianBlock,
    T2: HessianBlock,
    T3: HessianBlock,
    V: Jacobian + Solution,
{
    /// The internal variables an element starts from at every integration point.
    fn internal_variables_initial(&self, constitutive_model: &C) -> InternalVariables<G, V>;
    /// Solves the internal variables at every integration point, holding the
    /// deformation fixed.
    ///
    /// The integration points are independent of one another, the internal
    /// variables of one never entering the residual of another.
    fn internal_variables_root(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &ElementNodalCoordinates<N>,
        internal_variables: &InternalVariables<G, V>,
    ) -> Result<InternalVariables<G, V>, FiniteElementError>;
    /// Steps the internal variables alongside a decrement of the nodal
    /// coordinates, rather than solving them where the coordinates now are.
    fn internal_variables_increment(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &ElementNodalCoordinates<N>,
        internal_variables: &InternalVariables<G, V>,
        nodal_decrement: &ElementNodalCoordinates<N>,
    ) -> Result<InternalVariables<G, V>, FiniteElementError>;
    fn nodal_forces(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &ElementNodalCoordinates<N>,
        internal_variables: &InternalVariables<G, V>,
    ) -> Result<ElementNodalForcesSolid<N>, FiniteElementError>;
    /// The nodal forces with the residual of the internal variables eliminated
    /// into them, for when they are carried rather than solved.
    fn nodal_forces_eliminated(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &ElementNodalCoordinates<N>,
        internal_variables: &InternalVariables<G, V>,
    ) -> Result<ElementNodalForcesSolid<N>, FiniteElementError>;
    /// The tangent stiffnesses with the internal variables condensed out.
    ///
    /// ```math
    /// \mathcal{C} = \mathcal{K}_{uu} - \mathcal{K}_{uv}\mathcal{K}_{vv}^{-1}\mathcal{K}_{vu}
    /// ```
    fn nodal_stiffnesses(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &ElementNodalCoordinates<N>,
        internal_variables: &InternalVariables<G, V>,
    ) -> Result<ElementNodalStiffnessesSolid<N>, FiniteElementError>;
}

/// Solves the internal variables at one integration point.
///
/// The gauge freedom is fixed at the initial values, which are zero over those
/// indices, so the constraint is imposed by leaving them out of the system
/// rather than by a multiplier.
fn root_at_point<C, V, T1, T2, T3>(
    constitutive_model: &C,
    deformation_gradient: &DeformationGradient,
    internal_variables: &V,
) -> Result<V, ConstitutiveError>
where
    C: ElasticIV<V, T1, T2, T3>,
    T3: HessianBlock,
    V: Jacobian + Solution,
{
    let size = internal_variables.size();
    let unmap = free_indices(constitutive_model, size);
    let mut root = internal_variables.clone();
    let mut residual = Vector::zero(size);
    let mut reduced = Vector::zero(unmap.len());
    let mut decrement = Vector::zero(size);
    let mut block = SquareMatrix::zero(size);
    let mut local = SquareMatrix::zero(unmap.len());
    for _ in 0..MAX_STEPS {
        constitutive_model
            .internal_variables_residual(deformation_gradient, &root)?
            .fill_into(&mut residual);
        unmap
            .iter()
            .enumerate()
            .for_each(|(a, &i)| reduced[a] = residual[i]);
        if reduced.iter().fold(0.0, |m: Scalar, r| m.max(r.abs())) < ABS_TOL {
            return Ok(root);
        }
        let (_, _, _, tangent) = constitutive_model.tangents(deformation_gradient, &root)?;
        tangent.fill_into_block(&mut block, 0, 0);
        unmap.iter().enumerate().for_each(|(a, &i)| {
            unmap
                .iter()
                .enumerate()
                .for_each(|(b, &j)| local[a][b] = block[i][j])
        });
        let solution = local.solve_lu(&reduced).map_err(|error| {
            ConstitutiveError::Custom(format!("{error:?}"), format!("{deformation_gradient}"))
        })?;
        decrement.iter_mut().for_each(|entry| *entry = 0.0);
        unmap
            .iter()
            .enumerate()
            .for_each(|(a, &i)| decrement[i] = solution[a]);
        //
        // A full step can carry the internal variables somewhere the model
        // cannot be evaluated at all, so shorten it until the residual both
        // exists and has decreased.
        //
        let residual_norm = reduced.iter().fold(0.0, |m: Scalar, r| m.max(r.abs()));
        let mut fraction = 1.0;
        loop {
            let mut trial = root.clone();
            trial.decrement_from(&(&decrement * fraction));
            if let Ok(trial_residual) =
                constitutive_model.internal_variables_residual(deformation_gradient, &trial)
            {
                trial_residual.fill_into(&mut residual);
                if unmap
                    .iter()
                    .fold(0.0, |m: Scalar, &i| m.max(residual[i].abs()))
                    < residual_norm
                {
                    root = trial;
                    break;
                }
            }
            fraction *= CUT_BACK;
            if fraction < MIN_FRACTION {
                return Err(ConstitutiveError::Custom(
                    "The internal variables could not be decreased.".to_string(),
                    format!("{deformation_gradient}"),
                ));
            }
        }
    }
    Err(ConstitutiveError::Custom(
        "The internal variables did not converge.".to_string(),
        format!("{deformation_gradient}"),
    ))
}

/// The nodal forces a list of stresses integrates to.
fn assemble_forces<const G: usize, const N: usize>(
    stresses: FirstPiolaKirchhoffStressList<G>,
    gradient_vectors: &GradientVectors<3, G, N>,
    integration_weights: &ScalarList<G>,
) -> ElementNodalForcesSolid<N> {
    stresses
        .iter()
        .zip(gradient_vectors.iter().zip(integration_weights))
        .map(|(stress, (gradient_vectors_point, integration_weight))| {
            gradient_vectors_point
                .iter()
                .map(|gradient_vector| (stress * gradient_vector) * integration_weight)
                .collect()
        })
        .sum()
}

/// The local block over the free internal variables, the fixed ones leaving it
/// through their rows and columns rather than through the inverse.
fn local_block<T3>(tangent_vv: &T3, size: usize, unmap: &[usize]) -> SquareMatrix
where
    T3: HessianBlock,
{
    let mut block = SquareMatrix::zero(size);
    tangent_vv.fill_into_block(&mut block, 0, 0);
    let mut local = SquareMatrix::zero(unmap.len());
    unmap.iter().enumerate().for_each(|(a, &i)| {
        unmap
            .iter()
            .enumerate()
            .for_each(|(b, &j)| local[a][b] = block[i][j])
    });
    local
}

/// The stress at one integration point with the residual of the internal
/// variables eliminated into it.
///
/// ```math
/// \mathbf{P} - \mathcal{K}_{uv}\mathcal{K}_{vv}^{-1}\mathbf{r}_v
/// ```
///
/// The internal variables are carried rather than solved, so this residual
/// only agrees with the stress alone once they have converged.
fn eliminated_at_point<C, V, T1, T2, T3>(
    constitutive_model: &C,
    deformation_gradient: &DeformationGradient,
    internal_variables: &V,
) -> Result<FirstPiolaKirchhoffStress, ConstitutiveError>
where
    C: ElasticIV<V, T1, T2, T3>,
    T1: HessianBlock,
    T2: HessianBlock,
    T3: HessianBlock,
    V: Jacobian + Solution,
{
    let mut stress = constitutive_model
        .first_piola_kirchhoff_stress(deformation_gradient, internal_variables)?;
    let size = internal_variables.size();
    let unmap = free_indices(constitutive_model, size);
    let mut residual = Vector::zero(size);
    constitutive_model
        .internal_variables_residual(deformation_gradient, internal_variables)?
        .fill_into(&mut residual);
    let mut reduced = Vector::zero(unmap.len());
    unmap
        .iter()
        .enumerate()
        .for_each(|(a, &i)| reduced[a] = residual[i]);
    let (_, _, tangent_uv, tangent_vv) =
        constitutive_model.tangents(deformation_gradient, internal_variables)?;
    let eliminated = local_block(&tangent_vv, size, &unmap)
        .solve_lu(&reduced)
        .map_err(|error| {
            ConstitutiveError::Custom(format!("{error:?}"), format!("{deformation_gradient}"))
        })?;
    let mut cross = SquareMatrix::zero(size);
    tangent_uv.fill_into_block(&mut cross, 0, 0);
    (0..3).for_each(|i| {
        (0..3).for_each(|j| {
            stress[i][j] -= unmap
                .iter()
                .enumerate()
                .map(|(a, &v)| cross[3 * i + j][v] * eliminated[a])
                .sum::<Scalar>()
        })
    });
    Ok(stress)
}

/// The internal variables at one integration point stepped alongside a
/// decrement of the deformation.
///
/// ```math
/// \Delta\mathbf{v} = -\mathcal{K}_{vv}^{-1}\left(\mathbf{r}_v + \mathcal{K}_{vu}\Delta\mathbf{u}\right)
/// ```
fn increment_at_point<C, V, T1, T2, T3>(
    constitutive_model: &C,
    deformation_gradient: &DeformationGradient,
    deformation_gradient_decrement: &DeformationGradient,
    internal_variables: &V,
) -> Result<V, ConstitutiveError>
where
    C: ElasticIV<V, T1, T2, T3>,
    T1: HessianBlock,
    T2: HessianBlock,
    T3: HessianBlock,
    V: Jacobian + Solution,
{
    let size = internal_variables.size();
    let unmap = free_indices(constitutive_model, size);
    let mut residual = Vector::zero(size);
    constitutive_model
        .internal_variables_residual(deformation_gradient, internal_variables)?
        .fill_into(&mut residual);
    let (_, tangent_vu, _, tangent_vv) =
        constitutive_model.tangents(deformation_gradient, internal_variables)?;
    let mut coupling = SquareMatrix::zero(size);
    tangent_vu.fill_into_block(&mut coupling, 0, 0);
    //
    // The deformation is handed over as a decrement, so its contribution to
    // the local residual enters with the opposite sign.
    //
    let mut reduced = Vector::zero(unmap.len());
    unmap.iter().enumerate().for_each(|(a, &i)| {
        reduced[a] = residual[i]
            - (0..3)
                .map(|k| {
                    (0..3)
                        .map(|l| coupling[i][3 * k + l] * deformation_gradient_decrement[k][l])
                        .sum::<Scalar>()
                })
                .sum::<Scalar>()
    });
    let solution = local_block(&tangent_vv, size, &unmap)
        .solve_lu(&reduced)
        .map_err(|error| {
            ConstitutiveError::Custom(format!("{error:?}"), format!("{deformation_gradient}"))
        })?;
    let mut decrement = Vector::zero(size);
    unmap
        .iter()
        .enumerate()
        .for_each(|(a, &i)| decrement[i] = solution[a]);
    let mut incremented = internal_variables.clone();
    incremented.decrement_from(&decrement);
    Ok(incremented)
}

/// The tangent stiffness at one integration point with the internal variables
/// condensed out.
fn condensed_at_point<C, V, T1, T2, T3>(
    constitutive_model: &C,
    deformation_gradient: &DeformationGradient,
    internal_variables: &V,
) -> Result<FirstPiolaKirchhoffTangentStiffness, ConstitutiveError>
where
    C: ElasticIV<V, T1, T2, T3>,
    T1: HessianBlock,
    T2: HessianBlock,
    T3: HessianBlock,
    V: Tensor,
{
    let (tangent_uu, tangent_vu, tangent_uv, tangent_vv) =
        constitutive_model.tangents(deformation_gradient, internal_variables)?;
    let size = internal_variables.size();
    let unmap = free_indices(constitutive_model, size);
    let factorization = local_block(&tangent_vv, size, &unmap)
        .factorize_lu()
        .map_err(|error| {
            ConstitutiveError::Custom(format!("{error:?}"), format!("{deformation_gradient}"))
        })?;
    let mut coupling = SquareMatrix::zero(size);
    tangent_vu.fill_into_block(&mut coupling, 0, 0);
    let mut cross = SquareMatrix::zero(size);
    tangent_uv.fill_into_block(&mut cross, 0, 0);
    let mut column = Vector::zero(unmap.len());
    let mut eliminated = vec![Vector::zero(unmap.len()); size];
    (0..size).for_each(|c| {
        unmap
            .iter()
            .enumerate()
            .for_each(|(a, &i)| column[a] = coupling[i][c]);
        factorization.solve_into(&column, &mut eliminated[c])
    });
    let mut condensed = tangent_uu;
    (0..3).for_each(|i| {
        (0..3).for_each(|j| {
            (0..3).for_each(|k| {
                (0..3).for_each(|l| {
                    condensed[i][j][k][l] -= unmap
                        .iter()
                        .enumerate()
                        .map(|(a, &v)| cross[3 * i + j][v] * eliminated[3 * k + l][a])
                        .sum::<Scalar>()
                })
            })
        })
    });
    Ok(condensed)
}

impl<C, const G: usize, const N: usize, const O: usize, const P: usize, V, T1, T2, T3>
    ElasticIVFiniteElement<C, G, 3, N, P, V, T1, T2, T3> for Element<3, G, N, O>
where
    C: ElasticIV<V, T1, T2, T3>,
    Self: SolidFiniteElement<G, 3, N, P>,
    T1: HessianBlock,
    T2: HessianBlock,
    T3: HessianBlock,
    V: Jacobian + Solution,
{
    fn internal_variables_initial(&self, constitutive_model: &C) -> InternalVariables<G, V> {
        std::array::from_fn(|_| constitutive_model.internal_variables_initial()).into()
    }
    fn internal_variables_root(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &ElementNodalCoordinates<N>,
        internal_variables: &InternalVariables<G, V>,
    ) -> Result<InternalVariables<G, V>, FiniteElementError> {
        match self
            .deformation_gradients(nodal_coordinates)
            .iter()
            .zip(internal_variables)
            .map(|(deformation_gradient, internal_variables_point)| {
                root_at_point(
                    constitutive_model,
                    deformation_gradient,
                    internal_variables_point,
                )
            })
            .collect::<Result<InternalVariables<G, V>, _>>()
        {
            Ok(roots) => Ok(roots),
            Err(error) => Err(FiniteElementError::Upstream(
                format!("{error}"),
                format!("{self:?}"),
            )),
        }
    }
    fn internal_variables_increment(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &ElementNodalCoordinates<N>,
        internal_variables: &InternalVariables<G, V>,
        nodal_decrement: &ElementNodalCoordinates<N>,
    ) -> Result<InternalVariables<G, V>, FiniteElementError> {
        match self
            .deformation_gradients(nodal_coordinates)
            .iter()
            .zip(self.deformation_gradients(nodal_decrement).iter())
            .zip(internal_variables)
            .map(
                |((deformation_gradient, decrement), internal_variables_point)| {
                    increment_at_point(
                        constitutive_model,
                        deformation_gradient,
                        decrement,
                        internal_variables_point,
                    )
                },
            )
            .collect::<Result<InternalVariables<G, V>, _>>()
        {
            Ok(incremented) => Ok(incremented),
            Err(error) => Err(FiniteElementError::Upstream(
                format!("{error}"),
                format!("{self:?}"),
            )),
        }
    }
    fn nodal_forces(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &ElementNodalCoordinates<N>,
        internal_variables: &InternalVariables<G, V>,
    ) -> Result<ElementNodalForcesSolid<N>, FiniteElementError> {
        match self
            .deformation_gradients(nodal_coordinates)
            .iter()
            .zip(internal_variables)
            .map(|(deformation_gradient, internal_variables_point)| {
                constitutive_model
                    .first_piola_kirchhoff_stress(deformation_gradient, internal_variables_point)
            })
            .collect::<Result<FirstPiolaKirchhoffStressList<G>, _>>()
        {
            Ok(stresses) => Ok(assemble_forces(
                stresses,
                self.gradient_vectors(),
                self.integration_weights(),
            )),
            Err(error) => Err(FiniteElementError::Upstream(
                format!("{error}"),
                format!("{self:?}"),
            )),
        }
    }
    fn nodal_forces_eliminated(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &ElementNodalCoordinates<N>,
        internal_variables: &InternalVariables<G, V>,
    ) -> Result<ElementNodalForcesSolid<N>, FiniteElementError> {
        match self
            .deformation_gradients(nodal_coordinates)
            .iter()
            .zip(internal_variables)
            .map(|(deformation_gradient, internal_variables_point)| {
                eliminated_at_point(
                    constitutive_model,
                    deformation_gradient,
                    internal_variables_point,
                )
            })
            .collect::<Result<FirstPiolaKirchhoffStressList<G>, _>>()
        {
            Ok(stresses) => Ok(assemble_forces(
                stresses,
                self.gradient_vectors(),
                self.integration_weights(),
            )),
            Err(error) => Err(FiniteElementError::Upstream(
                format!("{error}"),
                format!("{self:?}"),
            )),
        }
    }
    fn nodal_stiffnesses(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &ElementNodalCoordinates<N>,
        internal_variables: &InternalVariables<G, V>,
    ) -> Result<ElementNodalStiffnessesSolid<N>, FiniteElementError> {
        match self
            .deformation_gradients(nodal_coordinates)
            .iter()
            .zip(internal_variables)
            .map(|(deformation_gradient, internal_variables_point)| {
                condensed_at_point(
                    constitutive_model,
                    deformation_gradient,
                    internal_variables_point,
                )
            })
            .collect::<Result<FirstPiolaKirchhoffTangentStiffnessList<G>, _>>()
        {
            Ok(condensed) => Ok(condensed
                .iter()
                .zip(
                    self.gradient_vectors()
                        .iter()
                        .zip(self.integration_weights()),
                )
                .map(|(tangent, (gradient_vectors, integration_weight))| {
                    gradient_vectors
                        .iter()
                        .map(|gradient_vector_a| {
                            gradient_vectors
                                .iter()
                                .map(|gradient_vector_b| {
                                    tangent.contract_second_fourth_with_first(
                                        gradient_vector_a,
                                        gradient_vector_b,
                                    ) * integration_weight
                                })
                                .collect()
                        })
                        .collect()
                })
                .sum()),
            Err(error) => Err(FiniteElementError::Upstream(
                format!("{error}"),
                format!("{self:?}"),
            )),
        }
    }
}

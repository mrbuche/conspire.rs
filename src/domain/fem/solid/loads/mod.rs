#[cfg(test)]
mod test;

use crate::{
    domain::{
        boundary::{Load, LoadAt, LoadStiffness, LoadStiffnessAt, Loads},
        multiphysics::solid_thermal::{
            NodalForcesSolidThermal, NodalStatesSolidThermal, NodalStiffnessesSolidThermal,
            split_states,
        },
        solid::loads::{DeadLoad, SolidLayout, TimedLoad},
    },
    fem::{
        NodalCoordinates, NodalReferenceCoordinates,
        block::element::{
            ElementNodalCoordinates, ElementNodalReferenceCoordinates,
            surface::{
                SurfaceFiniteElement,
                linear::{Quadrilateral, Triangle},
            },
        },
        solid::{NodalForcesSolid, NodalStiffnessesSolid},
    },
    geometry::mesh::SideSet,
    math::{Current, Quantity, Tensor, TensorRank2List2D, TensorTuple},
    mechanics::{ForceList, Traction},
    units::{Area, ForcePerLength, Stress, Time},
};

/// The area that each node has of the faces of a side set.
///
/// Fails if the side set has no faces, or a face that is not a triangle or a quadrilateral.
pub fn nodal_areas(
    coordinates: &NodalReferenceCoordinates<3>,
    side_set: &SideSet,
) -> Result<Vec<Quantity<Area>>, String> {
    if side_set.faces().len() != side_set.sides().len() {
        return Err("The side set has no faces on the mesh.".to_string());
    }
    let mut areas = vec![Quantity::<Area>::new(0.0); coordinates.len()];
    side_set.faces().iter().try_for_each(|face| {
        let nodal_areas = match *face.as_slice() {
            [a, b, c] => Triangle::nodal_areas(&ElementNodalReferenceCoordinates::from([
                coordinates[a].clone(),
                coordinates[b].clone(),
                coordinates[c].clone(),
            ])),
            [a, b, c, d] => Quadrilateral::nodal_areas(&ElementNodalReferenceCoordinates::from([
                coordinates[a].clone(),
                coordinates[b].clone(),
                coordinates[c].clone(),
                coordinates[d].clone(),
            ])),
            _ => {
                return Err(format!(
                    "A face with {} nodes is not a triangle or a quadrilateral.",
                    face.len()
                ));
            }
        };
        face.iter()
            .zip(nodal_areas)
            .for_each(|(&node, area)| areas[node] += area);
        Ok(())
    })?;
    Ok(areas)
}

/// The nodal forces of a dead traction per unit reference area on the faces of a side set.
///
/// Fails if the side set has no faces, or a face that is not a triangle or a quadrilateral.
pub fn traction_forces(
    coordinates: &NodalReferenceCoordinates<3>,
    side_set: &SideSet,
    traction: &Traction,
) -> Result<NodalForcesSolid<3>, String> {
    Ok(nodal_areas(coordinates, side_set)?
        .into_iter()
        .map(|area| traction * area)
        .collect())
}

impl<L, T> Loads<L, T>
where
    L: SolidLayout,
{
    /// Adds a dead traction per unit reference area on the faces of a side set.
    pub fn traction(
        self,
        coordinates: &NodalReferenceCoordinates<3>,
        side_set: &SideSet,
        traction: &Traction,
    ) -> Result<Loads<L, (T, DeadLoad)>, String> {
        Ok(self.with(DeadLoad::new(
            coordinates,
            traction_forces(coordinates, side_set, traction)?,
        )?))
    }
}

/// A traction per unit reference area on the faces of a side set, whose force does not change
/// as the body deforms, and is a function of time.
pub type TimedTraction<G> = TimedLoad<Area, G>;

type WithTimedTraction<L, T, G> = Result<Loads<L, (T, TimedTraction<G>)>, String>;

impl<L, T> Loads<L, T>
where
    L: SolidLayout,
{
    /// Adds a dead traction per unit reference area on the faces of a side set, that is a
    /// function of time.
    pub fn traction_with<G>(
        self,
        coordinates: &NodalReferenceCoordinates<3>,
        side_set: &SideSet,
        traction: G,
    ) -> WithTimedTraction<L, T, G>
    where
        G: Fn(Quantity<Time>) -> Traction,
    {
        Ok(self.with(TimedLoad::new(
            coordinates,
            nodal_areas(coordinates, side_set)?,
            traction,
        )?))
    }
}

/// The faces of a side set as lists of nodes of triangles and quadrilaterals.
fn faces_of(side_set: &SideSet) -> Result<Vec<Vec<usize>>, String> {
    if side_set.faces().len() != side_set.sides().len() {
        return Err("The side set has no faces on the mesh.".to_string());
    }
    side_set
        .faces()
        .iter()
        .try_for_each(|face| match face.len() {
            3 | 4 => Ok(()),
            nodes => Err(format!(
                "A face with {nodes} nodes is not a triangle or a quadrilateral."
            )),
        })?;
    Ok(side_set.faces().to_vec())
}

/// A pressure on the faces of a side set that follows them as the body deforms, which pushes into
/// the body when it is positive.
pub struct Pressure {
    faces: Vec<Vec<usize>>,
    pressure: Quantity<Stress>,
}

/// A pressure on the faces of a side set that follows them as the body deforms, and is a function
/// of time.
pub struct TimedPressure<G> {
    faces: Vec<Vec<usize>>,
    pressure: G,
}

impl<L, T> Loads<L, T>
where
    L: SolidLayout,
{
    /// Adds a pressure on the faces of a side set, which follows them as the body deforms.
    pub fn pressure(
        self,
        side_set: &SideSet,
        pressure: Quantity<Stress>,
    ) -> Result<Loads<L, (T, Pressure)>, String> {
        Ok(self.with(Pressure {
            faces: faces_of(side_set)?,
            pressure,
        }))
    }
    /// Adds a pressure on the faces of a side set, which follows them as the body deforms, and is
    /// a function of time.
    pub fn pressure_with<G>(
        self,
        side_set: &SideSet,
        pressure: G,
    ) -> Result<Loads<L, (T, TimedPressure<G>)>, String>
    where
        G: Fn(Quantity<Time>) -> Quantity<Stress>,
    {
        Ok(self.with(TimedPressure {
            faces: faces_of(side_set)?,
            pressure,
        }))
    }
}

fn face_coordinates<const N: usize>(
    coordinates: &NodalCoordinates<3>,
    face: &[usize],
) -> ElementNodalCoordinates<N> {
    face.iter().map(|&node| coordinates[node].clone()).collect()
}

fn check(faces: &[Vec<usize>], coordinates: &NodalCoordinates<3>) -> Result<(), String> {
    match faces
        .iter()
        .flatten()
        .any(|&node| node >= coordinates.len())
    {
        true => Err("The pressure is on a mesh with fewer nodes.".to_string()),
        false => Ok(()),
    }
}

fn subtract_forces<const N: usize>(
    face: &[usize],
    forces: ForceList<N>,
    residual: &mut NodalForcesSolid<3>,
) {
    face.iter()
        .zip(forces)
        .for_each(|(&node, force)| residual[node] -= force)
}

fn subtract_stiffnesses<const N: usize>(
    face: &[usize],
    blocks: TensorRank2List2D<3, Current, Current, N, N, ForcePerLength>,
    tangent: &mut NodalStiffnessesSolid<3>,
) {
    face.iter().zip(blocks).for_each(|(&node_a, row)| {
        face.iter()
            .zip(row)
            .for_each(|(&node_b, block)| tangent[node_a][node_b] -= block)
    })
}

fn subtract_pressure(
    faces: &[Vec<usize>],
    pressure: Quantity<Stress>,
    coordinates: &NodalCoordinates<3>,
    residual: &mut NodalForcesSolid<3>,
) -> Result<(), String> {
    check(faces, coordinates)?;
    faces.iter().for_each(|face| match face.len() {
        3 => subtract_forces(
            face,
            Triangle::pressure_forces(&face_coordinates(coordinates, face), pressure),
            residual,
        ),
        4 => subtract_forces(
            face,
            Quadrilateral::pressure_forces(&face_coordinates(coordinates, face), pressure),
            residual,
        ),
        _ => panic!(),
    });
    Ok(())
}

fn add_pressure_stiffness(
    faces: &[Vec<usize>],
    pressure: Quantity<Stress>,
    coordinates: &NodalCoordinates<3>,
    tangent: &mut NodalStiffnessesSolid<3>,
) -> Result<(), String> {
    check(faces, coordinates)?;
    faces.iter().for_each(|face| match face.len() {
        3 => subtract_stiffnesses(
            face,
            Triangle::pressure_stiffnesses(&face_coordinates(coordinates, face), pressure),
            tangent,
        ),
        4 => subtract_stiffnesses(
            face,
            Quadrilateral::pressure_stiffnesses(&face_coordinates(coordinates, face), pressure),
            tangent,
        ),
        _ => panic!(),
    });
    Ok(())
}

fn subtract_pressure_coupled(
    faces: &[Vec<usize>],
    pressure: Quantity<Stress>,
    states: &NodalStatesSolidThermal,
    residual: &mut NodalForcesSolidThermal,
) -> Result<(), String> {
    let (coordinates, _) = split_states(states);
    let mut forces = NodalForcesSolid::<3>::zero(coordinates.len());
    subtract_pressure(faces, pressure, &coordinates, &mut forces)?;
    residual
        .iter_mut()
        .zip(forces)
        .for_each(|(residual, force)| residual.0 += force);
    Ok(())
}

fn add_pressure_stiffness_coupled(
    faces: &[Vec<usize>],
    pressure: Quantity<Stress>,
    states: &NodalStatesSolidThermal,
    tangent: &mut NodalStiffnessesSolidThermal,
) -> Result<(), String> {
    let (coordinates, _) = split_states(states);
    let mut stiffnesses = NodalStiffnessesSolid::<3>::zero(coordinates.len());
    add_pressure_stiffness(faces, pressure, &coordinates, &mut stiffnesses)?;
    stiffnesses.iter().enumerate().for_each(|(a, row)| {
        row.entries().for_each(|(b, stiffness)| {
            tangent[a][b] += TensorTuple(
                TensorTuple(stiffness.clone(), Default::default()),
                TensorTuple(Default::default(), Default::default()),
            )
        })
    });
    Ok(())
}

impl Load<NodalCoordinates<3>, NodalForcesSolid<3>> for Pressure {
    fn subtract_from(
        &self,
        coordinates: &NodalCoordinates<3>,
        residual: &mut NodalForcesSolid<3>,
    ) -> Result<(), String> {
        subtract_pressure(&self.faces, self.pressure, coordinates, residual)
    }
}

impl LoadStiffness<NodalCoordinates<3>, NodalStiffnessesSolid<3>> for Pressure {
    fn add_stiffness_to(
        &self,
        coordinates: &NodalCoordinates<3>,
        tangent: &mut NodalStiffnessesSolid<3>,
    ) -> Result<(), String> {
        add_pressure_stiffness(&self.faces, self.pressure, coordinates, tangent)
    }
}

impl LoadAt<NodalCoordinates<3>, NodalForcesSolid<3>> for Pressure {
    fn subtract_at(
        &self,
        _: Quantity<Time>,
        coordinates: &NodalCoordinates<3>,
        residual: &mut NodalForcesSolid<3>,
    ) -> Result<(), String> {
        self.subtract_from(coordinates, residual)
    }
}

impl LoadStiffnessAt<NodalCoordinates<3>, NodalStiffnessesSolid<3>> for Pressure {
    fn add_stiffness_at(
        &self,
        _: Quantity<Time>,
        coordinates: &NodalCoordinates<3>,
        tangent: &mut NodalStiffnessesSolid<3>,
    ) -> Result<(), String> {
        self.add_stiffness_to(coordinates, tangent)
    }
}

impl Load<NodalStatesSolidThermal, NodalForcesSolidThermal> for Pressure {
    fn subtract_from(
        &self,
        states: &NodalStatesSolidThermal,
        residual: &mut NodalForcesSolidThermal,
    ) -> Result<(), String> {
        subtract_pressure_coupled(&self.faces, self.pressure, states, residual)
    }
}

impl LoadStiffness<NodalStatesSolidThermal, NodalStiffnessesSolidThermal> for Pressure {
    fn add_stiffness_to(
        &self,
        states: &NodalStatesSolidThermal,
        tangent: &mut NodalStiffnessesSolidThermal,
    ) -> Result<(), String> {
        add_pressure_stiffness_coupled(&self.faces, self.pressure, states, tangent)
    }
}

impl LoadAt<NodalStatesSolidThermal, NodalForcesSolidThermal> for Pressure {
    fn subtract_at(
        &self,
        _: Quantity<Time>,
        states: &NodalStatesSolidThermal,
        residual: &mut NodalForcesSolidThermal,
    ) -> Result<(), String> {
        self.subtract_from(states, residual)
    }
}

impl LoadStiffnessAt<NodalStatesSolidThermal, NodalStiffnessesSolidThermal> for Pressure {
    fn add_stiffness_at(
        &self,
        _: Quantity<Time>,
        states: &NodalStatesSolidThermal,
        tangent: &mut NodalStiffnessesSolidThermal,
    ) -> Result<(), String> {
        self.add_stiffness_to(states, tangent)
    }
}

impl<G> LoadAt<NodalCoordinates<3>, NodalForcesSolid<3>> for TimedPressure<G>
where
    G: Fn(Quantity<Time>) -> Quantity<Stress>,
{
    fn subtract_at(
        &self,
        time: Quantity<Time>,
        coordinates: &NodalCoordinates<3>,
        residual: &mut NodalForcesSolid<3>,
    ) -> Result<(), String> {
        subtract_pressure(&self.faces, (self.pressure)(time), coordinates, residual)
    }
}

impl<G> LoadStiffnessAt<NodalCoordinates<3>, NodalStiffnessesSolid<3>> for TimedPressure<G>
where
    G: Fn(Quantity<Time>) -> Quantity<Stress>,
{
    fn add_stiffness_at(
        &self,
        time: Quantity<Time>,
        coordinates: &NodalCoordinates<3>,
        tangent: &mut NodalStiffnessesSolid<3>,
    ) -> Result<(), String> {
        add_pressure_stiffness(&self.faces, (self.pressure)(time), coordinates, tangent)
    }
}

impl<G> LoadAt<NodalStatesSolidThermal, NodalForcesSolidThermal> for TimedPressure<G>
where
    G: Fn(Quantity<Time>) -> Quantity<Stress>,
{
    fn subtract_at(
        &self,
        time: Quantity<Time>,
        states: &NodalStatesSolidThermal,
        residual: &mut NodalForcesSolidThermal,
    ) -> Result<(), String> {
        subtract_pressure_coupled(&self.faces, (self.pressure)(time), states, residual)
    }
}

impl<G> LoadStiffnessAt<NodalStatesSolidThermal, NodalStiffnessesSolidThermal> for TimedPressure<G>
where
    G: Fn(Quantity<Time>) -> Quantity<Stress>,
{
    fn add_stiffness_at(
        &self,
        time: Quantity<Time>,
        states: &NodalStatesSolidThermal,
        tangent: &mut NodalStiffnessesSolidThermal,
    ) -> Result<(), String> {
        add_pressure_stiffness_coupled(&self.faces, (self.pressure)(time), states, tangent)
    }
}

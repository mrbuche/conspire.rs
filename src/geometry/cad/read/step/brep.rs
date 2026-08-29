use crate::{
    geometry::{
        Coordinate, Direction,
        cad::{
            brep::{
                Brep, Edge, Face, HalfEdge, Loop, Shell,
                curve::{Circle, Curve, Line},
                surface::{Cylinder, Plane, Sphere, Surface},
            },
            part_21::{Exchange, Parameter, Record},
        },
    },
    io::invalid,
};
use std::{array::from_fn, collections::HashMap, io::Result};

const D: usize = 3;

pub(super) fn read(exchange: &Exchange) -> Result<Brep> {
    let mut reader = Reader {
        exchange,
        vertices: Vec::new(),
        vertex_index: HashMap::new(),
        edges: Vec::new(),
        edge_index: HashMap::new(),
        faces: Vec::new(),
    };
    let solid = reader.find("MANIFOLD_SOLID_BREP")?;
    let shell = reader.shell(reference(&solid.parameters, 1)?)?;
    Ok(Brep {
        vertices: reader.vertices,
        edges: reader.edges,
        faces: reader.faces,
        shells: vec![shell],
    })
}

struct Reader<'a> {
    exchange: &'a Exchange,
    vertices: Vec<Coordinate<D>>,
    vertex_index: HashMap<u64, usize>,
    edges: Vec<Edge>,
    edge_index: HashMap<u64, usize>,
    faces: Vec<Face>,
}

impl<'a> Reader<'a> {
    fn find(&self, keyword: &str) -> Result<&'a Record> {
        self.exchange
            .data
            .values()
            .flat_map(|instance| &instance.records)
            .find(|record| record.keyword == keyword)
            .ok_or_else(|| invalid(format!("STEP: no {keyword} entity")))
    }

    fn record(&self, id: u64, keyword: &str) -> Result<&'a Record> {
        let instance = self
            .exchange
            .data
            .get(&id)
            .ok_or_else(|| invalid(format!("STEP: #{id} is not defined")))?;
        instance
            .records
            .iter()
            .find(|record| record.keyword == keyword)
            .ok_or_else(|| invalid(format!("STEP: #{id} is not a {keyword}")))
    }

    fn any(&self, id: u64, keywords: &[&str]) -> Result<&'a Record> {
        keywords
            .iter()
            .find_map(|keyword| self.record(id, keyword).ok())
            .ok_or_else(|| invalid(format!("STEP: #{id} is not one of {keywords:?}")))
    }

    fn point(&self, id: u64) -> Result<Coordinate<D>> {
        let record = self.record(id, "CARTESIAN_POINT")?;
        Ok(Coordinate::const_from(triple(&record.parameters, 1)?))
    }

    fn direction(&self, id: u64) -> Result<Direction<D>> {
        let record = self.record(id, "DIRECTION")?;
        let raw = triple(&record.parameters, 1)?;
        let norm = raw.iter().map(|entry| entry * entry).sum::<f64>().sqrt();
        if norm < f64::EPSILON {
            return Err(invalid(format!("STEP: #{id} is a zero direction")));
        }
        Ok(Direction::const_from(raw.map(|entry| entry / norm)))
    }

    fn axes(&self, id: u64) -> Result<(Coordinate<D>, Direction<D>, Direction<D>)> {
        let record = self.record(id, "AXIS2_PLACEMENT_3D")?;
        let origin = self.point(reference(&record.parameters, 1)?)?;
        let normal = match record.parameters.get(2) {
            Some(Parameter::Reference(axis)) => self.direction(*axis)?,
            _ => Direction::const_from([0.0, 0.0, 1.0]),
        };
        let reference_direction = match record.parameters.get(3) {
            Some(Parameter::Reference(axis)) => self.direction(*axis)?,
            _ => Direction::const_from([1.0, 0.0, 0.0]),
        };
        Ok((origin, normal, reference_direction))
    }

    fn surface(&self, id: u64) -> Result<Surface> {
        if let Ok(record) = self.record(id, "PLANE") {
            let (origin, normal, reference_direction) =
                self.axes(reference(&record.parameters, 1)?)?;
            return Ok(Surface::Plane(Plane {
                origin,
                normal,
                reference_direction,
            }));
        }
        if let Ok(record) = self.record(id, "CYLINDRICAL_SURFACE") {
            let (origin, axis, reference_direction) =
                self.axes(reference(&record.parameters, 1)?)?;
            let radius = scalar(parameter(&record.parameters, 2)?)?;
            return Ok(Surface::Cylinder(Cylinder {
                origin,
                axis,
                reference_direction,
                radius,
            }));
        }
        if let Ok(record) = self.record(id, "SPHERICAL_SURFACE") {
            let (origin, axis, reference_direction) =
                self.axes(reference(&record.parameters, 1)?)?;
            let radius = scalar(parameter(&record.parameters, 2)?)?;
            return Ok(Surface::Sphere(Sphere {
                origin,
                axis,
                reference_direction,
                radius,
            }));
        }
        Err(invalid(format!(
            "STEP: #{id} is not a supported surface (only PLANE, CYLINDRICAL_SURFACE, SPHERICAL_SURFACE)"
        )))
    }

    fn curve(&self, id: u64, same_sense: bool) -> Result<Curve> {
        if let Ok(record) = self.any(id, &["SURFACE_CURVE", "SEAM_CURVE", "INTERSECTION_CURVE"]) {
            return self.curve(reference(&record.parameters, 1)?, same_sense);
        }
        if let Ok(record) = self.record(id, "LINE") {
            let origin = self.point(reference(&record.parameters, 1)?)?;
            let vector = self.record(reference(&record.parameters, 2)?, "VECTOR")?;
            let mut direction = self.direction(reference(&vector.parameters, 1)?)?;
            if !same_sense {
                direction = Direction::const_from(from_fn(|axis| -direction[axis].value()));
            }
            return Ok(Curve::Line(Line { origin, direction }));
        }
        if let Ok(record) = self.record(id, "CIRCLE") {
            let (center, mut axis, reference_direction) =
                self.axes(reference(&record.parameters, 1)?)?;
            let radius = scalar(parameter(&record.parameters, 2)?)?;
            if !same_sense {
                axis = Direction::const_from(from_fn(|k| -axis[k].value()));
            }
            return Ok(Curve::Circle(Circle {
                center,
                axis,
                reference_direction,
                radius,
            }));
        }
        Err(invalid(format!(
            "STEP: #{id} is not a supported curve (only LINE, CIRCLE)"
        )))
    }

    fn vertex(&mut self, id: u64) -> Result<usize> {
        if let Some(&index) = self.vertex_index.get(&id) {
            return Ok(index);
        }
        let record = self.record(id, "VERTEX_POINT")?;
        let point = self.point(reference(&record.parameters, 1)?)?;
        let index = self.vertices.len();
        self.vertices.push(point);
        self.vertex_index.insert(id, index);
        Ok(index)
    }

    fn edge(&mut self, id: u64) -> Result<usize> {
        if let Some(&index) = self.edge_index.get(&id) {
            return Ok(index);
        }
        let record = self.record(id, "EDGE_CURVE")?;
        let start = self.vertex(reference(&record.parameters, 1)?)?;
        let end = self.vertex(reference(&record.parameters, 2)?)?;
        let geometry = reference(&record.parameters, 3)?;
        let same_sense = boolean(&record.parameters, 4)?;
        let curve = self.curve(geometry, same_sense)?;
        let index = self.edges.len();
        self.edges.push(Edge {
            vertices: [start, end],
            curve,
        });
        self.edge_index.insert(id, index);
        Ok(index)
    }

    fn half_edge(&mut self, id: u64) -> Result<HalfEdge> {
        let record = self.record(id, "ORIENTED_EDGE")?;
        let edge = self.edge(reference(&record.parameters, 3)?)?;
        Ok(HalfEdge {
            edge,
            forward: boolean(&record.parameters, 4)?,
        })
    }

    fn edge_loop(&mut self, id: u64, orientation: bool) -> Result<Loop> {
        let elements = list(&self.record(id, "EDGE_LOOP")?.parameters, 1)?
            .iter()
            .map(as_reference)
            .collect::<Result<Vec<_>>>()?;
        let mut half_edges = elements
            .into_iter()
            .map(|element| self.half_edge(element))
            .collect::<Result<Vec<_>>>()?;
        if !orientation {
            half_edges.reverse();
            half_edges
                .iter_mut()
                .for_each(|half_edge| half_edge.forward = !half_edge.forward);
        }
        Ok(Loop { half_edges })
    }

    fn bound(&mut self, id: u64) -> Result<(Loop, bool)> {
        let record = self.any(id, &["FACE_OUTER_BOUND", "FACE_BOUND"])?;
        let outer = record.keyword == "FACE_OUTER_BOUND";
        let edge_loop = self.edge_loop(
            reference(&record.parameters, 1)?,
            boolean(&record.parameters, 2)?,
        )?;
        Ok((edge_loop, outer))
    }

    fn face(&mut self, id: u64) -> Result<usize> {
        let record = self.record(id, "ADVANCED_FACE")?;
        let bound_ids = list(&record.parameters, 1)?
            .iter()
            .map(as_reference)
            .collect::<Result<Vec<_>>>()?;
        let surface = self.surface(reference(&record.parameters, 2)?)?;
        let forward = boolean(&record.parameters, 3)?;
        let mut outer = None;
        let mut inner = Vec::new();
        for bound_id in bound_ids {
            let (edge_loop, is_outer) = self.bound(bound_id)?;
            if is_outer && outer.is_none() {
                outer = Some(edge_loop);
            } else {
                inner.push(edge_loop);
            }
        }
        let mut bounds =
            vec![outer.ok_or_else(|| invalid(format!("STEP: #{id} has no outer bound")))?];
        bounds.append(&mut inner);
        let index = self.faces.len();
        self.faces.push(Face {
            surface,
            bounds,
            forward,
        });
        Ok(index)
    }

    fn shell(&mut self, id: u64) -> Result<Shell> {
        let record = self.any(id, &["CLOSED_SHELL", "OPEN_SHELL"])?;
        let closed = record.keyword == "CLOSED_SHELL";
        let face_ids = list(&record.parameters, 1)?
            .iter()
            .map(as_reference)
            .collect::<Result<Vec<_>>>()?;
        let faces = face_ids
            .into_iter()
            .map(|face_id| self.face(face_id))
            .collect::<Result<Vec<_>>>()?;
        Ok(Shell { faces, closed })
    }
}

fn parameter(parameters: &[Parameter], index: usize) -> Result<&Parameter> {
    parameters
        .get(index)
        .ok_or_else(|| invalid(format!("STEP: missing parameter {index}")))
}

fn as_reference(parameter: &Parameter) -> Result<u64> {
    match parameter {
        Parameter::Reference(id) => Ok(*id),
        other => Err(invalid(format!(
            "STEP: expected a reference, found {other:?}"
        ))),
    }
}

fn reference(parameters: &[Parameter], index: usize) -> Result<u64> {
    as_reference(parameter(parameters, index)?)
}

fn boolean(parameters: &[Parameter], index: usize) -> Result<bool> {
    match parameter(parameters, index)? {
        Parameter::Enumeration(value) if value == "T" => Ok(true),
        Parameter::Enumeration(value) if value == "F" => Ok(false),
        other => Err(invalid(format!(
            "STEP: expected .T. or .F., found {other:?}"
        ))),
    }
}

fn scalar(parameter: &Parameter) -> Result<f64> {
    match parameter {
        Parameter::Real(value) => Ok(*value),
        Parameter::Integer(value) => Ok(*value as f64),
        other => Err(invalid(format!("STEP: expected a number, found {other:?}"))),
    }
}

fn list(parameters: &[Parameter], index: usize) -> Result<&[Parameter]> {
    match parameter(parameters, index)? {
        Parameter::List(items) => Ok(items),
        other => Err(invalid(format!("STEP: expected a list, found {other:?}"))),
    }
}

fn triple(parameters: &[Parameter], index: usize) -> Result<[f64; D]> {
    let items = list(parameters, index)?;
    if items.len() != D {
        return Err(invalid(format!("STEP: expected {D} coordinates")));
    }
    let mut values = [0.0; D];
    for (value, item) in values.iter_mut().zip(items) {
        *value = scalar(item)?;
    }
    Ok(values)
}

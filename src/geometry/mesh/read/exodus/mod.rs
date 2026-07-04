#[cfg(test)]
mod test;

use crate::{
    geometry::{
        Coordinates,
        mesh::{Connectivities, Connectivity, Mesh, NodeSets},
    },
    io::{GetVariable, NetCDF},
    math::Set,
};
use std::{array::from_fn, ffi::NulError, path::Path};

pub trait ReadExodus<P>
where
    P: AsRef<Path>,
    Self: Sized,
{
    fn read_exodus(input: P) -> Result<Self, NulError>;
}

impl<const D: usize, P> ReadExodus<P> for Mesh<D>
where
    P: AsRef<Path>,
{
    fn read_exodus(input: P) -> Result<Self, NulError> {
        let path = input.as_ref().to_str().unwrap();
        let netcdf = NetCDF::open(path)?;
        let num_dim = netcdf.dimension_length("num_dim")?;
        assert_eq!(
            num_dim, D,
            "exodus num_dim={num_dim} but Mesh was asked for D={D}"
        );
        let num_el_blk = netcdf.dimension_length("num_el_blk")?;
        let num_elem = netcdf.dimension_length("num_elem")?;
        let num_nodes = netcdf.dimension_length("num_nodes")?;
        let mut connectivities = (1..=num_el_blk)
            .map(|block| read_block::<D>(&netcdf, block))
            .collect::<Result<Vec<_>, _>>()?;
        let blocks = netcdf
            .get_variable::<i32>("eb_prop1", num_el_blk)?
            .into_iter()
            .map(|id| id as usize)
            .collect();
        if let Some(element_numbers) = netcdf.try_get_variable::<i32>("elem_num_map", num_elem)? {
            let mut offset = 0;
            connectivities.iter_mut().for_each(|connectivity| {
                let count = connectivity.number_of_elements();
                connectivity.number_elements(
                    element_numbers[offset..offset + count]
                        .iter()
                        .map(|&id| id as usize)
                        .collect(),
                );
                offset += count;
            });
        }
        let coordx = netcdf.get_variable::<f64>("coordx", num_nodes)?;
        let coordy = netcdf.get_variable::<f64>("coordy", num_nodes)?;
        let coordz = match D {
            2 => Vec::new(),
            3 => netcdf.get_variable::<f64>("coordz", num_nodes)?,
            _ => unimplemented!(),
        };
        let coordinates: Coordinates<D> = (0..num_nodes)
            .map(|i| {
                from_fn(|ax| match ax {
                    0 => coordx[i],
                    1 => coordy[i],
                    2 => coordz[i],
                    _ => unreachable!(),
                })
                .into()
            })
            .collect();
        let coordinates = match netcdf.try_get_variable::<i32>("node_num_map", num_nodes)? {
            Some(node_numbers) => Set::from((
                coordinates,
                node_numbers.into_iter().map(|id| id as usize).collect(),
            )),
            None => Set::from(coordinates),
        };
        let mut mesh =
            Mesh::<D>::from((Connectivities::from((connectivities, blocks)), coordinates));
        if let Some(num_node_sets) = netcdf.try_dimension_length("num_node_sets")? {
            let node_set_numbers = netcdf
                .get_variable::<i32>("ns_prop1", num_node_sets)?
                .into_iter()
                .map(|id| id as usize)
                .collect::<Vec<_>>();
            let node_sets = (1..=num_node_sets)
                .map(|set| {
                    let num_nod_ns = netcdf.dimension_length(&format!("num_nod_ns{}", set))?;
                    netcdf
                        .get_variable::<i32>(&format!("node_ns{}", set), num_nod_ns)
                        .map(|nodes| nodes.into_iter().map(|id| (id - 1) as usize).collect())
                })
                .collect::<Result<Vec<Vec<usize>>, _>>()?;
            mesh.set_node_sets(NodeSets::from((node_sets, node_set_numbers)));
        }
        Ok(mesh)
    }
}

fn read_block<const D: usize>(netcdf: &NetCDF, block: usize) -> Result<Connectivity, NulError> {
    let num_el_in_blk = netcdf.dimension_length(&format!("num_el_in_blk{}", block))?;
    if let Some(num_nod_per_el) =
        netcdf.try_dimension_length(&format!("num_nod_per_el{}", block))?
    {
        read_primitive_block::<D>(netcdf, block, num_el_in_blk, num_nod_per_el)
    } else {
        read_polytopal_block(netcdf, block, num_el_in_blk)
    }
}

fn read_primitive_block<const D: usize>(
    netcdf: &NetCDF,
    block: usize,
    num_el_in_blk: usize,
    num_nod_per_el: usize,
) -> Result<Connectivity, NulError> {
    let elem_type = netcdf
        .get_variable_attribute_text(&format!("connect{}", block), "elem_type")?
        .to_lowercase();
    let flat =
        netcdf.get_variable::<i32>(&format!("connect{}", block), num_el_in_blk * num_nod_per_el)?;
    match (D, num_nod_per_el, elem_type.as_str()) {
        (3, 8, "hex8") => Ok(Connectivity::Hexahedral(unflatten::<8>(&flat).into())),
        (3, 6, "wedge6") => Ok(Connectivity::Wedge(unflatten::<6>(&flat).into())),
        (3, 5, "pyramid5") => Ok(Connectivity::Pyramidal(unflatten::<5>(&flat).into())),
        (3, 4, "tet4") => Ok(Connectivity::Tetrahedral(unflatten::<4>(&flat).into())),
        (_, 4, "quad4") => Ok(Connectivity::Quadrilateral(unflatten::<4>(&flat).into())),
        (_, 3, "tri3") => Ok(Connectivity::Triangular(unflatten::<3>(&flat).into())),
        _ => panic!("unknown element type: D={D}, N={num_nod_per_el}, elem_type={elem_type}"),
    }
}

fn read_polytopal_block(
    netcdf: &NetCDF,
    block: usize,
    num_el_in_blk: usize,
) -> Result<Connectivity, NulError> {
    let num_fac_per_el = netcdf.dimension_length(&format!("num_fac_per_el{}", block))?;
    let ebepecnt = netcdf.get_variable::<i32>(&format!("ebepecnt{}", block), num_el_in_blk)?;
    let facconn = netcdf.get_variable::<i32>(&format!("facconn{}", block), num_fac_per_el)?;
    let elements_faces = unflatten_var(&facconn, &ebepecnt);
    let num_fa_in_blk = netcdf.dimension_length(&format!("num_fa_in_blk{}", block))?;
    let num_nod_per_fa = netcdf.dimension_length(&format!("num_nod_per_fa{}", block))?;
    let fbepecnt = netcdf.get_variable::<i32>(&format!("fbepecnt{}", block), num_fa_in_blk)?;
    let fbconn = netcdf.get_variable::<i32>(&format!("fbconn{}", block), num_nod_per_fa)?;
    let faces_nodes = unflatten_var(&fbconn, &fbepecnt);
    let elem_type = netcdf
        .get_variable_attribute_text(&format!("facconn{}", block), "elem_type")?
        .to_lowercase();
    match elem_type.as_str() {
        "nfaced" => Ok(Connectivity::Polyhedral(
            (elements_faces, faces_nodes).into(),
        )),
        "nsided" => Ok(Connectivity::Polygonal(
            (elements_faces, faces_nodes).into(),
        )),
        _ => panic!("unknown polytopal element type: {elem_type}"),
    }
}

fn unflatten<const N: usize>(flat: &[i32]) -> Vec<[usize; N]> {
    assert_eq!(flat.len() % N, 0);
    flat.chunks_exact(N)
        .map(|chunk| from_fn(|i| (chunk[i] - 1) as usize))
        .collect()
}

fn unflatten_var(flat: &[i32], counts: &[i32]) -> Vec<Vec<usize>> {
    let mut out = Vec::with_capacity(counts.len());
    let mut idx = 0;
    for &count in counts {
        let count = count as usize;
        out.push(
            flat[idx..idx + count]
                .iter()
                .map(|&n| (n - 1) as usize)
                .collect(),
        );
        idx += count;
    }
    debug_assert_eq!(idx, flat.len());
    out
}

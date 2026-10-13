#[cfg(test)]
mod test;

use crate::{
    geometry::mesh::{Mesh, connectivity::base::FlatConnectivity},
    io::{DefineVariable, NetCDF, PutVariable},
};
use std::{ffi::NulError, path::Path};

/// Which on-disk Exodus flavour to write.
pub enum ExodusFormat<P>
where
    P: AsRef<Path>,
{
    /// Classic netCDF (CDF-5).
    Classic(P),
    /// netCDF-4 (HDF5): chunked, shuffle + deflate compressed. `threads` bounds
    /// the pool used to compress chunks (`0` or `1` compresses serially).
    Netcdf4 { path: P, threads: usize },
}

impl<P> AsRef<Path> for ExodusFormat<P>
where
    P: AsRef<Path>,
{
    fn as_ref(&self) -> &Path {
        match self {
            ExodusFormat::Classic(path) => path.as_ref(),
            ExodusFormat::Netcdf4 { path, .. } => path.as_ref(),
        }
    }
}

pub(crate) trait WriteExodus<P>
where
    P: AsRef<Path>,
{
    fn write_exodus(&self, output: P) -> Result<(), NulError>;
    fn write_exodus_compressed(&self, output: P, threads: usize) -> Result<(), NulError>;
}

impl<const D: usize, P> WriteExodus<P> for Mesh<D>
where
    P: AsRef<Path>,
{
    fn write_exodus(&self, output: P) -> Result<(), NulError> {
        self.write_exodus_format(output, None)
    }
    fn write_exodus_compressed(&self, output: P, threads: usize) -> Result<(), NulError> {
        self.write_exodus_format(output, Some(threads))
    }
}

impl<const D: usize> Mesh<D> {
    fn write_exodus_format<P: AsRef<Path>>(
        &self,
        output: P,
        netcdf4: Option<usize>,
    ) -> Result<(), NulError> {
        let path = output.as_ref().to_str().unwrap();
        let mut netcdf = match netcdf4 {
            Some(threads) => NetCDF::create_netcdf4(path, threads)?,
            None => NetCDF::create(path)?,
        };
        netcdf.global();
        let element_labels: Option<Vec<i32>> = self
            .connectivities()
            .iter()
            .map(|connectivity| {
                connectivity
                    .element_labels()
                    .map(|labels| labels.iter().map(|&label| label as i32).collect::<Vec<_>>())
            })
            .collect::<Option<Vec<Vec<i32>>>>()
            .map(|per_block| per_block.into_iter().flatten().collect());
        let node_labels: Option<Vec<i32>> = self
            .node_labels()
            .map(|labels| labels.iter().map(|&label| label as i32).collect());
        netcdf.define_dimension("num_dim", D)?;
        netcdf.define_dimension("num_elem", self.number_of_elements())?;
        netcdf.define_dimension("num_el_blk", self.number_of_element_blocks())?;
        netcdf.define_variable::<i32>("eb_prop1", 1, &["num_el_blk"])?;
        netcdf.put_variable_attribute_text("eb_prop1", "name", "ID")?;
        if let Some(num_fa_blk) = self.number_of_face_blocks() {
            netcdf.define_dimension("num_fa_blk", num_fa_blk)?;
            netcdf.define_variable::<i32>("fa_prop1", 1, &["num_fa_blk"])?;
            netcdf.put_variable_attribute_text("fa_prop1", "name", "ID")?;
        }
        if let Some(num_face) = self.number_of_faces() {
            netcdf.define_dimension("num_face", num_face)?;
        }
        if element_labels.is_some() {
            netcdf.define_variable::<i32>("elem_num_map", 1, &["num_elem"])?;
        }
        let mut face_block = 0;
        self.iter()
            .enumerate()
            .try_for_each(|(block, connectivity)| {
                let block = block + 1;
                netcdf.define_dimension(
                    &format!("num_el_in_blk{}", block),
                    connectivity.number_of_elements(),
                )?;
                if let Some(num) = connectivity.number_of_nodes_per_element() {
                    netcdf.define_dimension(&format!("num_nod_per_el{}", block), num)?;
                    netcdf.define_variable::<i32>(
                        &format!("connect{}", block),
                        2,
                        &[
                            &format!("num_el_in_blk{}", block),
                            &format!("num_nod_per_el{}", block),
                        ],
                    )?;
                    netcdf.put_variable_attribute_text(
                        &format!("connect{}", block),
                        "elem_type",
                        connectivity.exodus_element_type(),
                    )
                } else {
                    face_block += 1;
                    if let Some(ebepecnt) = connectivity.number_of_faces_per_element::<i32>() {
                        netcdf.define_dimension(
                            &format!("num_fac_per_el{}", block),
                            ebepecnt.into_iter().map(|c| c as usize).sum(),
                        )?;
                        netcdf.define_variable::<i32>(
                            &format!("facconn{}", block),
                            1,
                            &[&format!("num_fac_per_el{}", block)],
                        )?;
                        netcdf.put_variable_attribute_text(
                            &format!("facconn{}", block),
                            "elem_type",
                            connectivity.exodus_element_type(),
                        )?;
                        netcdf.define_variable::<i32>(
                            &format!("ebepecnt{}", block),
                            1,
                            &[&format!("num_el_in_blk{}", block)],
                        )?;
                        netcdf.put_variable_attribute_text(
                            &format!("ebepecnt{}", block),
                            "entity_type1",
                            "FACE",
                        )?;
                        netcdf.put_variable_attribute_text(
                            &format!("ebepecnt{}", block),
                            "entity_type2",
                            "ELEM",
                        )?;
                    } else {
                        panic!()
                    }
                    if let Some(num_fa_in_blk) = connectivity.number_of_faces() {
                        netcdf.define_dimension(
                            &format!("num_fa_in_blk{}", face_block),
                            num_fa_in_blk,
                        )?;
                    } else {
                        panic!()
                    }
                    if let Some(fbepecnt) = connectivity.number_of_nodes_per_face::<i32>() {
                        netcdf.define_dimension(
                            &format!("num_nod_per_fa{}", face_block),
                            fbepecnt.into_iter().map(|c| c as usize).sum(),
                        )?;
                        netcdf.define_variable::<i32>(
                            &format!("fbconn{}", face_block),
                            1,
                            &[&format!("num_nod_per_fa{}", face_block)],
                        )?;
                        netcdf.put_variable_attribute_text(
                            &format!("fbconn{}", face_block),
                            "elem_type",
                            "nsided",
                        )?;
                        netcdf.define_variable::<i32>(
                            &format!("fbepecnt{}", face_block),
                            1,
                            &[&format!("num_fa_in_blk{}", face_block)],
                        )?;
                        netcdf.put_variable_attribute_text(
                            &format!("fbepecnt{}", face_block),
                            "entity_type1",
                            "NODE",
                        )?;
                        netcdf.put_variable_attribute_text(
                            &format!("fbepecnt{}", face_block),
                            "entity_type2",
                            "FACE",
                        )
                    } else {
                        panic!()
                    }
                }
            })?;
        netcdf.define_dimension("num_nodes", self.number_of_nodes())?;
        if node_labels.is_some() {
            netcdf.define_variable::<i32>("node_num_map", 1, &["num_nodes"])?;
        }
        if !self.node_sets().is_empty() {
            netcdf.define_dimension("num_node_sets", self.node_sets().len())?;
            netcdf.define_variable::<i32>("ns_prop1", 1, &["num_node_sets"])?;
            netcdf.put_variable_attribute_text("ns_prop1", "name", "ID")?;
            self.node_sets()
                .iter()
                .enumerate()
                .try_for_each(|(set, node_set)| {
                    let set = set + 1;
                    netcdf
                        .define_dimension(&format!("num_nod_ns{}", set), node_set.nodes().len())?;
                    netcdf.define_variable::<i32>(
                        &format!("node_ns{}", set),
                        1,
                        &[&format!("num_nod_ns{}", set)],
                    )
                })?;
        }
        if !self.side_sets().is_empty() {
            netcdf.define_dimension("num_side_sets", self.side_sets().len())?;
            netcdf.define_variable::<i32>("ss_prop1", 1, &["num_side_sets"])?;
            netcdf.put_variable_attribute_text("ss_prop1", "name", "ID")?;
            self.side_sets()
                .iter()
                .enumerate()
                .try_for_each(|(set, side_set)| {
                    let set = set + 1;
                    netcdf
                        .define_dimension(&format!("num_side_ss{}", set), side_set.sides().len())?;
                    netcdf.define_variable::<i32>(
                        &format!("elem_ss{}", set),
                        1,
                        &[&format!("num_side_ss{}", set)],
                    )?;
                    netcdf.define_variable::<i32>(
                        &format!("side_ss{}", set),
                        1,
                        &[&format!("num_side_ss{}", set)],
                    )
                })?;
        }
        netcdf.define_dimension("time_step", 0)?;
        netcdf.define_variable::<f64>("coordx", 1, &["num_nodes"])?;
        netcdf.define_variable::<f64>("coordy", 1, &["num_nodes"])?;
        match D {
            2 => {}
            3 => {
                netcdf.define_variable::<f64>("coordz", 1, &["num_nodes"])?;
            }
            _ => unimplemented!(),
        }
        netcdf.end_definition();
        let block_labels: Vec<i32> = match self.connectivities.labels() {
            Some(labels) => labels.iter().map(|&label| label as i32).collect(),
            None => (1..=self.number_of_element_blocks() as i32).collect(),
        };
        netcdf.put_variable("eb_prop1", &block_labels)?;
        if self.number_of_face_blocks().is_some() {
            let face_block_labels: Vec<i32> = self
                .iter()
                .zip(&block_labels)
                .filter(|(connectivity, _)| connectivity.number_of_faces().is_some())
                .map(|(_, &label)| label)
                .collect();
            netcdf.put_variable("fa_prop1", &face_block_labels)?;
        }
        if let Some(element_labels) = &element_labels {
            netcdf.put_variable("elem_num_map", element_labels)?;
        }
        if let Some(node_labels) = &node_labels {
            netcdf.put_variable("node_num_map", node_labels)?;
        }
        if !self.node_sets().is_empty() {
            let node_set_labels: Vec<i32> = self
                .node_sets()
                .resolved_labels()
                .into_iter()
                .map(|label| label as i32)
                .collect();
            netcdf.put_variable("ns_prop1", &node_set_labels)?;
            self.node_sets()
                .iter()
                .enumerate()
                .try_for_each(|(set, node_set)| {
                    let set = set + 1;
                    let node_ids: Vec<i32> = node_set
                        .nodes()
                        .iter()
                        .map(|&node| node as i32 + 1)
                        .collect();
                    netcdf.put_variable(&format!("node_ns{}", set), &node_ids)
                })?;
        }
        if !self.side_sets().is_empty() {
            let side_set_labels: Vec<i32> = self
                .side_sets()
                .resolved_labels()
                .into_iter()
                .map(|label| label as i32)
                .collect();
            netcdf.put_variable("ss_prop1", &side_set_labels)?;
            self.side_sets()
                .iter()
                .enumerate()
                .try_for_each(|(set, side_set)| {
                    let set = set + 1;
                    let elem_ids: Vec<i32> = side_set
                        .sides()
                        .iter()
                        .map(|&(element, _)| match &element_labels {
                            Some(labels) => labels[element],
                            None => element as i32 + 1,
                        })
                        .collect();
                    let side_ids: Vec<i32> = side_set
                        .sides()
                        .iter()
                        .map(|&(_, side)| side as i32 + 1)
                        .collect();
                    netcdf.put_variable(&format!("elem_ss{}", set), &elem_ids)?;
                    netcdf.put_variable(&format!("side_ss{}", set), &side_ids)
                })?;
        }
        let mut face_block = 0;
        let mut faces_before = 0;
        self.iter()
            .enumerate()
            .try_for_each(|(block, connectivity)| {
                let block = block + 1;
                match connectivity.flat_connectivity::<i32>() {
                    FlatConnectivity::Primitive(flat) => {
                        netcdf.put_variable(&format!("connect{}", block), &flat)
                    }
                    FlatConnectivity::Polytopal(elements_faces, faces_nodes) => {
                        face_block += 1;
                        let elements_faces: Vec<i32> = elements_faces
                            .into_iter()
                            .map(|face| face + faces_before)
                            .collect();
                        faces_before += connectivity.number_of_faces().unwrap() as i32;
                        netcdf.put_variable(&format!("facconn{}", block), &elements_faces)?;
                        netcdf.put_variable(&format!("fbconn{}", face_block), &faces_nodes)?;
                        if let Some(ebepecnt) = connectivity.number_of_faces_per_element::<i32>() {
                            netcdf.put_variable(&format!("ebepecnt{}", block), &ebepecnt)?;
                        } else {
                            panic!()
                        }
                        if let Some(fbepecnt) = connectivity.number_of_nodes_per_face::<i32>() {
                            netcdf.put_variable(&format!("fbepecnt{}", face_block), &fbepecnt)
                        } else {
                            panic!()
                        }
                    }
                }
            })?;
        let coordinates: [_; D] = self.coordinates().into();
        netcdf.put_variable("coordx", &coordinates[0])?;
        netcdf.put_variable("coordy", &coordinates[1])?;
        match D {
            2 => {}
            3 => {
                netcdf.put_variable("coordz", &coordinates[2])?;
            }
            _ => unimplemented!(),
        }
        Ok(())
    }
}

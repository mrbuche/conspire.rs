use crate::io::netcdf::{
    DefineVariable, GetVariable, NcType, NetCDF, PutVariable, State, VarBuild,
    format::{self, Storage, decode_be, decode_le, hdf5},
    nc_lock, reject_nul,
};
use std::{
    ffi::NulError,
    io::{Seek, SeekFrom, Write},
};

impl DefineVariable for NetCDF {
    fn define_variable<T: NcType>(
        &mut self,
        name: &str,
        ndims: usize,
        dim_names: &[&str],
    ) -> Result<(), NulError> {
        reject_nul(name)?;
        for dim_name in dim_names {
            reject_nul(dim_name)?;
        }
        assert_eq!(ndims, dim_names.len(), "ndims must equal dim_names.len()");
        let _guard = nc_lock();
        self.writer_defining().variables.push(VarBuild {
            name: name.to_string(),
            xtype: T::XTYPE,
            dim_names: dim_names.iter().map(|dim| dim.to_string()).collect(),
            attributes: Vec::new(),
        });
        Ok(())
    }
}

impl PutVariable for NetCDF {
    fn put_variable<T: NcType>(&mut self, name: &str, data: &[T]) -> Result<(), NulError> {
        reject_nul(name)?;
        let _guard = nc_lock();
        let writer = match &mut self.state {
            State::Write(writer) => writer,
            State::Read(_) => panic!("put_variable on a NetCDF opened for reading"),
        };
        let netcdf4 = writer.netcdf4;
        let dims = std::mem::take(&mut writer.dims);
        let output = writer
            .output
            .as_mut()
            .unwrap_or_else(|| panic!("put_variable before end_definition"));
        let index = output
            .variables
            .iter()
            .position(|spec| spec.name == name)
            .unwrap_or_else(|| panic!("no variable named {name}"));
        let spec = &output.variables[index];
        assert_eq!(
            spec.xtype,
            T::XTYPE,
            "type mismatch writing variable {name}"
        );
        if netcdf4 {
            assert_eq!(
                data.len() as u64,
                spec.elements(&dims),
                "wrong element count for variable {name}"
            );
            let mut buffer = Vec::with_capacity(data.len() * T::SIZE);
            format::encode_le(data, &mut buffer);
            output.data[index] = buffer;
        } else {
            let (begin, vsize) = (spec.begin, spec.vsize as usize);
            assert_eq!(
                data.len(),
                vsize / T::SIZE,
                "wrong element count for variable {name}"
            );
            let mut buffer = Vec::with_capacity(vsize);
            format::encode_be(data, &mut buffer);
            buffer.resize(vsize, 0);
            output
                .file
                .seek(SeekFrom::Start(begin))
                .expect("seek failed");
            output
                .file
                .write_all(&buffer)
                .expect("variable write failed");
        }
        writer.dims = dims;
        Ok(())
    }
}

impl NetCDF {
    fn read_variable<T: NcType>(&self, name: &str, len: usize) -> Option<Vec<T>> {
        let reader = match &self.state {
            State::Read(reader) => reader,
            State::Write(_) => panic!("get_variable on a NetCDF opened for writing"),
        };
        let spec = reader.parsed.vars.iter().find(|spec| spec.name == name)?;
        assert_eq!(
            spec.xtype,
            T::XTYPE,
            "type mismatch reading variable {name}"
        );
        match &spec.storage {
            Storage::Classic => {
                let start = spec.begin as usize;
                let end = start + len * T::SIZE;
                assert!(
                    end <= reader.bytes.len(),
                    "variable {name} data runs past end of file"
                );
                Some(decode_be(&reader.bytes[start..end]))
            }
            Storage::Hdf5 {
                little_endian,
                layout,
                filters,
            } => {
                let raw = hdf5::read_data(&reader.bytes, layout, filters, len * T::SIZE);
                Some(if *little_endian {
                    decode_le(&raw)
                } else {
                    decode_be(&raw)
                })
            }
        }
    }
}

impl GetVariable for NetCDF {
    fn get_variable<T: NcType>(&self, name: &str, len: usize) -> Result<Vec<T>, NulError> {
        reject_nul(name)?;
        let _guard = nc_lock();
        Ok(self
            .read_variable(name, len)
            .unwrap_or_else(|| panic!("no variable named {name}")))
    }
    fn try_get_variable<T: NcType>(
        &self,
        name: &str,
        len: usize,
    ) -> Result<Option<Vec<T>>, NulError> {
        reject_nul(name)?;
        let _guard = nc_lock();
        Ok(self.read_variable(name, len))
    }
}

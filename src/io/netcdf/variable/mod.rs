use crate::io::netcdf::{
    DefineVariable, GetVariable, NcType, NetCDF, PutVariable, State, VarBuild,
    format::{self},
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
        let output = match &mut self.state {
            State::Write(writer) => writer
                .output
                .as_mut()
                .unwrap_or_else(|| panic!("put_variable before end_definition")),
            State::Read(_) => panic!("put_variable on a NetCDF opened for reading"),
        };
        let (begin, vsize, xtype) = output
            .variables
            .iter()
            .find(|spec| spec.name == name)
            .map(|spec| (spec.begin, spec.vsize as usize, spec.xtype))
            .unwrap_or_else(|| panic!("no variable named {name}"));
        assert_eq!(xtype, T::XTYPE, "type mismatch writing variable {name}");
        let expected = vsize / T::SIZE;
        assert_eq!(
            data.len(),
            expected,
            "variable {name} expects {expected} elements, got {}",
            data.len()
        );
        let mut buffer = Vec::with_capacity(vsize);
        format::encode_be(data, &mut buffer);
        buffer.resize(vsize, 0); // pad to a multiple of four
        output
            .file
            .seek(SeekFrom::Start(begin))
            .unwrap_or_else(|error| panic!("seek failed for {name}: {error}"));
        output
            .file
            .write_all(&buffer)
            .unwrap_or_else(|error| panic!("write failed for {name}: {error}"));
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
        let start = spec.begin as usize;
        let end = start + len * T::SIZE;
        assert!(
            end <= reader.bytes.len(),
            "variable {name} data runs past end of file"
        );
        Some(format::decode_be(&reader.bytes[start..end]))
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

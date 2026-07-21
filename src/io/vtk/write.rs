use crate::io::deflate::zlib_encode;
use std::path::Path;

const COMPRESSION_BLOCK_SIZE: usize = 32768;

pub enum Compression<P>
where
    P: AsRef<Path>,
{
    On(P),
    Off(P),
}

impl<P> AsRef<Path> for Compression<P>
where
    P: AsRef<Path>,
{
    fn as_ref(&self) -> &Path {
        match self {
            Compression::On(path) => path.as_ref(),
            Compression::Off(path) => path.as_ref(),
        }
    }
}

pub fn data_array(data: &[u8]) -> String {
    let mut buffer = Vec::with_capacity(8 + data.len());
    buffer.extend_from_slice(&(data.len() as u64).to_le_bytes());
    buffer.extend_from_slice(data);
    base64(&buffer)
}

pub fn data_array_compressed(data: &[u8]) -> String {
    let compressed_blocks: Vec<Vec<u8>> = data
        .chunks(COMPRESSION_BLOCK_SIZE)
        .map(zlib_encode)
        .collect();
    let num_blocks = compressed_blocks.len() as u64;
    let last_block_size =
        data.len() as u64 - num_blocks.saturating_sub(1) * COMPRESSION_BLOCK_SIZE as u64;
    let mut buffer = Vec::with_capacity(24 + compressed_blocks.iter().map(Vec::len).sum::<usize>());
    buffer.extend_from_slice(&num_blocks.to_le_bytes());
    buffer.extend_from_slice(&(COMPRESSION_BLOCK_SIZE as u64).to_le_bytes());
    buffer.extend_from_slice(&last_block_size.to_le_bytes());
    for block in &compressed_blocks {
        buffer.extend_from_slice(&(block.len() as u64).to_le_bytes());
    }
    for block in &compressed_blocks {
        buffer.extend_from_slice(block);
    }
    base64(&buffer)
}

pub fn base64(bytes: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let triple = ((chunk[0] as u32) << 16)
            | ((*chunk.get(1).unwrap_or(&0) as u32) << 8)
            | (*chunk.get(2).unwrap_or(&0) as u32);
        out.push(ALPHABET[(triple >> 18 & 63) as usize] as char);
        out.push(ALPHABET[(triple >> 12 & 63) as usize] as char);
        out.push(if chunk.len() > 1 {
            ALPHABET[(triple >> 6 & 63) as usize] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            ALPHABET[(triple & 63) as usize] as char
        } else {
            '='
        });
    }
    out
}

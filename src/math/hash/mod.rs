#[cfg(test)]
mod test;

use std::{
    collections::{HashMap, HashSet},
    hash::{BuildHasherDefault, Hasher},
};

const SEED: u64 = 0x51_7c_c1_b7_27_22_0a_95;
const ROTATE: u32 = 5;

/// A [`HashMap`] using [`FxHasher`] instead of the default SipHash. Much faster for the small
/// integer keys (node indices, `(usize, usize)` edges) the mesh code hashes in tight loops.
pub type FxHashMap<K, V> = HashMap<K, V, BuildHasherDefault<FxHasher>>;

/// A [`HashSet`] using [`FxHasher`]. See [`FxHashMap`].
pub type FxHashSet<T> = HashSet<T, BuildHasherDefault<FxHasher>>;

/// A non-cryptographic hasher (the "FxHash" algorithm): one rotate-xor-multiply per word.
/// Not DoS-resistant, so only use it where keys are trusted (internal mesh indices).
#[derive(Default)]
pub struct FxHasher {
    hash: u64,
}

impl FxHasher {
    #[inline]
    fn add(&mut self, word: u64) {
        self.hash = (self.hash.rotate_left(ROTATE) ^ word).wrapping_mul(SEED);
    }
}

impl Hasher for FxHasher {
    #[inline]
    fn write(&mut self, mut bytes: &[u8]) {
        while let Some((word, rest)) = bytes.split_first_chunk::<8>() {
            self.add(u64::from_le_bytes(*word));
            bytes = rest;
        }
        if let Some((word, rest)) = bytes.split_first_chunk::<4>() {
            self.add(u32::from_le_bytes(*word) as u64);
            bytes = rest;
        }
        if let Some((word, rest)) = bytes.split_first_chunk::<2>() {
            self.add(u16::from_le_bytes(*word) as u64);
            bytes = rest;
        }
        if let Some(&byte) = bytes.first() {
            self.add(byte as u64);
        }
    }
    #[inline]
    fn write_u8(&mut self, value: u8) {
        self.add(value as u64);
    }
    #[inline]
    fn write_u16(&mut self, value: u16) {
        self.add(value as u64);
    }
    #[inline]
    fn write_u32(&mut self, value: u32) {
        self.add(value as u64);
    }
    #[inline]
    fn write_u64(&mut self, value: u64) {
        self.add(value);
    }
    #[inline]
    fn write_usize(&mut self, value: usize) {
        self.add(value as u64);
    }
    #[inline]
    fn finish(&self) -> u64 {
        self.hash
    }
}

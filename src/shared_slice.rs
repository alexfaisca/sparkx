use memmap::{Mmap, MmapMut, MmapOptions};
use std::{
    fs::File,
    io::{Error, ErrorKind},
};

#[derive(Copy, Clone)]
pub struct SharedSlice<T> {
    ptr: *const T,
    len: usize,
}

unsafe impl<T> Send for SharedSlice<T> {}
unsafe impl<T> Sync for SharedSlice<T> {}

impl<T> SharedSlice<T> {
    pub fn new(ptr: *const T, len: usize) -> Self {
        SharedSlice::<T> { ptr, len }
    }
    pub fn from_file(file: &File) -> Result<(Self, Mmap), Error> {
        let mmap = unsafe { MmapOptions::new().map(file)? };
        let mmap_len = mmap.len();

        // Ensure the memory is properly aligned
        if mmap_len % std::mem::size_of::<T>() != 0 {
            return Err(Error::new(
                ErrorKind::InvalidData,
                "file length is not a multiple of u64",
            ));
        }

        // Convert &[u8] -> &[AtomicU64]
        Ok((
            Self::new(
                mmap.as_ptr() as *const T,
                mmap_len / std::mem::size_of::<T>(),
            ),
            mmap,
        ))
    }
    pub fn get(&self, idx: usize) -> &T {
        assert!(idx < self.len);
        unsafe { &*self.ptr.add(idx) }
    }
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn slice(&self, start: usize, end: usize) -> Option<&[T]> {
        if start >= self.len {
            return None;
        }
        let end = if end > self.len { self.len } else { end };
        unsafe { Some(std::slice::from_raw_parts(self.ptr.add(start), end - start)) }
    }
}

#[derive(Copy, Clone)]
pub struct SharedSliceMut<T> {
    ptr: *mut T,
    len: usize,
}

unsafe impl<T> Send for SharedSliceMut<T> {}
unsafe impl<T> Sync for SharedSliceMut<T> {}

impl<T> SharedSliceMut<T> {
    pub fn new(ptr: *mut T, len: usize) -> Self {
        SharedSliceMut::<T> { ptr, len }
    }
    pub fn from_file(file: &File) -> Result<(Self, MmapMut), Error> {
        let mmap = unsafe { MmapOptions::new().map_mut(file)? };
        let mmap_len = mmap.len();

        // Ensure the memory is properly aligned
        if mmap_len % std::mem::size_of::<T>() != 0 {
            return Err(Error::new(
                ErrorKind::InvalidData,
                "file length is not a multiple of u64",
            ));
        }

        // Convert &[u8] -> &[AtomicU64]
        Ok((
            Self::new(mmap.as_ptr() as *mut T, mmap_len / std::mem::size_of::<T>()),
            mmap,
        ))
    }
    pub fn get(&self, idx: usize) -> &T {
        assert!(idx < self.len);
        unsafe { &*self.ptr.add(idx) }
    }
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn slice(&self, start: usize, end: usize) -> Option<&[T]> {
        if start >= self.len {
            return None;
        }
        let end = if end > self.len { self.len } else { end };
        unsafe { Some(std::slice::from_raw_parts(self.ptr.add(start), end - start)) }
    }
    pub fn mut_slice(&mut self, start: usize, end: usize) -> Option<&mut [T]> {
        if start >= self.len {
            return None;
        }
        let end = if end > self.len { self.len } else { end };
        unsafe {
            Some(std::slice::from_raw_parts_mut(
                self.ptr.add(start),
                end - start,
            ))
        }
    }
}

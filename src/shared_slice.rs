use memmap::{Mmap, MmapMut, MmapOptions};
use std::{
    fs::{File, OpenOptions},
    io::{Error, ErrorKind},
};

#[derive(Copy, Clone)]
pub struct SharedSlice<T> {
    ptr: *const T,
    len: usize,
}

pub struct AbstractedProceduralMemory<T> {
    pub slice: SharedSlice<T>,
    mmap: Mmap,
    vec: Vec<T>,
    mmapped: bool,
}

pub struct AbstractedProceduralMemoryMut<'a, T> {
    pub slice: SharedSliceMut<T>,
    mmap: MmapMut,
    vec: &'a mut Vec<T>,
    mmapped: bool,
}

impl<'a, T> AbstractedProceduralMemoryMut<'a, T> {
    pub fn flush(&self) -> Result<(), Error> {
        if self.mmapped {
            self.mmap.flush()
        } else {
            Ok(())
        }
    }
    pub fn _flush_async(&self) -> Result<(), Error> {
        if self.mmapped {
            self.mmap.flush_async()
        } else {
            Ok(())
        }
    }
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
    pub fn _len(&self) -> usize {
        self.len
    }
    pub fn slice(&self, start: usize, end: usize) -> Option<&[T]> {
        if start >= self.len {
            return None;
        }
        let end = if end > self.len { self.len } else { end };
        unsafe { Some(std::slice::from_raw_parts(self.ptr.add(start), end - start)) }
    }
    pub fn abstract_mem(
        mfn: &str,
        vec: Vec<T>,
        len: usize,
        mmapped: bool,
    ) -> Result<AbstractedProceduralMemory<T>, Error> {
        let file = OpenOptions::new().read(true).open(mfn)?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };

        // Ensure the memory is properly aligned
        if mmapped && mmap.len() < std::mem::size_of::<T>() * len {
            return Err(Error::new(
                ErrorKind::InvalidData,
                "file length is too small for the amount of elements requested",
            ));
        }
        // Use an in-memory vector for degrees
        // Initialize with 0s
        if !mmapped && vec.len() < len {
            return Err(Error::new(
                ErrorKind::InvalidData,
                "file length is too small for the amount of elements requested",
            ));
        }
        let slice: SharedSlice<T> = {
            if mmapped {
                SharedSlice::<T>::new(mmap.as_ptr() as *const T, len)
            } else {
                SharedSlice::<T>::new(vec.as_ptr(), len)
            }
        };
        Ok(AbstractedProceduralMemory {
            slice,
            vec,
            mmap,
            mmapped,
        })
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

        Ok((
            Self::new(mmap.as_ptr() as *mut T, mmap_len / std::mem::size_of::<T>()),
            mmap,
        ))
    }
    pub fn get_mut(&mut self, idx: usize) -> &mut T {
        assert!(idx < self.len);
        unsafe { &mut *self.ptr.add(idx) }
    }
    pub fn get(&self, idx: usize) -> &T {
        assert!(idx < self.len);
        unsafe { &*self.ptr.add(idx) }
    }
    pub fn _len(&self) -> usize {
        self.len
    }
    pub fn _slice(&self, start: usize, end: usize) -> Option<&[T]> {
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
    pub fn abstract_mem_mut<'a>(
        mfn: &str,
        vec: &'a mut Vec<T>,
        len: usize,
        mmapped: bool,
    ) -> Result<AbstractedProceduralMemoryMut<'a, T>, Error> {
        let file = OpenOptions::new()
            .create(true)
            .truncate(true)
            .read(true)
            .write(true)
            .open(mfn)?;
        file.set_len((if mmapped { len } else { 1 } * std::mem::size_of::<T>()) as u64)?;
        let mut mmap = unsafe { MmapOptions::new().map_mut(&file)? };
        if !mmapped {
            vec.reserve_exact(len - vec.len());
        }
        let slice: SharedSliceMut<T> = {
            if mmapped {
                SharedSliceMut::<T>::new(mmap.as_mut_ptr() as *mut T, len)
            } else {
                SharedSliceMut::<T>::new(vec.as_mut_ptr(), len)
            }
        };
        Ok(AbstractedProceduralMemoryMut {
            slice,
            vec,
            mmap,
            mmapped,
        })
    }
}

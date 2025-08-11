use memmap2::{Mmap, MmapMut, MmapOptions};
use std::{
    fmt::Debug,
    fs::{File, OpenOptions},
    io::{Error, ErrorKind},
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

#[derive(Copy, Debug)]
pub struct SharedSlice<T> {
    ptr: *const T,
    len: usize,
}

#[derive(Copy, Debug)]
pub struct SharedSliceMut<T> {
    pub ptr: *mut T,
    len: usize,
}

unsafe impl<T> Send for SharedSliceMut<T> {}
unsafe impl<T> Sync for SharedSliceMut<T> {}

impl<T> Clone for SharedSlice<T> {
    fn clone(&self) -> Self {
        Self {
            ptr: self.ptr,
            len: self.len,
        }
    }
}

impl<T> Clone for SharedSliceMut<T> {
    fn clone(&self) -> Self {
        Self {
            ptr: self.ptr,
            len: self.len,
        }
    }
}

#[derive(Debug)]
pub struct AbstractedProceduralMemory<T> {
    pub slice: SharedSlice<T>,
    _mmap: Mmap,
    _vec: Vec<T>,
    _mmapped: bool,
}

#[derive(Debug)]
pub struct AbstractedProceduralMemoryMut<T> {
    pub slice: SharedSliceMut<T>,
    mmap: MmapMut,
    _vec: Vec<T>,
    mmapped: bool,
}

#[allow(dead_code)]
impl<T> AbstractedProceduralMemory<T> {
    pub fn shared_slice(&self) -> SharedSlice<T> {
        self.slice.clone()
    }
}

#[allow(dead_code)]
impl<T> AbstractedProceduralMemoryMut<T> {
    pub fn shared_slice(&self) -> SharedSliceMut<T> {
        self.slice.clone()
    }
    pub fn get(&self, idx: usize) -> &T {
        self.slice.get(idx)
    }
    pub fn get_mut(&mut self, idx: usize) -> &mut T {
        self.slice.get_mut(idx)
    }
    pub fn slice(&self, start: usize, end: usize) -> Option<&[T]> {
        self.slice.slice(start, end)
    }
    pub fn mut_slice(&mut self, start: usize, end: usize) -> Option<&mut [T]> {
        self.slice.mut_slice(start, end)
    }
    pub fn write_slice(&mut self, idx: usize, slice: &[T]) -> Option<usize> {
        self.slice.write_slice(idx, slice)
    }
    pub fn flush(&self) -> Result<(), Error> {
        if self.mmapped {
            self.mmap.flush()
        } else {
            Ok(())
        }
    }
    pub fn flush_async(&self) -> Result<(), Error> {
        if self.mmapped {
            self.mmap.flush_async()
        } else {
            Ok(())
        }
    }
}

unsafe impl<T> Send for SharedSlice<T> {}
unsafe impl<T> Sync for SharedSlice<T> {}

#[allow(dead_code)]
impl<T> SharedSlice<T> {
    pub fn new(ptr: *const T, len: usize) -> Self {
        SharedSlice::<T> { ptr, len }
    }
    pub fn from_slice(slice: &[T]) -> Self {
        SharedSlice::<T> {
            ptr: slice.as_ptr(),
            len: slice.len(),
        }
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
    pub fn abstract_mem(
        mfn: String,
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
            _vec: vec,
            _mmap: mmap,
            _mmapped: mmapped,
        })
    }
}

#[allow(dead_code)]
impl<T> SharedSliceMut<T> {
    pub fn new(ptr: *mut T, len: usize) -> Self {
        SharedSliceMut::<T> { ptr, len }
    }
    pub fn from_slice(slice: &mut [T]) -> Self {
        SharedSliceMut::<T> {
            ptr: slice.as_mut_ptr(),
            len: slice.len(),
        }
    }
    pub fn _from_shared_slice(slice: SharedSliceMut<T>) -> Self {
        SharedSliceMut::<T> {
            ptr: slice.ptr,
            len: slice.len,
        }
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
    pub unsafe fn _cast<U>(&self) -> Option<SharedSliceMut<U>> {
        if std::mem::size_of::<T>() != std::mem::size_of::<U>() {
            return None;
        }
        Some(SharedSliceMut {
            ptr: self.ptr as *mut U,
            len: self.len,
        })
    }
    pub fn get_mut(&mut self, idx: usize) -> &mut T {
        assert!(idx < self.len);
        unsafe { &mut *self.ptr.add(idx) }
    }
    pub fn get(&self, idx: usize) -> &T {
        assert!(idx < self.len);
        unsafe { &*self.ptr.add(idx) }
    }
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn slice(&self, start: usize, end: usize) -> Option<&[T]> {
        assert!(start <= end && end <= self.len);
        unsafe { Some(std::slice::from_raw_parts(self.ptr.add(start), end - start)) }
    }
    pub fn mut_slice(&mut self, start: usize, end: usize) -> Option<&mut [T]> {
        assert!(start <= end && end <= self.len);
        unsafe {
            Some(std::slice::from_raw_parts_mut(
                self.ptr.add(start),
                end - start,
            ))
        }
    }

    pub fn write_shared_slice(
        &mut self,
        slice: SharedSlice<T>,
        idx: usize,
        from: usize,
        len: usize,
    ) -> Option<usize> {
        assert!(idx + len <= self.len);
        unsafe {
            std::ptr::copy_nonoverlapping(slice.ptr.add(from), self.ptr.add(idx), len);
        };
        Some(idx + slice.len())
    }

    pub fn write_slice(&mut self, idx: usize, slice: &[T]) -> Option<usize> {
        // if idx == len but slice.len() == 0, the write should be valid
        assert!(idx + slice.len() <= self.len);
        unsafe {
            std::ptr::copy_nonoverlapping(slice.as_ptr(), self.ptr.add(idx), slice.len());
        };
        Some(idx + slice.len())
    }
    pub fn abst_mem_mut(
        mfn: String,
        len: usize,
        mmapped: bool,
    ) -> Result<AbstractedProceduralMemoryMut<T>, Error> {
        let file = OpenOptions::new()
            .create(true)
            .truncate(true)
            .read(true)
            .write(true)
            .open(mfn)?;
        file.set_len((if mmapped { len } else { 1 } * std::mem::size_of::<T>()) as u64)?;
        let mut mmap = unsafe { MmapOptions::new().map_mut(&file)? };
        let mut vec: Vec<T> = Vec::new();
        if !mmapped {
            vec.reserve_exact(len);
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
            _vec: vec,
            mmap,
            mmapped,
        })
    }
}

#[derive(Clone)]
pub struct SharedQueueMut<T> {
    pub ptr: *mut T,
    max: usize,
    read: Arc<AtomicUsize>,
    write: Arc<AtomicUsize>,
    len: Arc<AtomicUsize>,
}

unsafe impl<T> Send for SharedQueueMut<T> {}
unsafe impl<T> Sync for SharedQueueMut<T> {}

impl<T: Debug + Copy + Clone + Eq> SharedQueueMut<T> {
    pub fn from_shared_slice(slice: SharedSliceMut<T>) -> Self {
        SharedQueueMut::<T> {
            ptr: slice.ptr,
            max: slice.len,
            read: Arc::new(AtomicUsize::new(0)),
            write: Arc::new(AtomicUsize::new(0)),
            len: Arc::new(AtomicUsize::new(0)),
        }
    }
    pub fn len(&self) -> usize {
        self.len.load(Ordering::Relaxed)
    }
    pub fn is_empty(&self) -> bool {
        self.len.load(Ordering::Relaxed) == 0
    }
    // FIXME: In concurrent push pops reads and writes may give undefined behaviour? push_async?
    pub fn push_slice(&mut self, slice: &[T]) -> Option<usize> {
        let write_idx = self.write.fetch_add(slice.len(), Ordering::SeqCst);
        if write_idx + slice.len() < self.max {
            unsafe {
                std::ptr::copy_nonoverlapping(slice.as_ptr(), self.ptr.add(write_idx), slice.len());
            };
            self.len.fetch_add(slice.len(), Ordering::SeqCst);
            Some(write_idx)
        } else {
            println!(
                "Failed to push write_idx {} slice len {}",
                write_idx,
                slice.len()
            );
            self.write.fetch_sub(slice.len(), Ordering::SeqCst);
            None
        }
    }
    pub fn push(&mut self, el: T) -> Option<usize> {
        let write_idx = self.write.fetch_add(1, Ordering::SeqCst);
        if write_idx < self.max {
            unsafe { *self.ptr.add(write_idx) = el };
            self.len.fetch_add(1, Ordering::SeqCst);
            Some(write_idx)
        } else {
            // println!("Failed to push");
            None
        }
    }
    pub fn pop(&mut self) -> Option<T> {
        let idx = self
            .read
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |x| {
                if x < self.len.load(Ordering::SeqCst) {
                    Some(x + 1)
                } else {
                    // print!(
                    //     "FAIL (read: {} len: {} max: {}) ",
                    //     x,
                    //     self.len.load(Ordering::SeqCst),
                    //     self.max
                    // );
                    None
                }
            })
            .ok();
        if let Some(offset) = idx {
            Some(unsafe { *self.ptr.add(offset) })
        } else {
            None
        }
    }
    pub fn slice(&self, start: usize, end: usize) -> Option<SharedSliceMut<T>> {
        let start = if start < self.read.load(Ordering::Relaxed) {
            self.read.load(Ordering::Relaxed)
        } else {
            start
        };
        let end = if end > self.len.load(Ordering::Relaxed) {
            self.len.load(Ordering::Relaxed)
        } else {
            if end < start {
                return None;
            }
            end
        };
        unsafe { Some(SharedSliceMut::new(self.ptr.add(start), end - start)) }
    }
    pub unsafe fn _raw_slice(&self) -> SharedSliceMut<T> {
        SharedSliceMut::new(self.ptr, self.max)
    }
    pub fn clear(self) -> Self {
        self.read.store(0, Ordering::SeqCst);
        self.write.store(0, Ordering::SeqCst);
        self.len.store(0, Ordering::SeqCst);
        self
    }
}

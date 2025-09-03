use memmap2::{Mmap, MmapMut, MmapOptions};
use std::{
    fmt::Debug,
    fs::{File, OpenOptions},
    mem::{ManuallyDrop, size_of},
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

#[derive(Debug)]
#[allow(dead_code)]
pub struct AbstractedProceduralMemory<T> {
    slice: SharedSlice<T>,
    mmap: Mmap,
    _vec: Vec<T>,
    mmapped: bool,
}

#[derive(Debug)]
pub struct AbstractedProceduralMemoryMut<T> {
    slice: SharedSliceMut<T>,
    mmap: MmapMut,
    _vec: Vec<T>,
    mmapped: bool,
}

#[derive(Copy, Debug)]
pub(crate) struct SharedSlice<T> {
    ptr: *const T,
    len: usize,
}

#[derive(Copy, Debug)]
pub struct SharedSliceMut<T> {
    pub ptr: *mut T,
    len: usize,
}

#[derive(Clone)]
pub(crate) struct SharedQueueMut<T> {
    pub ptr: *mut T,
    max: usize,
    read: Arc<AtomicUsize>,
    write: Arc<AtomicUsize>,
    len: Arc<AtomicUsize>,
}

unsafe impl<T> Send for SharedSlice<T> {}
unsafe impl<T> Sync for SharedSlice<T> {}

unsafe impl<T> Send for SharedSliceMut<T> {}
unsafe impl<T> Sync for SharedSliceMut<T> {}

unsafe impl<T> Send for SharedQueueMut<T> {}
unsafe impl<T> Sync for SharedQueueMut<T> {}

impl<T> Clone for SharedSlice<T> {
    #[inline(always)]
    fn clone(&self) -> Self {
        Self {
            ptr: self.ptr,
            len: self.len,
        }
    }
}

impl<T> Clone for SharedSliceMut<T> {
    #[inline(always)]
    fn clone(&self) -> Self {
        Self {
            ptr: self.ptr,
            len: self.len,
        }
    }
}

#[allow(dead_code)]
impl<T> AbstractedProceduralMemory<T> {
    pub(crate) fn from_file_name(file_name: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let file = OpenOptions::new()
            .read(true)
            .truncate(false)
            .write(false)
            .open(file_name)?;
        let (slice, mmap) = SharedSlice::<T>::from_file(&file)?;

        Ok(AbstractedProceduralMemory {
            slice,
            mmap,
            _vec: Vec::new(),
            mmapped: true,
        })
    }
    #[inline(always)]
    pub(crate) fn shared_slice(&self) -> SharedSlice<T> {
        self.slice.clone()
    }

    #[inline(always)]
    pub(crate) fn get(&self, idx: usize) -> &T {
        self.slice.get(idx)
    }

    #[inline(always)]
    pub(crate) fn as_ptr(&self) -> *const T {
        self.slice.ptr
    }

    #[inline(always)]
    pub(crate) fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.slice.ptr, self.slice.len) }
    }

    #[allow(unused_mut)]
    #[inline(always)]
    pub(crate) fn into_bodex_slice(mut self) -> Box<[T]> {
        let this = ManuallyDrop::new(self);
        let raw: *mut [T] =
            std::ptr::slice_from_raw_parts_mut(this.slice.ptr as *mut T, this.slice.len);
        // SAFETY: `raw` points to a uniquely-owned heap allocation of exactly `len` Ts,
        unsafe { Box::from_raw(raw) }
    }

    #[inline(always)]
    pub(crate) fn slice(&self, start: usize, end: usize) -> Option<&[T]> {
        self.slice.slice(start, end)
    }

    #[inline(always)]
    pub(crate) fn len(&self) -> usize {
        self.shared_slice().len()
    }

    #[inline(always)]
    pub(crate) fn is_empty(&self) -> bool {
        self.shared_slice().len() == 0
    }
}

#[allow(dead_code)]
impl<T> AbstractedProceduralMemoryMut<T> {
    pub(crate) fn from_file(file: &File, len: usize) -> Result<Self, Box<dyn std::error::Error>> {
        let mmap_len = (len * size_of::<T>()) as u64;
        file.set_len(mmap_len)?;
        let (slice, mmap) = SharedSliceMut::<T>::from_file(file)?;

        Ok(AbstractedProceduralMemoryMut {
            slice,
            mmap,
            _vec: Vec::new(),
            mmapped: true,
        })
    }

    #[inline(always)]
    pub(crate) fn shared_slice(&self) -> SharedSliceMut<T> {
        self.slice.clone()
    }

    #[inline(always)]
    pub(crate) fn get(&self, idx: usize) -> &T {
        self.slice.get(idx)
    }

    #[inline(always)]
    pub fn get_mut(&mut self, idx: usize) -> &mut T {
        self.slice.get_mut(idx)
    }

    #[inline(always)]
    pub(crate) fn as_ptr(&self) -> *const T {
        self.slice.ptr
    }

    #[inline(always)]
    pub(crate) fn as_mut_ptr(&mut self) -> *mut T {
        self.slice.ptr
    }

    #[inline(always)]
    pub(crate) fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.slice.ptr, self.slice.len) }
    }

    #[inline(always)]
    pub(crate) fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.slice.ptr, self.slice.len) }
    }

    #[allow(unused_mut)]
    #[inline(always)]
    pub(crate) fn into_bodex_slice(mut self) -> Box<[T]> {
        let this = ManuallyDrop::new(self);
        let raw: *mut [T] = std::ptr::slice_from_raw_parts_mut(this.slice.ptr, this.slice.len);
        // SAFETY: `raw` points to a uniquely-owned heap allocation of exactly `len` Ts,
        unsafe { Box::from_raw(raw) }
    }

    #[inline(always)]
    pub(crate) fn slice(&self, start: usize, end: usize) -> Option<&[T]> {
        self.slice.slice(start, end)
    }

    #[inline(always)]
    pub(crate) fn mut_slice(&mut self, start: usize, end: usize) -> Option<&mut [T]> {
        self.slice.mut_slice(start, end)
    }

    #[inline(always)]
    pub(crate) fn write_slice(&mut self, idx: usize, slice: &[T]) -> Option<usize> {
        self.slice.write_slice(idx, slice)
    }

    #[inline(always)]
    pub(crate) fn len(&self) -> usize {
        self.shared_slice().len()
    }

    #[inline(always)]
    pub(crate) fn is_empty(&self) -> bool {
        self.shared_slice().len() == 0
    }

    pub fn flush(&self) -> Result<(), Box<dyn std::error::Error>> {
        if self.mmapped {
            Ok(self.mmap.flush()?)
        } else {
            Ok(())
        }
    }

    pub(crate) fn flush_async(&self) -> Result<(), Box<dyn std::error::Error>> {
        if self.mmapped {
            Ok(self.mmap.flush_async()?)
        } else {
            Ok(())
        }
    }
}

#[allow(dead_code)]
impl<T> SharedSlice<T> {
    pub(crate) fn new(ptr: *const T, len: usize) -> Self {
        SharedSlice::<T> { ptr, len }
    }

    pub(crate) fn from_slice(slice: &[T]) -> Self {
        SharedSlice::<T> {
            ptr: slice.as_ptr(),
            len: slice.len(),
        }
    }

    #[allow(clippy::ptr_arg)]
    pub(crate) fn from_vec(vec: &Vec<T>) -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self::new(vec.as_ptr(), vec.len()))
    }

    pub(crate) fn from_file(file: &File) -> Result<(Self, Mmap), Box<dyn std::error::Error>> {
        let file_len = file.metadata()?.len();
        let mmap = unsafe { MmapOptions::new().len(file_len as usize).map(file)? };
        let mmap_len = mmap.len();

        // Ensure the memory is properly aligned
        if mmap_len % size_of::<T>() != 0 {
            return Err("file length is not a multiple of u64".into());
        }

        Ok((
            Self::new(mmap.as_ptr() as *const T, mmap_len / size_of::<T>()),
            mmap,
        ))
    }

    #[inline(always)]
    pub(self) fn ptr(&self) -> *const T {
        self.ptr
    }

    #[inline(always)]
    pub(crate) fn get(&self, idx: usize) -> &T {
        assert!(idx < self.len);
        unsafe { &*self.ptr.add(idx) }
    }

    #[inline(always)]
    pub(crate) fn len(&self) -> usize {
        self.len
    }

    #[inline(always)]
    pub(crate) fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline(always)]
    pub(crate) fn as_ptr(&self) -> *const T {
        self.ptr
    }

    #[inline(always)]
    pub(crate) fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    // FIXME: if `mut` is removed does unique ownership get comprimized? Is it alwready
    // comprimized?
    #[allow(unused_mut)]
    #[inline(always)]
    pub(crate) fn into_bodex_slice(mut self) -> Box<[T]> {
        let this = ManuallyDrop::new(self);
        let raw: *mut [T] = std::ptr::slice_from_raw_parts_mut(this.ptr as *mut T, this.len);
        // SAFETY: `raw` points to a uniquely-owned heap allocation of exactly `len` Ts,
        unsafe { Box::from_raw(raw) }
    }

    #[inline(always)]
    pub(crate) fn slice(&self, start: usize, end: usize) -> Option<&[T]> {
        if start > self.len {
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
    ) -> Result<AbstractedProceduralMemory<T>, Box<dyn std::error::Error>> {
        let file = OpenOptions::new().read(true).open(mfn)?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };

        // Ensure the memory is properly aligned
        if mmapped && mmap.len() < size_of::<T>() * len {
            return Err("file length is too small for the amount of elements requested".into());
        }
        // Use an in-memory vector for degrees
        // Initialize with 0s
        if !mmapped && vec.len() < len {
            return Err("vec length is too small for the amount of elements requested".into());
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
            mmap,
            mmapped,
        })
    }
}

#[allow(dead_code)]
impl<T> SharedSliceMut<T> {
    pub(crate) fn new(ptr: *mut T, len: usize) -> Self {
        SharedSliceMut::<T> { ptr, len }
    }

    pub(crate) fn from_slice(slice: &mut [T]) -> Self {
        SharedSliceMut::<T> {
            ptr: slice.as_mut_ptr(),
            len: slice.len(),
        }
    }

    pub(crate) fn from_shared_slice(slice: SharedSliceMut<T>) -> Self {
        SharedSliceMut::<T> {
            ptr: slice.ptr,
            len: slice.len,
        }
    }

    pub(crate) fn from_vec(vec: &mut Vec<T>) -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self::new(vec.as_mut_ptr(), vec.len()))
    }

    pub(crate) fn from_file(file: &File) -> Result<(Self, MmapMut), Box<dyn std::error::Error>> {
        let mmap = unsafe { MmapOptions::new().map_mut(file)? };
        let mmap_len = mmap.len();

        // Ensure the memory is properly aligned
        if mmap_len % size_of::<T>() != 0 {
            return Err("file length is not a multiple of u64".into());
        }

        Ok((
            Self::new(mmap.as_ptr() as *mut T, mmap_len / size_of::<T>()),
            mmap,
        ))
    }

    #[inline(always)]
    pub(crate) unsafe fn cast<U>(&self) -> Option<SharedSliceMut<U>> {
        if size_of::<T>() != size_of::<U>() {
            return None;
        }
        Some(SharedSliceMut {
            ptr: self.ptr as *mut U,
            len: self.len,
        })
    }

    #[inline(always)]
    pub(self) fn ptr(&self) -> *const T {
        self.ptr
    }

    #[inline(always)]
    pub(crate) fn get_mut(&mut self, idx: usize) -> &mut T {
        assert!(idx < self.len, "index {} >= len {}", idx, self.len);
        unsafe { &mut *self.ptr.add(idx) }
    }

    #[inline(always)]
    pub(crate) fn get(&self, idx: usize) -> &T {
        assert!(idx < self.len, "index {} >= len {}", idx, self.len);
        unsafe { &*self.ptr.add(idx) }
    }

    #[inline(always)]
    pub(crate) fn len(&self) -> usize {
        self.len
    }

    #[inline(always)]
    pub(crate) fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline(always)]
    pub(crate) fn as_ptr(&self) -> *const T {
        self.ptr
    }

    #[inline(always)]
    pub(crate) fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }

    #[inline(always)]
    pub(crate) fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    #[inline(always)]
    pub(crate) fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }

    #[allow(unused_mut)]
    #[inline(always)]
    pub(crate) fn into_bodex_slice(mut self) -> Box<[T]> {
        let this = ManuallyDrop::new(self);
        let raw: *mut [T] = std::ptr::slice_from_raw_parts_mut(this.ptr, this.len);
        // SAFETY: `raw` points to a uniquely-owned heap allocation of exactly `len` Ts,
        unsafe { Box::from_raw(raw) }
    }

    #[inline(always)]
    pub(crate) fn slice(&self, start: usize, end: usize) -> Option<&[T]> {
        assert!(start <= end && end <= self.len);
        unsafe { Some(std::slice::from_raw_parts(self.ptr.add(start), end - start)) }
    }

    #[inline(always)]
    pub(crate) fn mut_slice(&mut self, start: usize, end: usize) -> Option<&mut [T]> {
        assert!(start <= end && end <= self.len);
        unsafe {
            Some(std::slice::from_raw_parts_mut(
                self.ptr.add(start),
                end - start,
            ))
        }
    }

    #[inline(always)]
    pub(crate) fn write_shared_slice(
        &mut self,
        slice: SharedSlice<T>,
        idx: usize,
        from: usize,
        len: usize,
    ) -> Option<usize> {
        assert!(idx + len <= self.len);
        unsafe {
            std::ptr::copy_nonoverlapping(slice.ptr().add(from), self.ptr.add(idx), len);
        };
        Some(idx + slice.len())
    }

    #[inline(always)]
    pub(crate) fn write_shared_slice_mut(
        &mut self,
        slice: SharedSliceMut<T>,
        idx: usize,
        from: usize,
        len: usize,
    ) -> Option<usize> {
        assert!(idx + len <= self.len);
        unsafe {
            std::ptr::copy_nonoverlapping(slice.ptr().add(from), self.ptr.add(idx), len);
        };
        Some(idx + slice.len())
    }

    #[inline(always)]
    pub(crate) fn write_slice(&mut self, idx: usize, slice: &[T]) -> Option<usize> {
        // if idx == len but slice.len() == 0, the write should be valid
        assert!(idx + slice.len() <= self.len);
        unsafe {
            std::ptr::copy_nonoverlapping(slice.as_ptr(), self.ptr.add(idx), slice.len());
        };
        Some(idx + slice.len())
    }

    pub fn abst_mem_mut(
        mfn: &str,
        len: usize,
        mmapped: bool,
    ) -> Result<AbstractedProceduralMemoryMut<T>, Box<dyn std::error::Error>> {
        let file = OpenOptions::new()
            .create(true)
            .truncate(true)
            .read(true)
            .write(true)
            .open(mfn)?;
        file.set_len((if mmapped { len } else { 1 } * size_of::<T>()) as u64)?;

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

#[allow(dead_code)]
impl<T: Debug + Copy + Clone + Eq> SharedQueueMut<T> {
    pub(crate) fn from_shared_slice(slice: SharedSliceMut<T>) -> Self {
        SharedQueueMut::<T> {
            ptr: slice.ptr,
            max: slice.len,
            read: Arc::new(AtomicUsize::new(0)),
            write: Arc::new(AtomicUsize::new(0)),
            len: Arc::new(AtomicUsize::new(0)),
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.len.load(Ordering::Relaxed)
    }
    pub(crate) fn is_empty(&self) -> bool {
        self.len.load(Ordering::Relaxed) == 0
    }

    pub(crate) fn push_slice(&mut self, slice: &[T]) -> Option<usize> {
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

    pub(crate) fn push(&mut self, el: T) -> Option<usize> {
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

    pub(crate) fn pop(&mut self) -> Option<T> {
        let idx = self
            .read
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |x| {
                if x < self.len.load(Ordering::SeqCst) {
                    Some(x + 1)
                } else {
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

    #[inline(always)]
    pub(crate) fn as_ptr(&self) -> *const T {
        self.ptr
    }

    #[inline(always)]
    pub(crate) fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }

    #[inline(always)]
    pub(crate) fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len()) }
    }

    #[inline(always)]
    pub(crate) fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len()) }
    }

    #[allow(unused_mut)]
    #[inline(always)]
    /// Returns a boxed slice over the entirity of the allocated memory[^1].
    ///
    /// [^1]: naturally, if any uninitialized memory cells remain at the time of this method's calling,
    /// behavioiur upon their reading is undefined.
    pub(crate) fn into_bodex_slice(mut self) -> Box<[T]> {
        let this = ManuallyDrop::new(self);
        let raw: *mut [T] = std::ptr::slice_from_raw_parts_mut(this.ptr, this.max);
        // SAFETY: `raw` points to a uniquely-owned heap allocation of exactly `len` Ts,
        unsafe { Box::from_raw(raw) }
    }

    pub(crate) fn slice(&self, start: usize, end: usize) -> Option<SharedSliceMut<T>> {
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

    pub(crate) unsafe fn raw_slice(&self) -> SharedSliceMut<T> {
        SharedSliceMut::new(self.ptr, self.max)
    }

    pub(crate) fn clear(self) -> Self {
        self.read.store(0, Ordering::SeqCst);
        self.write.store(0, Ordering::SeqCst);
        self.len.store(0, Ordering::SeqCst);
        self
    }
}

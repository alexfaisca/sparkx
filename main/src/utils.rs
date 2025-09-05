use smallvec::SmallVec;
use std::any::type_name;

#[allow(dead_code)]
pub fn type_of<T>() -> &'static str {
    type_name::<T>()
}

#[derive(Debug)]
pub(crate) enum OneOrMany<T: Sized> {
    One(T),
    Many(SmallVec<[T; 4]>),
}

/// Checks if a `val` is a normal `f64`. Outputs a result with a custom error message.
///
/// # Arguments
///
/// * `val`: `f64` --- the value to be checked.
/// * `op_description`: `&str` --- the custom error message.
///
/// # Returns
///
/// `Ok(val)` if `val` is normal, or `Err(op_description.into())` if not.
///
#[inline(always)]
pub(crate) fn f64_is_nomal(
    val: f64,
    op_description: &str,
) -> Result<f64, Box<dyn std::error::Error>> {
    if !val.is_normal() {
        return Err(format!("error hk-relax abnormal value at {op_description} = {val}",).into());
    }
    Ok(val)
}

/// Safely converts an `f64` `val` into `usize`. Outputs an option.
///
/// Conversion is successful if `val` is normal, bigger than zero (not equal) and less than or equal to `usize::MAX`.
///
/// # Arguments
///
/// * `val`: `f64` --- the value to be cast.
///
/// # Returns
///
/// `Some(val as usize)` if successful, or None if not.
///
pub(crate) fn f64_to_usize_safe(val: f64) -> Option<usize> {
    if val.is_normal() && val > 0f64 && val <= usize::MAX as f64 {
        Some(val as usize) // truncates toward zero
    } else {
        None
    }
}

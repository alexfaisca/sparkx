use core::panic;

use proc_macro::TokenStream;
use quote::quote;
use syn::{Attribute, Data, DeriveInput, Fields, Meta, MetaList, Type, parse_macro_input};

#[proc_macro_derive(GenericEdge, attributes(edge_dest, edge_type))]
pub fn derive_generic_edge(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = input.ident.clone();

    let default_edge_dest_field_name = "dest_node";
    let default_edge_type_field_name = "edge_type";
    assert_ne!(default_edge_dest_field_name, default_edge_type_field_name);

    // check if #[repr(C)] is present
    let has_repr_c = input.attrs.iter().any(|attr: &Attribute| {
        if attr.path().is_ident("repr") {
            if let Meta::List(MetaList { tokens, .. }) = &attr.meta {
                tokens.to_string().contains("C")
            } else {
                false
            }
        } else {
            false
        }
    });

    if !has_repr_c {
        return syn::Error::new_spanned(
            input.ident,
            "Types deriving `GenericEdge` must be annotated with #[repr(C)]",
        )
        .to_compile_error()
        .into();
    }

    let _data = if let Data::Struct(ds) = input.clone().data {
        ds
    } else {
        return syn::Error::new_spanned(input.ident, "Only structs may derive `GenericEdge`")
            .to_compile_error()
            .into();
    };

    let mut impl_new_for_self = None;
    let mut impl_default_for_self = None;
    let mut quoted_dest_access = None;
    let mut quoted_edge_type_access = None;
    let mut quoted_mut_dest = None;
    let mut quoted_mut_edge_type = None;
    let mut quoted_dest_type = None;
    let mut quoted_edge_type = None;

    // --- Extract info from struct ---
    if let Data::Struct(s) = &input.data {
        match &s.fields {
            // ✅ Named fields (normal struct or bitfield struct with visible fields)
            Fields::Named(fields) => {
                let mut dest_access = None;
                let mut edge_type_access = None;
                let mut dest_access_real = None;
                let mut edge_type_access_real = None;
                let mut dest_setter = None;
                let mut edge_setter = None;
                let mut dest_access_fallback = None;
                let mut edge_type_access_fallback = None;
                let mut dest_access_type_fallback = None;
                let mut edge_type_access_type_fallback = None;

                for field in &fields.named {
                    let ident = field.ident.as_ref();
                    let ty = field.ty.clone();

                    // setup fallback if field name matches
                    if let Some(field_name) = ident {
                        if field_name == default_edge_dest_field_name {
                            dest_access_fallback = Some(field_name);
                            dest_access_type_fallback = Some(ty);
                        } else if field_name == default_edge_type_field_name {
                            edge_type_access_fallback = Some(field_name);
                            edge_type_access_type_fallback = Some(ty);
                        }
                    }

                    // Look for #[edge_dest] or #[edge_type] attributes
                    for attr in &field.attrs {
                        if attr.path().is_ident("edge_dest") {
                            let (getter, setter, ty) = parse_generic_edge_attr(attr);
                            dest_access = Some(getter.map_or_else(
                                || {
                                    let ident = match ident {
                                        Some(i) => i,
                                        None => {
                                            return syn::Error::new_spanned(name.clone(), "Struct derives `GenericEdge`, but no getter method or field was found for attribute `edge_dest`")
                                                .to_compile_error();
                                        }
                                    };
                                    quote! {
                                        self.#ident
                                    }
                                },
                                |g| quote! { self.#g() },
                            ));
                            dest_access_real = ident;
                            let type_of_dest = if let Some(t) = ty {
                                quoted_dest_type = Some(quote! {#t});
                                t
                            } else {
                                let t = field.ty.clone();
                                quoted_dest_type = Some(quote! {#t});
                                t
                            };
                            dest_setter = setter.map(|g| quote! { .#g( #type_of_dest::try_from(new_edge_dest).unwrap() ) });
                        }
                        if attr.path().is_ident("edge_type") {
                            let (getter, setter, ty) = parse_generic_edge_attr(attr);
                            edge_type_access = Some(getter.map_or_else(
                                || {
                                    let ident = match ident {
                                        Some(i) => i,
                                        None => {
                                            return syn::Error::new_spanned(name.clone(), "Struct derives `GenericEdge`, but no getter method or field was found for attribute `edge_type`")
                                                .to_compile_error();
                                        }
                                    };
                                    quote! {
                                        self.#ident
                                    }
                                },
                                |g| quote! { self.#g() },
                            ));
                            edge_type_access_real = ident;
                            let type_of_edge_type = if let Some(t) = ty {
                                quoted_edge_type = Some(quote! {#t});
                                t
                            } else {
                                let t = field.ty.clone();
                                quoted_edge_type = Some(quote! {#t});
                                t
                            };
                            edge_setter = setter
                                .map(|g| quote! { .#g( #type_of_edge_type::from(new_edge_type) ) });
                        }
                    }
                }

                if dest_access.is_none() {
                    match (dest_access_fallback, dest_access_type_fallback) {
                        (Some(field), Some(field_ty)) => {
                            quoted_dest_type = Some(quote! { #field_ty });
                            quoted_dest_access = Some(quote! {self.#field});
                        }
                        _ => {
                            return syn::Error::new_spanned(
                                name.clone(),
                                format!(
                                    "Missing #[edge_dest] attribute or fallback field `{}`",
                                    default_edge_dest_field_name
                                ),
                            )
                            .to_compile_error()
                            .into();
                        }
                    }
                } else {
                    quoted_dest_access = dest_access;
                }
                if edge_type_access.is_none() {
                    match (edge_type_access_fallback, edge_type_access_type_fallback) {
                        (Some(field), Some(field_ty)) => {
                            quoted_edge_type = Some(quote! { #field_ty });
                            quoted_edge_type_access = Some(quote! {self.#field});
                        }
                        _ => {
                            return syn::Error::new_spanned(
                                name.clone(),
                                format!(
                                    "Missing #[edge_type] attribute or fallback field `{}`",
                                    default_edge_type_field_name
                                ),
                            )
                            .to_compile_error()
                            .into();
                        }
                    }
                } else {
                    quoted_edge_type_access = edge_type_access;
                }
                let dest_ty = if let Some(ty) = quoted_dest_type.clone() {
                    ty
                } else {
                    panic!("can't proceed as edge_type clone failed")
                };
                let edge_ty = if let Some(ty) = quoted_edge_type.clone() {
                    ty
                } else {
                    panic!("can't proceed as edge_type clone failed")
                };
                // if named struct decide how to  implement method new(edge_dest: u64, edge_type: u64)
                match (
                    dest_setter,
                    edge_setter,
                    dest_access_real,
                    edge_type_access_real,
                    dest_access_fallback,
                    edge_type_access_fallback,
                ) {
                    (Some(d_s), Some(e_s), a, b, c, d) => {
                        // // interpolation d_s
                        // quote! { .#g( #type_of_edge_type::from(edge_type) ) }
                        quoted_mut_dest = Some(quote! { self #d_s; self });
                        quoted_mut_edge_type = Some(quote! { self #e_s; self });
                        match (a, b, c, d) {
                            (Some(d_r), Some(e_r), _, _) => {
                                impl_default_for_self = Some(
                                    quote! {#name {#d_r : #dest_ty::try_from(0u64).unwrap() , #e_r : #edge_ty::from(0u64) } },
                                );
                                impl_new_for_self = Some(
                                    quote! {#name {#d_r : #dest_ty::try_from(edge_dest).unwrap() , #e_r : #edge_ty::from(edge_type) } },
                                );
                            }
                            (None, Some(e_r), Some(d_f), _) => {
                                impl_default_for_self = Some(
                                    quote! {#name {#d_f : #dest_ty::try_from(0u64).unwrap() , #e_r : #edge_ty::from(0u64) } },
                                );
                                impl_new_for_self = Some(
                                    quote! {#name {#d_f : #dest_ty::try_from(edge_dest).unwrap() , #e_r : #edge_ty::from(edge_type) } },
                                );
                            }
                            (Some(d_r), _, _, Some(e_f)) => {
                                impl_default_for_self = Some(
                                    quote! {#name {#d_r : #dest_ty::try_from(0u64).unwrap() , #e_f : #edge_ty::from(0u64) } },
                                );
                                impl_new_for_self = Some(
                                    quote! {#name {#d_r : #dest_ty::try_from(edge_dest).unwrap() , #e_f : #edge_ty::from(edge_type) } },
                                );
                            }
                            (None, None, Some(d_f), Some(e_f)) => {
                                impl_default_for_self = Some(
                                    quote! {#name {#d_f : #dest_ty::try_from(0u64).unwrap() , #e_f : #edge_ty::from(0u64) } },
                                );
                                impl_new_for_self = Some(
                                    quote! {#name {#d_f : #dest_ty::try_from(edge_dest).unwrap() , #e_f : #edge_ty::from(edge_type) } },
                                );
                            }
                            _ => {
                                return syn::Error::new_spanned(
                                    name.clone(),
                                    "Named struct specified setter, but necessary fields (and their respective fallbacks) for default initialization"
                                    )
                                    .to_compile_error()
                                    .into();
                            }
                        };
                    }
                    (d_s, e_s, Some(d_r), Some(e_r), _, _) => {
                        if let Some(d_s) = d_s {
                            quoted_mut_dest = Some(quote! { self #d_s; self });
                        } else {
                            quoted_mut_dest = Some(
                                quote! { self.#d_r = #dest_ty::try_from(new_edge_dest).unwrap(); self },
                            );
                        }

                        if let Some(e_s) = e_s {
                            quoted_mut_edge_type = Some(quote! { self #e_s; self });
                        } else {
                            quoted_mut_edge_type =
                                Some(quote! { self.#e_r = #edge_ty::from(new_edge_type); self });
                        }

                        impl_default_for_self = Some(
                            quote! {#name {#d_r : #dest_ty::try_from(0u64).unwrap() , #e_r : #edge_ty::from(0u64) } },
                        );

                        impl_new_for_self = Some(
                            quote! {#name {#d_r : #dest_ty::try_from(edge_dest).unwrap() , #e_r : #edge_ty::from(edge_type) } },
                        );
                    }
                    (d_s, e_s, None, None, Some(d_f), Some(e_f)) => {
                        if let Some(d_s) = d_s {
                            quoted_mut_dest = Some(quote! { self #d_s; self });
                        } else {
                            quoted_mut_dest = Some(
                                quote! { self.#d_f = #dest_ty::try_from(new_edge_dest).unwrap(); self },
                            );
                        }

                        if let Some(e_s) = e_s {
                            quoted_mut_edge_type = Some(quote! { self #e_s; self });
                        } else {
                            quoted_mut_edge_type =
                                Some(quote! { self.#e_f = #edge_ty::from(new_edge_type); self });
                        }
                        impl_default_for_self = Some(
                            quote! {#name {#d_f : #dest_ty::try_from(0u64).unwrap() , #e_f : #edge_ty::from(0u64) } },
                        );
                        impl_new_for_self = Some(
                            quote! {#name {#d_f : #dest_ty::try_from(edge_dest).unwrap(), #e_f : #edge_ty::from(edge_type) } },
                        );
                    }
                    (d_s, e_s, Some(d_r), None, _, Some(e_f)) => {
                        if let Some(d_s) = d_s {
                            quoted_mut_dest = Some(quote! { self #d_s; self });
                        } else {
                            quoted_mut_dest = Some(
                                quote! { self.#d_r = #dest_ty::try_from(new_edge_dest).unwrap(); self },
                            );
                        }

                        if let Some(e_s) = e_s {
                            quoted_mut_edge_type = Some(quote! { self #e_s; self });
                        } else {
                            quoted_mut_edge_type =
                                Some(quote! { self.#e_f = #edge_ty::from(new_edge_type); self });
                        }
                        impl_default_for_self = Some(
                            quote! {#name {#d_r : #dest_ty::try_from(0u64).unwrap() , #e_f : #edge_ty::from(0u64) } },
                        );
                        impl_new_for_self = Some(
                            quote! {#name {#d_r : #dest_ty::try_from(edge_dest).unwrap(), #e_f : #edge_ty::from(edge_type) } },
                        );
                    }
                    // interpolation e_s
                    (d_s, e_s, None, Some(e_r), Some(d_f), _) => {
                        if let Some(d_s) = d_s {
                            quoted_mut_dest = Some(quote! { self #d_s; self });
                        } else {
                            quoted_mut_dest = Some(
                                quote! { self.#d_f = #dest_ty::try_from(new_edge_dest).unwrap(); self },
                            );
                        }

                        if let Some(e_s) = e_s {
                            quoted_mut_edge_type = Some(quote! { self #e_s; self });
                        } else {
                            quoted_mut_edge_type =
                                Some(quote! { self.#e_r = #edge_ty::from(new_edge_type); self });
                        }
                        impl_default_for_self = Some(
                            quote! {#name {#d_f : #dest_ty::try_from(0u64).unwrap() , #e_r : #edge_ty::from(0u64) } },
                        );
                        impl_new_for_self = Some(
                            quote! {#name {#d_f : #dest_ty::try_from(edge_dest).unwrap(), #e_r : #edge_ty::from(edge_type) } },
                        );
                    }
                    (_a, _b, _c, _d, _e, _f) => {
                        // println!("setter dest {:?}", _a);
                        // println!("setter type {:?}", _b);
                        // println!("real dest {:?}", _c);
                        // println!("real type {:?}", _d);
                        // println!("fallback dest {:?}", _e);
                        // println!("fallback type {:?}", _f);
                        return syn::Error::new_spanned(
                        name.clone(),
                            "Named struct must specify #[edge_type(getter=..., setter=..., real_type=...)], but setter was missing and no fallback initialization method was found"
                    )
                    .to_compile_error()
                    .into();
                    }
                };
            }

            // Tuple structs: Either attributes describing the methods are provided or assume (dest, edge_type)
            Fields::Unnamed(fields) => {
                // bitfield mode: Look for struct-level attributes
                if fields.unnamed.len() == 1 {
                    let dest_attr = input
                        .attrs
                        .iter()
                        .find(|a| a.path().is_ident("edge_dest"))
                        .expect(
                            "Bitfield tuple struct must specify #[edge_dest(getter=..., setter=..., real_type=...)]",
                        );
                    let edge_attr = input
                        .attrs
                        .iter()
                        .find(|a| a.path().is_ident("edge_type"))
                        .expect(
                            "Bitfield tuple struct must specify #[edge_type(getter=..., setter=..., real_type=...)]",
                        );

                    let (dest_getter, d_setter, dest_ty) = parse_generic_edge_attr(dest_attr);
                    let (edge_getter, e_setter, edge_ty) = parse_generic_edge_attr(edge_attr);

                    if dest_getter.is_none() || edge_getter.is_none() {
                        panic!(
                            "Bitfield tuple struct must specify #[edge_type(getter=..., setter=..., real_type=...)], getter was missing"
                        );
                    }
                    if d_setter.is_none() || e_setter.is_none() {
                        panic!(
                            "Bitfield tuple struct must specify #[edge_type(getter=..., setter=..., real_type=...)], setter was missing"
                        );
                    }
                    if dest_ty.is_none() || edge_ty.is_none() {
                        panic!(
                            "Bitfield tuple struct must specify #[edge_type(getter=..., setter=..., real_type=...)], real_type was missing"
                        );
                    }

                    if let Some(d_s) = d_setter {
                        quoted_mut_dest = Some(quote! { self.#d_s( new_edge_dest ); self });
                    } else {
                        panic!(
                            "Bitfield tuple struct must specify #[edge_dest(getter=..., setter=..., real_type=...)], setter was missing"
                        );
                    }
                    if let Some(e_s) = e_setter {
                        quoted_mut_edge_type = Some(quote! { self.#e_s( new_edge_type ); self });
                    } else {
                        panic!(
                            "Bitfield tuple struct must specify #[edge_type(getter=..., setter=..., real_type=...)], setter was missing"
                        );
                    }
                    quoted_dest_access = dest_getter.map(|e| quote! { self.#e() });
                    quoted_edge_type_access = edge_getter.map(|e| quote! { self.#e() });
                    quoted_dest_type = dest_ty.map(|ty| quote! {#ty});
                    quoted_edge_type = edge_ty.map(|ty| quote! {#ty});
                    impl_default_for_self = Some(quote! { #name::new(0u64, 0u64)});
                    impl_new_for_self = Some(quote! { #name::new(edge_type, edge_dest)});
                } else if fields.unnamed.len() == 2 {
                    quoted_dest_access = Some(quote! { self.0 });
                    quoted_edge_type_access = Some(quote! { self.0 });
                    let type_of_dest = fields.unnamed[0].ty.clone();
                    let type_of_edge_type = fields.unnamed[1].ty.clone();
                    quoted_dest_type = Some(type_of_dest.clone()).map(|ty| quote! {#ty});
                    quoted_edge_type = Some(type_of_edge_type.clone()).map(|ty| quote! {#ty});
                    quoted_mut_dest = Some(
                        quote! { self.0 = #type_of_dest::try_from(new_edge_dest).unwrap(); self },
                    );
                    quoted_mut_edge_type =
                        Some(quote! { self.1 = #type_of_edge_type::from(new_edge_type); self });
                    // dest node is a primitive field edge type not (necessarily or not always?)
                    impl_default_for_self = Some(
                        quote! { (#type_of_dest::try_from(0u64).unwrap(), #type_of_edge_type::from(0u64))},
                    );
                    impl_new_for_self = Some(
                        quote! { (#type_of_dest::try_from(edge_dest).unwrap(), #type_of_edge_type::from(edge_type)) },
                    );
                } else {
                    return syn::Error::new_spanned(
                        name.clone(),
                        "Unnamed structs are only supported for GenericEdge with 1 <= len() <= 2",
                    )
                    .to_compile_error()
                    .into();
                }
            }

            // ❌ Unit structs not supported
            Fields::Unit => {
                return syn::Error::new_spanned(
                    name.clone(),
                    "Unit structs are not supported for GenericEdge",
                )
                .to_compile_error()
                .into();
            }
        }
    }

    // ensure accesses were found if not search for fallbacks if not error out
    let quoted_dest_access = quoted_dest_access.unwrap_or_else(|| {
        syn::Error::new_spanned(name.clone(), "Missing #[edge_dest] attribute".to_string())
            .to_compile_error()
    });

    let quoted_edge_type_access = quoted_edge_type_access.unwrap_or_else(|| {
        syn::Error::new_spanned(name.clone(), "Missing #[edge_type] attribute".to_string())
            .to_compile_error()
    });

    let impl_new_for_self = impl_new_for_self.unwrap_or_else(|| {
        syn::Error::new_spanned(name.clone(), "Couldn't build constructor for type")
            .to_compile_error()
    });

    let impl_default_for_self = impl_default_for_self.unwrap_or_else(|| {
        syn::Error::new_spanned(name.clone(), "Couldn't build default for type").to_compile_error()
    });

    let static_type_assertions = quote! {
        const _: () = {
            struct _AssertDestTraits<
                T:
                    std::marker::Copy +
                    std::clone::Clone +
                    std::convert::TryFrom<u64> +
                    std::convert::TryFrom<usize> +
                    std::default::Default +
                    std::cmp::PartialEq +
                    std::cmp::Eq +
                    std::fmt::Debug +
                    std::fmt::Display +
                    std::cmp::PartialOrd +
                    std::cmp::Ord +
                    bytemuck::Pod +
                    bytemuck::Zeroable
                >(::core::marker::PhantomData<T>);
            struct _AssertEdgeTraits<T: GenericEdgeType>(::core::marker::PhantomData<T>);
            let _ = _AssertDestTraits::<#quoted_dest_type>(::core::marker::PhantomData);
            let _ = _AssertEdgeTraits::<#quoted_edge_type>(::core::marker::PhantomData);
        };
    };

    let expanded = quote! {

        #static_type_assertions

        impl std::marker::Copy for #name {}

        impl std::clone::Clone for #name {
            fn clone(&self) -> Self {
                *self
            }
        }

        unsafe impl bytemuck::Zeroable for #name {}

        unsafe impl bytemuck::Pod for #name {}

        impl GenericEdge<#quoted_edge_type> for #name {
            #[inline] fn new(edge_dest: u64, edge_type: u64) -> Self { #impl_new_for_self }
            #[inline] fn set_edge_dest(&mut self, new_edge_dest: u64) -> &mut Self { #quoted_mut_dest }
            #[inline] fn set_edge_type(&mut self, new_edge_type: u64) -> &mut Self { #quoted_mut_edge_type }
            #[inline] fn dest(&self) -> usize { #quoted_dest_access as usize }
            #[inline] fn e_type(&self) -> #quoted_edge_type { #quoted_edge_type::from(#quoted_edge_type_access) }
        }

        /// WARNING! Returns zeroed out `GenericEdgeType`. Beware if using out of box.
        impl std::default::Default for #name {
            fn default() -> Self {
                #impl_default_for_self
            }
        }

        impl std::cmp::PartialOrd for #name {
            fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                Some(self.dest().cmp(&other.dest()))
            }
        }

        impl std::cmp::Ord for #name {
            fn cmp(&self, other: &Self) -> std::cmp::Ordering {
                self.dest().cmp(&other.dest())
            }
        }
        impl std::cmp::PartialEq for #name {
            fn eq(&self, other: &Self) -> bool {
                self.dest() == other.dest() && self.e_type() == other.e_type()
            }
        }

        impl std::cmp::Eq for #name {}

        impl std::fmt::Debug for #name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "Edge {{\n\tdest: {:?}\n\ttype: {:?}\n}}", self.dest() , self.e_type())
            }
        }

        impl std::fmt::Display for #name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "Edge {{\n\tdest: {}\n\ttype: {}\n}}", self.dest() , self.e_type())
            }
        }

    };

    // let dump_file = "/tmp/macro_damp.rs";
    // match std::fs::write(dump_file, expanded.to_string()) {
    //     Ok(_) => {}
    //     Err(e) => panic!("error dumpoing macro onto {}: {}", dump_file, e),
    // };

    TokenStream::from(expanded)
}

fn parse_generic_edge_attr(
    attr: &Attribute,
) -> (Option<syn::Ident>, Option<syn::Ident>, Option<Type>) {
    let mut getter = None;
    let mut setter = None;
    let mut ty = None;
    if let Meta::List(meta_list) = &attr.meta {
        // Parse as punctuated Meta, e.g. #[edge_type(getter = "edge_type", real_type = "EdgeType")]
        let parsed: syn::punctuated::Punctuated<Meta, syn::Token![,]> =
            match meta_list.parse_args_with(syn::punctuated::Punctuated::parse_terminated) {
                Ok(i) => i,
                Err(e) => panic!("error parsinf token metadata: {}", e),
            };

        for meta in parsed {
            if let Meta::NameValue(nv) = meta {
                if nv.path.is_ident("getter") {
                    if let syn::Expr::Lit(expr_lit) = nv.value {
                        if let syn::Lit::Str(lit_str) = expr_lit.lit {
                            getter = Some(syn::Ident::new(&lit_str.value(), lit_str.span()));
                        }
                    }
                } else if nv.path.is_ident("t") || nv.path.is_ident("real_type") {
                    if let syn::Expr::Lit(expr_lit) = nv.value {
                        if let syn::Lit::Str(lit_str) = expr_lit.lit {
                            ty = syn::parse_str::<Type>(&lit_str.value()).ok();
                        }
                    }
                } else if nv.path.is_ident("setter") {
                    if let syn::Expr::Lit(expr_lit) = nv.value {
                        if let syn::Lit::Str(lit_str) = expr_lit.lit {
                            setter = Some(syn::Ident::new(&lit_str.value(), lit_str.span()));
                        }
                    }
                }
            }
        }
    }

    (getter, setter, ty)
}

#[proc_macro_derive(GenericEdgeType)]
pub fn derive_generic_edge_type(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = input.ident.clone();

    // check if #[repr(C)] is present
    let has_repr_c = input.attrs.iter().any(|attr: &Attribute| {
        if attr.path().is_ident("repr") {
            if let Meta::List(MetaList { tokens, .. }) = &attr.meta {
                tokens.to_string().contains("C")
            } else {
                false
            }
        } else {
            false
        }
    });

    if !has_repr_c {
        return syn::Error::new_spanned(
            input.ident,
            "Types deriving `GenericEdgeType` must be annotated with #[repr(C)]",
        )
        .to_compile_error()
        .into();
    }

    let static_type_assertions = quote! {
        const _: () = {
            struct _AssertEdgeTypeTraits<
                T:
                    std::marker::Copy +
                    std::clone::Clone +
                    std::convert::From<u64> +
                    std::convert::From<usize> +
                    std::default::Default +
                    std::cmp::PartialEq +
                    std::cmp::Eq +
                    std::fmt::Debug +
                    std::fmt::Display +
                    bytemuck::Pod +
                    bytemuck::Zeroable
            >(::core::marker::PhantomData<T>);
            let _ = _AssertEdgeTypeTraits::<#name>(::core::marker::PhantomData);
        };
    };

    let expanded = quote! {

        #static_type_assertions

        impl std::marker::Copy for #name {}

        impl std::clone::Clone for #name {
            fn clone(&self) -> Self {
                *self
            }
        }

        unsafe impl bytemuck::Zeroable for #name {}

        unsafe impl bytemuck::Pod for #name {}

        impl GenericEdgeType for #name {
            fn label(&self) -> usize {
                usize::from(*self)
            }
        }

        /// WARNING! Returns zeroed out `GenericEdgeType`. Beware if using out of box.
        impl std::default::Default for #name {
            fn default() -> Self {
                Self::from(0usize)
            }
        }

        impl std::fmt::Debug for #name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "EdgeType {{ label: {:?}}}", self.label())
            }
        }

        impl std::fmt::Display for #name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "EdgeType {{ label: {}}}", self.label())
            }
        }

        impl std::cmp::PartialEq for #name{
            fn eq(&self, other: &Self) -> bool {
                self.label() == other.label()
            }
        }

        impl std::cmp::Eq for #name{}
    };

    TokenStream::from(expanded)
}

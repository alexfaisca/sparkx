use core::panic;

use proc_macro::TokenStream;
use quote::quote;
use syn::{Attribute, Data, DeriveInput, Fields, Meta, MetaList, Type, parse_macro_input};

#[proc_macro_derive(GenericEdge, attributes(edge_dest, edge_type))]
pub fn derive_generic_edge(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
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
            "Types deriving `GenericEdge` must be annotated with #[repr(C)]",
        )
        .to_compile_error()
        .into();
    }

    let _data = if let Data::Struct(ds) = input.clone().data {
        ds
    } else {
        panic!("GenericEdge can only be derived on structs");
    };

    // accesses field-based or method-based
    let mut dest_access = None;
    let mut edge_type_access = None;
    // types
    let mut type_of_dest: Option<Type> = None;
    let mut type_of_edge_type: Option<Type> = None;
    // let mut field_vec = vec![];

    // --- Extract info from struct ---
    if let Data::Struct(s) = &input.data {
        match &s.fields {
            // ✅ Named fields (normal struct or bitfield struct with visible fields)
            Fields::Named(fields) => {
                for field in &fields.named {
                    let ident = field.ident.as_ref();

                    // Look for #[edge_dest] or #[edge_type] attributes
                    for attr in &field.attrs {
                        if attr.path().is_ident("edge_dest") {
                            let (getter, ty) = parse_generic_edge_attr(attr);
                            dest_access = Some(getter.map_or_else(
                                || {
                                    let ident = match ident {
                                        Some(i) => i,
                                        None => panic!("GenericEdge {} is annotated, but no getter was provided and no substitute field was found", "edge_dest")
                                    };
                                    quote! {
                                        self.#ident
                                    }
                                },
                                |g| quote! { self.#g() },
                            ));
                            type_of_dest = if ty.is_some() {
                                ty
                            } else {
                                Some(field.ty.clone())
                            };
                        }
                        if attr.path().is_ident("edge_type") {
                            let (getter, ty) = parse_generic_edge_attr(attr);
                            edge_type_access = Some(getter.map_or_else(
                                || {
                                    let ident = match ident {
                                        Some(i) => i,
                                        None => panic!("GenericEdge {} is annotated, but no getter was provided and no substitute field was found", "edge_type")
                                    };
                                    quote! {
                                        self.#ident
                                    }
                                },
                                |g| quote! { self.#g() },
                            ));
                            type_of_edge_type = if ty.is_some() {
                                ty
                            } else {
                                Some(field.ty.clone())
                            };
                        }
                    }
                }
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
                            "Bitfield tuple struct must specify #[edge_dest(getter=..., real_type=...)]",
                        );
                    let edge_attr = input
                        .attrs
                        .iter()
                        .find(|a| a.path().is_ident("edge_type"))
                        .expect(
                            "Bitfield tuple struct must specify #[edge_type(getter=..., real_type=...)]",
                        );

                    let (dest_getter, dest_ty) = parse_generic_edge_attr(dest_attr);
                    let (edge_getter, edge_ty) = parse_generic_edge_attr(edge_attr);

                    if dest_getter.is_none() || edge_getter.is_none() {
                        panic!(
                            "Bitfield tuple struct must specify #[edge_type(getter=..., real_type=...)], getter was missing"
                        );
                    }
                    if dest_ty.is_none() || edge_ty.is_none() {
                        panic!(
                            "Bitfield tuple struct must specify #[edge_type(getter=..., real_type=...)], real_type was missing"
                        );
                    }

                    dest_access = Some(quote! { self.#dest_getter() });
                    edge_type_access = Some(quote! { self.#edge_getter() });
                    type_of_dest = dest_ty;
                    type_of_edge_type = edge_ty;
                } else if fields.unnamed.len() == 2 {
                    dest_access = Some(quote! { self.0 });
                    type_of_dest = Some(fields.unnamed[0].ty.clone());
                    edge_type_access = Some(quote! { self.1 });
                    type_of_edge_type = Some(fields.unnamed[1].ty.clone());
                } else {
                    return syn::Error::new_spanned(
                        name,
                        "Unnamed structs are only supported for GenericEdge with len() == 2",
                    )
                    .to_compile_error()
                    .into();
                }
            }

            // ❌ Unit structs not supported
            Fields::Unit => {
                return syn::Error::new_spanned(
                    name,
                    "Unit structs are not supported for GenericEdge",
                )
                .to_compile_error()
                .into();
            }
        }
    }

    // Ensure we found accesses or error out
    let dest_access = dest_access.unwrap_or_else(|| {
        syn::Error::new_spanned(name.clone(), "Missing #[edge_dest] or field").to_compile_error()
    });
    let edge_type_access = edge_type_access.unwrap_or_else(|| {
        syn::Error::new_spanned(name.clone(), "Missing #[edge_type] or field").to_compile_error()
    });

    let quoted_dest_type = type_of_dest.map(|t| quote! { #t });
    let quoted_edge_type = type_of_edge_type.map(|t| quote! { #t });

    let static_type_assertions = quote! {
        const _: () = {
            struct _AssertDestTraits<T: Copy + Clone + PartialEq + Eq + std::fmt::Debug + std::fmt::Display + PartialOrd + Ord + bytemuck::Pod + bytemuck::Zeroable>(::core::marker::PhantomData<T>);
            struct _AssertEdgeTraits<T: GenericEdgeType>(::core::marker::PhantomData<T>);
            let _ = _AssertDestTraits::<#quoted_dest_type>(::core::marker::PhantomData);
            let _ = _AssertEdgeTraits::<#quoted_edge_type>(::core::marker::PhantomData);
        };
    };

    let expanded = quote! {

        #static_type_assertions

        impl Copy for #name {}

        impl Clone for #name {
            fn clone(&self) -> Self {
                *self
            }
        }

        unsafe impl bytemuck::Zeroable for #name {}

        unsafe impl bytemuck::Pod for #name {}

        impl GenericEdge<#quoted_edge_type> for #name {
            #[inline] fn new(edge_type: u64, edge_dest: u64) -> Self { #name::new(edge_type, edge_dest) }
            #[inline] fn dest(&self) -> usize { #dest_access as usize }
            #[inline] fn e_type(&self) -> #quoted_edge_type { #quoted_edge_type::from(#edge_type_access) }
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
        impl PartialEq for #name {
            fn eq(&self, other: &Self) -> bool {
                self.dest() == other.dest() && self.e_type() == other.e_type()
            }
        }

        impl Eq for #name {}

        impl std::fmt::Debug for #name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "Edge {{\n\tdest: {:?}\n\ttype: {:?}\n}}", self.dest(), self.e_type())
            }
        }

        impl std::fmt::Display for #name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "Edge {{\n\tdest: {}\n\ttype: {}\n}}", self.dest(), self.e_type())
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

fn parse_generic_edge_attr(attr: &Attribute) -> (Option<syn::Ident>, Option<Type>) {
    let mut getter = None;
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
                }
            }
        }
    }

    (getter, ty)
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
            "Types deriving `GenericEdge` must be annotated with #[repr(C)]",
        )
        .to_compile_error()
        .into();
    }

    let static_type_assertions = quote! {
        const _: () = {
            struct _AssertEdgeTypeTraits<T: Copy + Clone + PartialEq + Eq + std::fmt::Debug + std::fmt::Display + bytemuck::Pod + bytemuck::Zeroable + std::convert::From<u64> + std::convert::From<usize>>(::core::marker::PhantomData<T>);
            let _ = _AssertEdgeTypeTraits::<#name>(::core::marker::PhantomData);
        };
    };

    let expanded = quote! {

        #static_type_assertions

        impl Copy for #name {}

        impl Clone for #name {
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

        impl std::fmt::Debug for #name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "EdgeType {{{:?}}}", self)
            }
        }

        impl std::fmt::Display for #name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "EdgeType {{{}}}", self)
            }
        }

        impl PartialEq for #name{
            fn eq(&self, other: &Self) -> bool {
                self.label() == other.label()
            }
        }

        impl Eq for #name{

        }
    };

    // let dump_file = "/tmp/macro_type_damp.rs";
    // match std::fs::write(dump_file, expanded.to_string()) {
    //     Ok(_) => {}
    //     Err(e) => panic!("error dumpoing macro onto {}: {}", dump_file, e),
    // };

    TokenStream::from(expanded)
}

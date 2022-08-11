extern crate bindgen;

use std::env;
use std::path::PathBuf;

fn env_var_rerun(name: &str) -> Result<String, env::VarError> {
    println!("cargo:rerun-if-env-changed={}", name);
    env::var(name)
}

fn main() {
    // Tell cargo to look for shared libraries in the specified directory
    // println!("cargo:rustc-link-search=/path/to/lib");

    // Tell cargo to tell rustc to link the system bzip2
    // shared library.
    // println!("cargo:rustc-link-lib=cuda_runtime");

    let cuda_home = env_var_rerun("CUDA_HOME").unwrap_or_else(|_| "/usr/local/cuda/".to_string());
    let nccl_home = env_var_rerun("NCCL_HOME").unwrap_or_else(|_| "/usr/local/nccl/".to_string());
    println!("cargo:rustc-link-search=native={cuda_home}lib64/");
    println!("cargo:rustc-link-search=native={nccl_home}lib");
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=nccl");

    // Tell cargo to invalidate the built crate whenever the wrapper changes
    println!("cargo:rerun-if-changed=wrapper.h");

    // The bindgen::Builder is the main entry point
    // to bindgen, and lets you build up options for
    // the resulting bindings.
    let bindings = bindgen::Builder::default()
        // The input header we would like to generate
        // bindings for.
        .header("wrapper.h")
        .clang_arg(&format!("-I{cuda_home}include"))
        .clang_arg(&format!("-I{nccl_home}include"))
        // Tell cargo to invalidate the built crate whenever any of the
        // included header files changed.
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        // Finish the builder and generate the bindings.
        .generate()
        // Unwrap the Result and panic on failure.
        .expect("Unable to generate bindings");

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}

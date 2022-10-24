extern crate bindgen;

use std::env;
use std::path::{Path, PathBuf};

fn env_var_rerun(name: &str) -> Result<String, env::VarError> {
    println!("cargo:rerun-if-env-changed={}", name);
    env::var(name)
}

fn main() {
    let cuda_home = find_cuda_root()
                .expect("Failed to find CUDA ROOT, make sure the CUDA SDK is installed and CUDA_PATH or CUDA_ROOT are set!");
    // Tell cargo to look for shared libraries in the specified directory
    println!("cargo:rustc-link-search={}/lib/", cuda_home.display());

    // Tell cargo to tell rustc to link the system bzip2
    // shared library.
    println!("cargo:rustc-link-lib=cuda");

    // Tell cargo to invalidate the built crate whenever the wrapper changes
    println!("cargo:rerun-if-changed=src/wrapper.h");

    // The bindgen::Builder is the main entry point
    // to bindgen, and lets you build up options for
    // the resulting bindings.
    let bindings = bindgen::Builder::default()
        // The input header we would like to generate
        // bindings for.
        .header("src/wrapper.h")
        // Tell cargo to invalidate the built crate whenever any of the
        // included header files changed.
        .clang_arg(format!("-I{}/include/", cuda_home.display()))
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

pub fn find_cuda_root() -> Option<PathBuf> {
    // search through the common environment variables first
    for path in ["CUDA_PATH", "CUDA_ROOT", "CUDA_TOOLKIT_ROOT_DIR"]
        .iter()
        .filter_map(|name| std::env::var(*name).ok())
    {
        if is_cuda_root_path(&path) {
            return Some(path.into());
        }
    }

    // If it wasn't specified by env var, try the default installation paths
    #[cfg(not(target_os = "windows"))]
    let default_paths = ["/usr/local/cuda", "/opt/cuda"];
    #[cfg(target_os = "windows")]
    let default_paths = ["C:/CUDA"]; // TODO (AL): what's the actual path here?

    for path in default_paths {
        if is_cuda_root_path(path) {
            return Some(path.into());
        }
    }

    None
}

fn is_cuda_root_path<P: AsRef<Path>>(path: P) -> bool {
    path.as_ref().join("include").join("cuda.h").is_file()
}

pub fn read_env() -> Vec<PathBuf> {
    if let Ok(path) = env_var_rerun("CUDA_LIBRARY_PATH") {
        // The location of the libcuda, libcudart, and libcublas can be hardcoded with the
        // CUDA_LIBRARY_PATH environment variable.
        let split_char = if cfg!(target_os = "windows") {
            ";"
        } else {
            ":"
        };
        path.split(split_char).map(PathBuf::from).collect()
    } else {
        vec![]
    }
}

// SPDX-License-Identifier: AGPL-3.0-only

use std::ffi::OsString;
use std::fs;
use std::path::{Path, PathBuf};

pub(crate) const POLICY_ENV: &[&str] = &[
    "UNSLOTH_AMD_ONLY",
    "UNSLOTH_NVIDIA_LIBRARY_PROBE",
    "UNSLOTH_DISABLE_XPU_AUTO",
    "UNSLOTH_LLAMA_CPP_BACKEND",
    "UV_TORCH_BACKEND",
    "VK_DRIVER_FILES",
    "VK_ICD_FILENAMES",
];

fn is_amd_icd(path: &Path) -> bool {
    if path.extension().is_none_or(|ext| ext != "json") || !path.is_file() {
        return false;
    }
    let Some(stem) = path.file_stem().and_then(|stem| stem.to_str()) else {
        return false;
    };
    let stem = stem.to_ascii_lowercase();
    if stem.ends_with("32") || stem.contains("i686") || stem.contains("i386") {
        return false;
    }
    let Some(manifest) = fs::read(path)
        .ok()
        .and_then(|data| serde_json::from_slice::<serde_json::Value>(&data).ok())
    else {
        return false;
    };
    manifest["ICD"]["library_path"]
        .as_str()
        .and_then(|library| Path::new(library).file_name())
        .and_then(|name| name.to_str())
        .is_some_and(|name| {
            let name = name.to_ascii_lowercase();
            [
                "radeon",
                "radv",
                "amdvlk",
                "amd_icd",
                "amd_pro",
                "amd_vulkan",
            ]
            .iter()
            .any(|marker| name.contains(marker))
        })
}

fn configured_paths(var: &str) -> Vec<PathBuf> {
    std::env::var_os(var)
        .map(|paths| {
            std::env::split_paths(&paths)
                .filter(|p| p.is_absolute())
                .collect()
        })
        .unwrap_or_default()
}

fn default_icd_dirs(home: &Path) -> Vec<PathBuf> {
    let config_home = std::env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .filter(|path| path.is_absolute())
        .unwrap_or_else(|| home.join(".config"));
    let data_home = std::env::var_os("XDG_DATA_HOME")
        .map(PathBuf::from)
        .filter(|path| path.is_absolute())
        .unwrap_or_else(|| home.join(".local/share"));
    let mut roots = vec![config_home, data_home];
    roots.extend(configured_paths("XDG_CONFIG_DIRS"));
    roots.extend(configured_paths("XDG_DATA_DIRS"));
    roots.extend(
        [
            "/etc/xdg",
            "/etc",
            "/usr/local/etc",
            "/usr/local/share",
            "/usr/share",
        ]
        .map(PathBuf::from),
    );
    roots
        .into_iter()
        .map(|root| root.join("vulkan/icd.d"))
        .collect()
}

fn amd_icds(paths: impl IntoIterator<Item = PathBuf>) -> Vec<PathBuf> {
    let mut manifests = Vec::new();
    for path in paths {
        let candidates = if path.is_dir() {
            let mut entries: Vec<_> = fs::read_dir(path)
                .into_iter()
                .flatten()
                .filter_map(Result::ok)
                .map(|entry| entry.path())
                .collect();
            entries.sort();
            entries
        } else {
            vec![path]
        };
        for candidate in candidates {
            if is_amd_icd(&candidate) && !manifests.contains(&candidate) {
                manifests.push(candidate);
            }
        }
    }
    manifests
}

pub(crate) fn configure(home: &Path, studio: &Path) {
    std::env::set_var("UNSLOTH_AMD_ONLY", "1");
    std::env::set_var("UNSLOTH_NVIDIA_LIBRARY_PROBE", "0");
    std::env::set_var("UNSLOTH_DISABLE_XPU_AUTO", "1");
    // uv's automatic backend can select CUDA independently of the installer.
    std::env::remove_var("UV_TORCH_BACKEND");
    if std::env::var("UNSLOTH_LLAMA_CPP_BACKEND")
        .is_ok_and(|backend| backend.trim().eq_ignore_ascii_case("cuda"))
    {
        std::env::remove_var("UNSLOTH_LLAMA_CPP_BACKEND");
    }

    // The force lists work with both the current and the older Vulkan loader.
    let forced = configured_paths("VK_DRIVER_FILES");
    let forced = if forced.is_empty() {
        configured_paths("VK_ICD_FILENAMES")
    } else {
        forced
    };
    let mut manifests = amd_icds(forced);
    if manifests.is_empty() {
        let mut candidates = configured_paths("VK_ADD_DRIVER_FILES");
        candidates.extend(default_icd_dirs(home));
        manifests = amd_icds(candidates);
    }
    // A nonempty force list keeps the loader from searching other vendors when AMD is absent.
    let value = if manifests.is_empty() {
        studio
            .join("amd-vulkan-icd-unavailable.json")
            .into_os_string()
    } else {
        std::env::join_paths(manifests).unwrap_or_else(|_| OsString::from("/dev/null"))
    };
    std::env::set_var("VK_DRIVER_FILES", &value);
    std::env::set_var("VK_ICD_FILENAMES", value);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vulkan_icd_selection_keeps_only_amd_64_bit_drivers() {
        let dir = tempfile::tempdir().unwrap();
        for (filename, library) in [
            ("radeon_icd.x86_64.json", "libvulkan_radeon.so"),
            ("radeon_icd.i686.json", "libvulkan_radeon.so"),
            ("intel_icd.x86_64.json", "libvulkan_intel.so"),
            ("nvidia_icd.json", "libGLX_nvidia.so.0"),
            ("amd_icd64.json", "/opt/amd/lib/amdvlk64.so"),
        ] {
            fs::write(
                dir.path().join(filename),
                serde_json::json!({"ICD":{"library_path":library}}).to_string(),
            )
            .unwrap();
        }
        assert_eq!(
            amd_icds([dir.path().to_owned()]),
            ["amd_icd64.json", "radeon_icd.x86_64.json"].map(|name| dir.path().join(name))
        );
    }
}

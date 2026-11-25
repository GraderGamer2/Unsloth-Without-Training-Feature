// SPDX-License-Identifier: AGPL-3.0-only

//! Storage paths for the Linux desktop app and its managed backend and installer.

use std::ffi::OsString;
use std::fs::{self, DirBuilder};
use std::io;
use std::os::unix::fs::{DirBuilderExt, MetadataExt};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;
use tauri::utils::config::{AppDirectoriesOverride, AppDirectoryOverrides};
use tauri::Manager;

pub(crate) const FIXED_STORAGE_ENV: &str = "UNSLOTH_LINUX_DESKTOP_STORAGE";
static HOST_ENV: OnceLock<Vec<(&'static str, Option<OsString>)>> = OnceLock::new();
static BACKEND_WHEEL: OnceLock<PathBuf> = OnceLock::new();

pub(crate) fn load_bundled_backend(
    app: &tauri::AppHandle,
) -> Result<(), Box<dyn std::error::Error>> {
    if cfg!(debug_assertions) {
        return Ok(());
    }
    let dir = app
        .path()
        .resolve("backend-wheel", tauri::path::BaseDirectory::Resource)?;
    let mut wheels = Vec::new();
    for entry in fs::read_dir(&dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().is_some_and(|extension| extension == "whl")
            && entry.file_type()?.is_file()
        {
            wheels.push(path);
        }
    }
    let [wheel] = wheels.as_slice() else {
        return Err(io::Error::other(format!(
            "Expected one bundled backend wheel in {}",
            dir.display()
        ))
        .into());
    };
    let _ = BACKEND_WHEEL.set(wheel.clone());
    Ok(())
}

pub(crate) fn set_bundled_backend_environment(command: &mut Command) {
    if let Some(wheel) = BACKEND_WHEEL.get() {
        command.env("UNSLOTH_DESKTOP_BACKEND_WHEEL", wheel);
    } else {
        command.env_remove("UNSLOTH_DESKTOP_BACKEND_WHEEL");
    }
}

pub(crate) fn bundled_backend_version() -> Option<&'static str> {
    BACKEND_WHEEL
        .get()?
        .file_name()?
        .to_str()?
        .strip_prefix("unsloth-")?
        .split('-')
        .next()
}

pub(crate) fn configured() -> bool {
    std::env::var_os(FIXED_STORAGE_ENV).is_some_and(|value| value == "1")
}

pub(crate) fn desktop_dir(home: &Path) -> PathBuf {
    crate::studio_home::default_root_for_home(home).join("desktop")
}

pub(crate) fn downloads_dir() -> Option<PathBuf> {
    dirs::home_dir().map(|home| home.join(".unsloth").join("Downloads"))
}

pub(crate) fn restore_host_environment(command: &mut Command) {
    if let Some(environment) = HOST_ENV.get() {
        for (name, value) in environment {
            match value {
                Some(value) => {
                    command.env(name, value);
                }
                None => {
                    command.env_remove(name);
                }
            }
        }
    }
}

fn private_dir(path: &Path) -> io::Result<()> {
    DirBuilder::new().mode(0o700).recursive(true).create(path)?;
    let info = fs::symlink_metadata(path)?;
    if !info.is_dir() || info.uid() != unsafe { libc::geteuid() } || info.mode() & 0o077 != 0 {
        return Err(io::Error::new(
            io::ErrorKind::PermissionDenied,
            format!(
                "{} must be a private directory owned by this user",
                path.display()
            ),
        ));
    }
    Ok(())
}

fn copy_profile(source: &Path, target: &Path) -> io::Result<()> {
    for entry in fs::read_dir(source)? {
        let entry = entry?;
        let name = entry.file_name();
        if matches!(
            name.to_str(),
            Some(
                "WebKitCache"
                    | "CacheStorage"
                    | "serviceworkers"
                    | ".webview-cache-lock"
                    | ".webview-cache-cleared"
            )
        ) {
            continue;
        }
        // A legacy profile can contain links to the host filesystem. Copy its own data only.
        let kind = entry.file_type()?;
        let path = target.join(name);
        if kind.is_dir() {
            fs::create_dir(&path)?;
            copy_profile(&entry.path(), &path)?;
        } else if kind.is_file() {
            fs::copy(entry.path(), path)?;
        }
    }
    Ok(())
}

fn migrate_profile(source: &Path, target: &Path) -> io::Result<()> {
    if target.try_exists()? {
        return Ok(());
    }
    match fs::symlink_metadata(source) {
        Ok(info) if info.is_dir() => (),
        Ok(_) => return Ok(()),
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(error),
    }

    // Old releases hold this lock while their WebKit profile is open. Do not copy live SQLite files.
    let _legacy_lock = match fs::File::open(source.join(".webview-cache-lock")) {
        Ok(lock) => {
            lock.try_lock_shared().map_err(|error| {
                io::Error::other(format!(
                    "Close the previous Unsloth window before migrating {}: {error}",
                    source.display()
                ))
            })?;
            Some(lock)
        }
        Err(error) if error.kind() == io::ErrorKind::NotFound => None,
        Err(error) => return Err(error),
    };
    let staging = tempfile::tempdir_in(target.parent().expect("profile has a parent"))?;
    copy_profile(source, staging.path())?;
    match fs::rename(staging.path(), target) {
        Ok(()) => Ok(()),
        Err(error)
            if matches!(
                error.kind(),
                io::ErrorKind::AlreadyExists | io::ErrorKind::DirectoryNotEmpty
            ) =>
        {
            Ok(())
        }
        Err(error) => Err(error),
    }
}

pub(crate) fn configure<R: tauri::Runtime>(context: &mut tauri::Context<R>) -> io::Result<PathBuf> {
    let home = dirs::home_dir()
        .filter(|path| path.is_absolute())
        .ok_or_else(|| io::Error::other("Could not resolve the Unsloth home directory"))?;
    let root = home.join(".unsloth");
    let studio = crate::studio_home::default_root_for_home(&home);
    let desktop = desktop_dir(&home);
    let config = desktop.join("config");
    let data = desktop.join("data");
    let cache = studio.join("cache");
    let models = root.join("Models");
    let hub = models.join("hub");
    let hf_cache = cache.join("huggingface");
    let hf_home = studio.join("huggingface");
    let temp = studio.join("tmp");

    private_dir(&desktop)?;
    private_dir(&temp)?;
    private_dir(&hf_home)?;
    fs::create_dir_all(&hub)?;
    fs::create_dir_all(cache.join("xdg"))?;

    let identifier = &context.config().identifier;
    if let Some(old) = dirs::config_dir() {
        migrate_profile(&old.join(identifier), &config)?;
    }
    if let Some(old) = dirs::data_local_dir() {
        migrate_profile(&old.join(identifier), &data)?;
    }

    context.config_mut().app.app_directories_override =
        Some(AppDirectoriesOverride::Directories(AppDirectoryOverrides {
            config: Some(config.clone()),
            data: Some(data.clone()),
            local_data: Some(data),
            cache: Some(desktop.join("cache")),
            log: Some(studio.join("logs")),
        }));

    // Set before the GUI, Python and their dependencies read import- or process-time defaults.
    let paths = [
        ("XDG_CACHE_HOME", cache.join("xdg")),
        ("XDG_STATE_HOME", studio.join("state")),
        ("TMPDIR", temp.clone()),
        ("TMP", temp.clone()),
        ("TEMP", temp),
        ("HF_HOME", hf_home.clone()),
        ("HF_TOKEN_PATH", hf_home.join("token")),
        ("HF_HUB_CACHE", hub.clone()),
        ("HUGGINGFACE_HUB_CACHE", hub.clone()),
        ("TRANSFORMERS_CACHE", hub.clone()),
        ("SENTENCE_TRANSFORMERS_HOME", hub.clone()),
        ("LLAMA_CACHE", hub),
        ("HF_XET_CACHE", hf_cache.join("xet")),
        ("HF_ASSETS_CACHE", hf_cache.join("assets")),
        ("HF_DATASETS_CACHE", hf_cache.join("datasets")),
        ("HF_MODULES_CACHE", hf_cache.join("modules")),
        ("TORCH_HOME", models.join("torch")),
        ("PIP_CACHE_DIR", cache.join("pip")),
        ("UV_CACHE_DIR", cache.join("uv")),
        ("UV_PYTHON_INSTALL_DIR", studio.join("python")),
        ("UV_PYTHON_BIN_DIR", studio.join("bin")),
        ("UV_INSTALL_DIR", studio.join("uv")),
        ("UV_UNMANAGED_INSTALL", studio.join("uv")),
        ("NPM_CONFIG_CACHE", cache.join("npm")),
        ("TRITON_CACHE_DIR", cache.join("triton")),
        ("TRITON_DUMP_DIR", cache.join("triton-dump")),
        ("CUDA_CACHE_PATH", cache.join("cuda")),
        ("MESA_SHADER_CACHE_DIR", cache.join("mesa")),
        ("UNSLOTH_STUDIO_PROJECTS_HOME", root.join("Projects")),
    ];
    let _ = HOST_ENV.set(
        paths
            .iter()
            .map(|(name, _)| *name)
            .chain([
                FIXED_STORAGE_ENV,
                "UV_NO_MODIFY_PATH",
                "UNSLOTH_LEGACY_HF_HUB_CACHE",
            ])
            .map(|name| (name, std::env::var_os(name)))
            .collect(),
    );
    let legacy_hub = std::env::var_os("HF_HUB_CACHE")
        .or_else(|| std::env::var_os("HUGGINGFACE_HUB_CACHE"))
        .filter(|path| Path::new(path).is_absolute())
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var_os("HF_HOME")
                .filter(|path| Path::new(path).is_absolute())
                .map(|path| PathBuf::from(path).join("hub"))
        })
        .or_else(|| dirs::cache_dir().map(|path| path.join("huggingface").join("hub")));
    if let Some(path) = legacy_hub {
        std::env::set_var("UNSLOTH_LEGACY_HF_HUB_CACHE", path);
    }
    for (name, path) in paths {
        std::env::set_var(name, path);
    }
    std::env::set_var("UV_NO_MODIFY_PATH", "1");
    std::env::set_var(FIXED_STORAGE_ENV, "1");
    Ok(config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::symlink;

    #[test]
    fn legacy_profile_migration_keeps_user_state_without_copying_caches_or_links() {
        let home = tempfile::tempdir().unwrap();
        let old = home.path().join("old");
        let new = home.path().join("new");
        fs::create_dir_all(old.join("WebKitCache")).unwrap();
        fs::write(old.join("settings.json"), "old settings").unwrap();
        symlink(home.path(), old.join("link")).unwrap();

        migrate_profile(&old, &new).unwrap();
        assert_eq!(
            fs::read_to_string(new.join("settings.json")).unwrap(),
            "old settings"
        );
        assert!(!new.join("WebKitCache").exists());
        assert!(!new.join("link").exists());

        fs::write(new.join("settings.json"), "new settings").unwrap();
        migrate_profile(&old, &new).unwrap();
        assert_eq!(
            fs::read_to_string(new.join("settings.json")).unwrap(),
            "new settings"
        );
        assert_eq!(
            fs::read_to_string(old.join("settings.json")).unwrap(),
            "old settings"
        );
    }
}

use std::fs::{File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::error::{Error, Result};

static NEXT_TEMPORARY: AtomicU64 = AtomicU64::new(0);

/// Reject destinations that cannot safely participate in a replace operation.
///
/// Symbolic links are deliberately rejected. Replacing a link rather than its
/// referent varies between platforms and is too surprising for a destructive
/// CLI output. Existing regular files, including hard links, are safe: the
/// transaction replaces only the requested directory entry.
pub(crate) fn prevalidate_output_path(path: &Path) -> Result<()> {
    if path.file_name().is_none() {
        return Err(Error::InvalidArgument(format!(
            "output path '{}' has no file name",
            path.display()
        )));
    }
    let parent = output_parent(path);
    let metadata = std::fs::metadata(parent)?;
    if !metadata.is_dir() {
        return Err(Error::InvalidArgument(format!(
            "output parent '{}' is not a directory",
            parent.display()
        )));
    }
    match std::fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_symlink() => Err(Error::InvalidArgument(format!(
            "output path '{}' must not be a symbolic link",
            path.display()
        ))),
        Ok(metadata) if !metadata.is_file() => Err(Error::InvalidArgument(format!(
            "output path '{}' is not a regular file",
            path.display()
        ))),
        Ok(_) => Ok(()),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error.into()),
    }
}

/// Persist one output without exposing partially-written contents.
pub(crate) fn atomic_write(path: &Path, contents: &[u8]) -> Result<()> {
    let staged = StagedOutput::from_writer(path, |file| {
        file.write_all(contents)?;
        Ok(())
    })?;
    OutputTransaction::new(vec![staged]).commit()
}

/// Persist one output produced by a streaming serializer.
pub(crate) fn atomic_write_with(
    path: &Path,
    write: impl FnOnce(&mut File) -> Result<()>,
) -> Result<()> {
    let staged = StagedOutput::from_writer(path, write)?;
    OutputTransaction::new(vec![staged]).commit()
}

/// Persist a related set of outputs as one recoverable transaction.
///
/// Every file is completely staged and synced before any destination changes.
/// Existing destinations are backed up before installation. Any ordinary I/O
/// failure, and any unwinding panic, restores every destination to its original
/// state instead of leaving a half-committed pair.
pub(crate) fn atomic_write_many(outputs: &[(&Path, &[u8])]) -> Result<()> {
    if outputs.is_empty() {
        return Ok(());
    }
    ensure_distinct_targets(outputs.iter().map(|(path, _)| *path))?;
    let mut staged = Vec::with_capacity(outputs.len());
    for (path, contents) in outputs {
        staged.push(StagedOutput::from_writer(path, |file| {
            file.write_all(contents)?;
            Ok(())
        })?);
    }
    OutputTransaction::new(staged).commit()
}

fn ensure_distinct_targets<'a>(paths: impl Iterator<Item = &'a Path>) -> Result<()> {
    let paths = paths.collect::<Vec<_>>();
    for left in 0..paths.len() {
        for right in left + 1..paths.len() {
            let same_existing = paths[left].exists()
                && paths[right].exists()
                && same_file::is_same_file(paths[left], paths[right])?;
            if same_existing || normalized_path(paths[left])? == normalized_path(paths[right])? {
                return Err(Error::InvalidArgument(
                    "transaction output paths must be distinct".into(),
                ));
            }
        }
    }
    Ok(())
}

fn normalized_path(path: &Path) -> Result<PathBuf> {
    if path.exists() {
        return Ok(std::fs::canonicalize(path)?);
    }
    let parent = std::fs::canonicalize(output_parent(path))?;
    let file_name = path.file_name().ok_or_else(|| {
        Error::InvalidArgument(format!("path '{}' has no file name", path.display()))
    })?;
    let joined = parent.join(file_name);
    #[cfg(windows)]
    {
        Ok(PathBuf::from(joined.to_string_lossy().to_lowercase()))
    }
    #[cfg(not(windows))]
    {
        Ok(joined)
    }
}

struct StagedOutput {
    target: PathBuf,
    temporary: Option<PathBuf>,
    file: Option<File>,
    backup: Option<PathBuf>,
    installed: bool,
}

impl StagedOutput {
    fn from_writer(path: &Path, write: impl FnOnce(&mut File) -> Result<()>) -> Result<Self> {
        prevalidate_output_path(path)?;
        let (temporary, file) = create_unique(path, "stage")?;
        let mut staged = Self {
            target: path.to_path_buf(),
            temporary: Some(temporary),
            file: Some(file),
            backup: None,
            installed: false,
        };
        let file = staged
            .file
            .as_mut()
            .expect("a newly staged output owns its file handle");
        write(file)?;
        file.flush()?;
        file.sync_all()?;
        staged.file.take();
        // Re-check after serialization so a destination changed concurrently to
        // a directory or symlink is never replaced.
        prevalidate_output_path(path)?;
        Ok(staged)
    }

    fn prepare_backup(&mut self) -> Result<()> {
        prevalidate_output_path(&self.target)?;
        if !self.target.exists() {
            return Ok(());
        }
        let backup = unique_path(&self.target, "backup")?;
        match std::fs::hard_link(&self.target, &backup) {
            Ok(()) => {}
            Err(hard_link_error) => {
                let mut source = File::open(&self.target)?;
                let mut destination = match OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(&backup)
                {
                    Ok(file) => file,
                    Err(error) => {
                        return Err(io::Error::new(
                            error.kind(),
                            format!(
                                "could not back up '{}': hard-link failed ({hard_link_error}); copy failed ({error})",
                                self.target.display()
                            ),
                        )
                        .into());
                    }
                };
                if let Err(error) = io::copy(&mut source, &mut destination)
                    .and_then(|_| destination.flush())
                    .and_then(|()| destination.sync_all())
                {
                    drop(destination);
                    let _ = std::fs::remove_file(&backup);
                    return Err(error.into());
                }
            }
        }
        self.backup = Some(backup);
        Ok(())
    }

    fn install(&mut self) -> Result<()> {
        let temporary = self
            .temporary
            .as_ref()
            .expect("a staged output has a temporary file until installation");
        std::fs::rename(temporary, &self.target)?;
        self.temporary = None;
        self.installed = true;
        Ok(())
    }

    fn rollback(&mut self) -> io::Result<()> {
        // Close the staging handle before unlinking it. This order matters on
        // Windows when a serializer panics while the handle is still open.
        self.file.take();
        let mut first_error = None;
        if self.installed {
            if let Err(error) = remove_if_exists(&self.target) {
                first_error = Some(error);
            }
            if let Some(backup) = self.backup.take() {
                if let Err(error) = std::fs::rename(&backup, &self.target) {
                    first_error.get_or_insert(error);
                }
            }
            self.installed = false;
        } else if let Some(backup) = self.backup.take() {
            if let Err(error) = remove_if_exists(&backup) {
                first_error.get_or_insert(error);
            }
        }
        if let Some(temporary) = self.temporary.take() {
            if let Err(error) = remove_if_exists(&temporary) {
                first_error.get_or_insert(error);
            }
        }
        first_error.map_or(Ok(()), Err)
    }

    fn finish(&mut self) {
        if let Some(backup) = self.backup.take() {
            let _ = remove_if_exists(&backup);
        }
        self.temporary = None;
        self.file = None;
        self.installed = false;
    }
}

impl Drop for StagedOutput {
    fn drop(&mut self) {
        let _ = self.rollback();
    }
}

struct OutputTransaction {
    outputs: Vec<StagedOutput>,
    finished: bool,
}

impl OutputTransaction {
    fn new(outputs: Vec<StagedOutput>) -> Self {
        Self {
            outputs,
            finished: false,
        }
    }

    fn commit(mut self) -> Result<()> {
        let result = self.commit_inner();
        match result {
            Ok(()) => {
                self.finished = true;
                for output in &mut self.outputs {
                    output.finish();
                }
                Ok(())
            }
            Err(primary) => {
                let rollback = self.rollback();
                self.finished = true;
                match rollback {
                    Ok(()) => Err(primary),
                    Err(rollback) => Err(io::Error::other(format!(
                        "output transaction failed ({primary}); rollback also failed ({rollback})"
                    ))
                    .into()),
                }
            }
        }
    }

    fn commit_inner(&mut self) -> Result<()> {
        for output in &mut self.outputs {
            output.prepare_backup()?;
        }
        #[cfg(test)]
        let mut installed = 0;
        for output in &mut self.outputs {
            output.install()?;
            #[cfg(test)]
            {
                installed += 1;
                inject_commit_failure(installed)?;
            }
        }
        sync_parent_directories(self.outputs.iter().map(|output| output.target.as_path()))?;
        Ok(())
    }

    fn rollback(&mut self) -> io::Result<()> {
        let mut first_error = None;
        for output in self.outputs.iter_mut().rev() {
            if let Err(error) = output.rollback() {
                first_error.get_or_insert(error);
            }
        }
        first_error.map_or(Ok(()), Err)
    }
}

impl Drop for OutputTransaction {
    fn drop(&mut self) {
        if !self.finished {
            let _ = self.rollback();
            self.finished = true;
        }
    }
}

fn create_unique(target: &Path, role: &str) -> Result<(PathBuf, File)> {
    let parent = output_parent(target);
    let mut last_collision = None;
    for _ in 0..100 {
        let candidate = temporary_name(parent, role);
        match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&candidate)
        {
            Ok(file) => return Ok((candidate, file)),
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {
                last_collision = Some(error);
            }
            Err(error) => return Err(error.into()),
        }
    }
    Err(last_collision
        .unwrap_or_else(|| io::Error::new(io::ErrorKind::AlreadyExists, "temporary collision"))
        .into())
}

fn unique_path(target: &Path, role: &str) -> Result<PathBuf> {
    let parent = output_parent(target);
    for _ in 0..100 {
        let candidate = temporary_name(parent, role);
        if !candidate.exists() {
            return Ok(candidate);
        }
    }
    Err(io::Error::new(io::ErrorKind::AlreadyExists, "temporary collision").into())
}

fn temporary_name(parent: &Path, role: &str) -> PathBuf {
    let number = NEXT_TEMPORARY.fetch_add(1, Ordering::Relaxed);
    parent.join(format!(
        ".indexsail-{role}-{}-{number}.tmp",
        std::process::id()
    ))
}

fn output_parent(path: &Path) -> &Path {
    path.parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or(Path::new("."))
}

fn remove_if_exists(path: &Path) -> io::Result<()> {
    match std::fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error),
    }
}

#[cfg(unix)]
fn sync_parent_directories<'a>(paths: impl Iterator<Item = &'a Path>) -> Result<()> {
    let mut parents = Vec::new();
    for path in paths {
        let parent = std::fs::canonicalize(output_parent(path))?;
        if !parents.contains(&parent) {
            File::open(&parent)?.sync_all()?;
            parents.push(parent);
        }
    }
    Ok(())
}

#[cfg(not(unix))]
#[allow(clippy::unnecessary_wraps)]
fn sync_parent_directories<'a>(_paths: impl Iterator<Item = &'a Path>) -> Result<()> {
    Ok(())
}

#[cfg(test)]
#[derive(Clone, Copy)]
enum InjectedFailure {
    Error(usize),
    Panic(usize),
}

#[cfg(test)]
thread_local! {
    static INJECTED_FAILURE: std::cell::Cell<Option<InjectedFailure>> = const { std::cell::Cell::new(None) };
}

#[cfg(test)]
pub(crate) fn inject_error_after_install_for_test(count: usize) {
    INJECTED_FAILURE.set(Some(InjectedFailure::Error(count)));
}

#[cfg(test)]
pub(crate) fn inject_panic_after_install_for_test(count: usize) {
    INJECTED_FAILURE.set(Some(InjectedFailure::Panic(count)));
}

#[cfg(test)]
fn inject_commit_failure(count: usize) -> Result<()> {
    match INJECTED_FAILURE.get() {
        Some(InjectedFailure::Error(expected)) if expected == count => {
            INJECTED_FAILURE.set(None);
            Err(io::Error::other("injected output commit failure").into())
        }
        Some(InjectedFailure::Panic(expected)) if expected == count => {
            INJECTED_FAILURE.set(None);
            panic!("injected output commit panic");
        }
        _ => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use std::panic::{AssertUnwindSafe, catch_unwind};

    use super::*;

    fn temp_dir(label: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "indexsail-atomic-{label}-{}-{}",
            std::process::id(),
            NEXT_TEMPORARY.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&path).unwrap();
        path
    }

    #[test]
    fn second_install_error_restores_both_old_outputs() {
        let directory = temp_dir("rollback");
        let first = directory.join("run.txt");
        let second = directory.join("report.json");
        std::fs::write(&first, b"old run").unwrap();
        std::fs::write(&second, b"old report").unwrap();
        inject_error_after_install_for_test(1);
        assert!(atomic_write_many(&[(&first, b"new run"), (&second, b"new report")]).is_err());
        assert_eq!(std::fs::read(&first).unwrap(), b"old run");
        assert_eq!(std::fs::read(&second).unwrap(), b"old report");
        assert_eq!(std::fs::read_dir(&directory).unwrap().count(), 2);
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn unwinding_panic_restores_both_old_outputs_and_cleans_staging() {
        let directory = temp_dir("panic");
        let first = directory.join("run.txt");
        let second = directory.join("report.json");
        std::fs::write(&first, b"old run").unwrap();
        std::fs::write(&second, b"old report").unwrap();
        inject_panic_after_install_for_test(1);
        let result = catch_unwind(AssertUnwindSafe(|| {
            atomic_write_many(&[(&first, b"new run"), (&second, b"new report")]).unwrap();
        }));
        assert!(result.is_err());
        assert_eq!(std::fs::read(&first).unwrap(), b"old run");
        assert_eq!(std::fs::read(&second).unwrap(), b"old report");
        assert_eq!(std::fs::read_dir(&directory).unwrap().count(), 2);
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn replacing_one_hard_link_does_not_modify_its_sibling() {
        let directory = temp_dir("hard-link");
        let target = directory.join("report.json");
        let sibling = directory.join("report-backup.json");
        std::fs::write(&target, b"old").unwrap();
        std::fs::hard_link(&target, &sibling).unwrap();
        atomic_write(&target, b"new").unwrap();
        assert_eq!(std::fs::read(&target).unwrap(), b"new");
        assert_eq!(std::fs::read(&sibling).unwrap(), b"old");
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn serialization_error_preserves_old_file_and_removes_partial_stage() {
        let directory = temp_dir("serializer-error");
        let target = directory.join("index.ciff");
        std::fs::write(&target, b"old index").unwrap();
        let error = atomic_write_with(&target, |file| {
            file.write_all(b"partial replacement")?;
            Err(io::Error::other("injected serializer error").into())
        })
        .unwrap_err();
        assert!(error.to_string().contains("injected serializer error"));
        assert_eq!(std::fs::read(&target).unwrap(), b"old index");
        assert_eq!(std::fs::read_dir(&directory).unwrap().count(), 1);
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn serialization_panic_closes_handle_and_removes_partial_stage() {
        let directory = temp_dir("serializer-panic");
        let target = directory.join("index.ciff");
        std::fs::write(&target, b"old index").unwrap();
        let result = catch_unwind(AssertUnwindSafe(|| {
            atomic_write_with(&target, |file| {
                file.write_all(b"partial replacement")?;
                panic!("injected serializer panic");
            })
            .unwrap();
        }));
        assert!(result.is_err());
        assert_eq!(std::fs::read(&target).unwrap(), b"old index");
        assert_eq!(std::fs::read_dir(&directory).unwrap().count(), 1);
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn symbolic_link_output_is_rejected_without_touching_referent() {
        use std::os::unix::fs::symlink;

        let directory = temp_dir("symlink");
        let referent = directory.join("referent.json");
        let output = directory.join("output.json");
        std::fs::write(&referent, b"referent").unwrap();
        symlink(&referent, &output).unwrap();
        assert!(atomic_write(&output, b"replacement").is_err());
        assert_eq!(std::fs::read(&referent).unwrap(), b"referent");
        std::fs::remove_dir_all(directory).unwrap();
    }
}

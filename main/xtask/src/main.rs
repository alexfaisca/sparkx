use chrono::Utc;
use clap::{Parser, Subcommand, ValueEnum};
use duct::cmd;
use serde::Serialize;
use std::{
    fs::{self, File},
    path::{Path, PathBuf},
};

#[derive(Parser)]
#[command(name = "xtask")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Reproducible cache/memory measurements
    Cache {
        /// Which tool to use
        #[arg(long, value_enum, default_value_t=Tool::Cachegrind)]
        tool: Tool,
        /// Dataset name (used by your builder)
        #[arg(long, default_value = "default")]
        dataset: String,
        /// Binary to run (cargo run -r --bin <name>)
        #[arg(long, default_value = "your-crate")]
        bin: String,
        /// Extra args passed to your binary
        #[arg(trailing_var_arg = true)]
        args: Vec<String>,
    },
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
enum Tool {
    Cachegrind,
    Massif,
}

#[derive(Serialize)]
struct Meta {
    timestamp_utc: String,
    rustc: String,
    git_rev: String,
    tool: String,
    dataset: String,
    cmdline: Vec<String>,
    sysinfo: SysInfo,
}

#[derive(Serialize)]
struct SysInfo {
    cpu: String,
    governor: Option<String>,
    numa_nodes: Option<String>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Cache {
            tool,
            dataset,
            bin,
            args,
        } => run_cache(tool, dataset, bin, args)?,
    }
    Ok(())
}

fn run_cache(
    tool: Tool,
    dataset: String,
    bin: String,
    args: Vec<String>,
) -> Result<(), Box<dyn std::error::Error>> {
    // 0) Ensure artifacts dir
    let out = PathBuf::from("artifacts").join("measurements");
    fs::create_dir_all(&out)?;

    // 2) Resolve paths
    let stamp = Utc::now().format("%Y%m%dT%H%M%SZ").to_string();
    let outfilepath: &Path = dataset.as_ref();
    let outfilename = outfilepath
        .file_name()
        .ok_or_else(|| -> Box<dyn std::error::Error> { "error getting out file name".into() })?
        .to_str()
        .ok_or_else(|| -> Box<dyn std::error::Error> {
            "error getting out file name str".into()
        })?;
    let prefix = out.join(format!("{}_{}.{}", outfilename, stamp, "cg.out"));
    drop(File::create(prefix.as_path())?);

    // 3) Collect meta for reproducibility
    let rustc_v = String::from_utf8(cmd!("rustc", "--version").read()?.into())?;
    let git_rev = String::from_utf8(cmd!("git", "rev-parse", "HEAD").read()?.into())?;
    let cpu = String::from_utf8(cmd!("sh", "-c", "lscpu | head -n1").read()?.into())?;

    let meta = Meta {
        timestamp_utc: stamp.clone(),
        rustc: rustc_v.trim().to_string(),
        git_rev: git_rev.trim().to_string(),
        tool: format!("{tool:?}"),
        dataset: dataset.clone(),
        cmdline: args.clone(),
        sysinfo: SysInfo {
            cpu: cpu.trim().to_string(),
            governor: read_file("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"),
            numa_nodes: read_file("/sys/devices/system/node/possible"),
        },
    };
    fs::write(
        prefix.with_extension("meta.json"),
        serde_json::to_vec_pretty(&meta)?,
    )?;

    // 4) Run under the chosen tool
    match tool {
        Tool::Cachegrind => {
            cmd!(
                "valgrind",
                "--tool=cachegrind",
                "--cachegrind-out-file",
                &prefix,
                "cargo",
                "run",
                "-r",
                "--release",
                &bin,
                "--",
                &args.concat(),
            )
            .run()?;
            // Optional: postprocess
            let txt = prefix.with_extension("cg.txt");
            let cg = cmd!("cg_annotate", &prefix).read()?;
            fs::write(txt, cg)?;
        }
        Tool::Massif => {
            let outfile = prefix.with_extension("massif.out");
            cmd!(
                "valgrind",
                "--tool=massif",
                "--massif-out-file",
                &outfile,
                "cargo",
                "run",
                "-r",
                "--release",
                &bin,
                "--",
                &args.concat()
            )
            .run()?;
            // Optional: keep both ms_print and raw
            let txt = prefix.with_extension("massif.txt");
            let ms = cmd!("ms_print", &outfile).read()?;
            fs::write(txt, ms)?;
        }
    }

    println!("Results in {}", out.display());
    Ok(())
}

fn read_file(p: &str) -> Option<String> {
    fs::read_to_string(p)
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

use criterion::{
    Throughput,
    measurement::{Measurement, ValueFormatter},
};
use criterion_perf_events::Perf;
use perfcnt::linux::{
    HardwareEventType as HardwareEvent, PerfCounterBuilderLinux as Builder,
    SoftwareEventType as SoftwareEvent,
};

pub type IPC = PerfRatio;
pub type CacheMissRate = PerfRatio;
pub type BranchMissPredictionRate = PerfRatio;
pub type TaskCPUClockRate = PerfRatio;
pub type FaultRate = PerfRatio;
pub type MajorMinorFaultRate = PerfRatio;

#[macro_export]
macro_rules! emit_criterion_bench {
    // Public entry: any number of labels, optional trailing comma
    ( $( $label:ident ),* $(,)? ) => {
        $crate::emit_criterion_bench!(@collect [] [ $( $label ),* ]);
    };

    // --------- COLLECT PHASE (build a list without `ipc`) ---------

    // Done collecting: emit code. Ensure `time_throughput` is present exactly once.
    (@collect [$($acc:ident),*] []) => {
        // gen labels: time_throughput + collected
        $crate::emit_criterion_bench!(@emit [time_throughput, $($acc),*] [ $($acc),*, time_throughput ]);
    };

    // Skip user-provided `time_throughput`; we’ll add it once later.
    (@collect [$($acc:ident),*] [time_throughput $(, $rest:ident)*]) => {
        $crate::emit_criterion_bench!(@collect [ $($acc),* ] [ $($rest),* ]);
    };

    // Push a non-time_throughput label and continue.
    (@collect [$($acc:ident),*] [$head:ident $(, $rest:ident)*]) => {
        $crate::emit_criterion_bench!(@collect [ $($acc,)* $head ] [ $($rest),* ]);
    };

    // // --------- EMIT PHASE (generate groups and main) ---------
    // ($($label:ident),* $(,)?) => {
    //     $(
    //         $crate::emit_criterion_bench!(@maybe $label);
    //     )*
    //     // Always add time and throughput exactly once
    //     $crate::emit_criterion_bench!(@ensure_time_throughput);
    // };
        // --------- EMIT PHASE (generate groups and main) ---------

    (@emit [ $($gen:ident),* ] [ $($main:ident),* ]) => {
        $(
            $crate::emit_criterion_bench!(@maybe $gen);
        )*
        // Always add time and throughput exactly once
        $crate::emit_criterion_bench!(@ensure_time_throughput);

        ::criterion::criterion_main!( $( $main ),* );
    };

    (@maybe time_throughput) => {};

    (@maybe $label:ident) => {
        paste::paste! {
            fn [<criterion_ $label>](c: &mut ::criterion::Criterion<::criterion_benches::PerfRatio>) {
                bench_generic::<TinyEdgeType, TinyLabelStandardEdge>(c, DATASETS, stringify!($label));
            }

            ::criterion::criterion_group!(
                name = $label;
                config = ::criterion::Criterion::default()
                .with_measurement(::criterion_benches::[<criterion_ $label>]());
                targets = [<criterion_ $label>]
                );
        }
    };
    (@ensure_time_throughput) => {
        fn criterion_time_throughput(c: &mut Criterion) {
            bench_time_throughput::<TinyEdgeType, TinyLabelStandardEdge>(c, DATASETS);
        }
        ::criterion::criterion_group!(time_throughput, criterion_time_throughput);
    };
}

static IPC_UNITS: &str = "instructions / cycle";
static CACHE_MISS_RATE_UNITS: &str = "misses / ref";
static BRANCH_MISSPREDICTION_RATE_UNITS: &str = "misses / instruction";
static TASK_CPU_CLOCK_RATE_UNITS: &str = "task cycles / cpu cycle";
static FAULT_RATE_UNITS: &str = "faults / cycle";
static MAJOR_MINOR_FAULT_RATE_UNITS: &str = "major faults / minor fault";

pub fn criterion_ipc() -> IPC {
    PerfRatio::new(
        Builder::from_hardware_event(HardwareEvent::Instructions),
        Builder::from_hardware_event(HardwareEvent::CPUCycles),
        IPC_UNITS,
    )
}

pub fn criterion_cache_miss_rate() -> CacheMissRate {
    PerfRatio::new(
        Builder::from_hardware_event(HardwareEvent::CacheMisses),
        Builder::from_hardware_event(HardwareEvent::CacheReferences),
        CACHE_MISS_RATE_UNITS,
    )
}

pub fn criterion_branch_missprediction_rate() -> BranchMissPredictionRate {
    PerfRatio::new(
        Builder::from_hardware_event(HardwareEvent::BranchMisses),
        Builder::from_hardware_event(HardwareEvent::BranchInstructions),
        BRANCH_MISSPREDICTION_RATE_UNITS,
    )
}

pub fn criterion_task_cpu_clock_rate() -> TaskCPUClockRate {
    PerfRatio::new(
        Builder::from_software_event(SoftwareEvent::TaskClock),
        Builder::from_software_event(SoftwareEvent::CpuClock),
        TASK_CPU_CLOCK_RATE_UNITS,
    )
}

pub fn criterion_fault_rate() -> FaultRate {
    PerfRatio::new(
        Builder::from_software_event(SoftwareEvent::PageFaults),
        Builder::from_software_event(SoftwareEvent::TaskClock),
        FAULT_RATE_UNITS,
    )
}

pub fn criterion_major_minor_fault_rate() -> MajorMinorFaultRate {
    PerfRatio::new(
        Builder::from_software_event(SoftwareEvent::PageFaultsMaj),
        Builder::from_software_event(SoftwareEvent::PageFaultsMin),
        MAJOR_MINOR_FAULT_RATE_UNITS,
    )
}

#[allow(dead_code)]
pub struct PerfRatio {
    perf_nominator: Perf,
    perf_denominator: Perf,
    formatter: PerfRatioFormatter,
}

impl PerfRatio {
    pub(crate) fn new(
        builder_nominator: Builder,
        builder_denominator: Builder,
        units: &'static str,
    ) -> Self {
        Self {
            perf_nominator: Perf::new(builder_nominator),
            perf_denominator: Perf::new(builder_denominator),
            formatter: PerfRatioFormatter {
                ration_units: units,
            },
        }
    }
}

#[allow(dead_code)]
impl Measurement for PerfRatio {
    type Intermediate = (u64, u64);
    type Value = f64;

    fn start(&self) -> Self::Intermediate {
        (self.perf_nominator.start(), self.perf_denominator.start())
    }

    fn end(&self, _i: Self::Intermediate) -> Self::Value {
        let res = (
            self.perf_nominator.end(_i.0),
            self.perf_denominator.end(_i.1),
        );
        if res.1 > 0 {
            if res.0 > 0 {
                let div = res.0 as f64 / res.1 as f64;
                if div.is_normal() { div } else { 1e-9 }
            } else {
                1e-9
            }
        } else if res.0 > 0 {
            res.0 as f64 * 10000000000.
        } else {
            1.
        }
    }

    fn add(&self, v1: &Self::Value, v2: &Self::Value) -> Self::Value {
        v1 + v2
    }

    fn zero(&self) -> Self::Value {
        0.
    }

    fn to_f64(&self, value: &Self::Value) -> f64 {
        if value.is_normal() { *value } else { 1e-9 }
    }

    fn formatter(&self) -> &dyn ValueFormatter {
        &self.formatter
    }
}

#[allow(dead_code)]
struct PerfWrapper {
    perf: Perf,
}

#[allow(dead_code)]
impl PerfWrapper {
    pub fn new(builder: Builder) -> Self {
        Self {
            perf: Perf::new(builder),
        }
    }
}

impl Measurement for PerfWrapper {
    type Intermediate = u64;
    type Value = u64;

    fn start(&self) -> Self::Intermediate {
        self.perf.start();
        0
    }

    fn end(&self, _i: Self::Intermediate) -> Self::Value {
        self.perf.end(_i)
    }

    fn add(&self, v1: &Self::Value, v2: &Self::Value) -> Self::Value {
        v1 + v2
    }

    fn zero(&self) -> Self::Value {
        0
    }

    fn to_f64(&self, value: &Self::Value) -> f64 {
        if *value != 0 { *value as f64 } else { 1e-9 }
    }

    fn formatter(&self) -> &dyn ValueFormatter {
        self.perf.formatter()
    }
}

struct PerfRatioFormatter {
    ration_units: &'static str,
}

impl ValueFormatter for PerfRatioFormatter {
    fn format_value(&self, value: f64) -> String {
        format!("{value:.3} {}", self.ration_units)
    }

    fn format_throughput(&self, throughput: &Throughput, value: f64) -> String {
        match throughput {
            Throughput::Bytes(bytes) => format!("{:.3} events/byte", value / *bytes as f64),
            Throughput::BytesDecimal(bytes) => {
                let event_per_byte = value / *bytes as f64;

                let (denominator, unit) = if *bytes < 1000 {
                    (1.0, "events/byte")
                } else if *bytes < 1000 * 1000 {
                    (1000.0, "events/kilobyte")
                } else if *bytes < 1000 * 1000 * 1000 {
                    (1000.0 * 1000.0, "events/megabyte")
                } else {
                    (1000.0 * 1000.0 * 1000.0, "events/gigabyte")
                };

                format!("{:.3} {}", event_per_byte / denominator, unit)
            }
            Throughput::Elements(els) => format!("{:.3} events/element", value / *els as f64),
        }
    }

    fn scale_values(&self, _typical_value: f64, values: &mut [f64]) -> &'static str {
        for v in values {
            if !v.is_normal() {
                *v = 1e-9;
            }
        }
        self.ration_units
    }

    fn scale_throughputs(
        &self,
        _typical_value: f64,
        throughput: &Throughput,
        values: &mut [f64],
    ) -> &'static str {
        match throughput {
            Throughput::Bytes(bytes) => {
                for val in values {
                    *val /= *bytes as f64;
                }
                "events/byte"
            }
            Throughput::BytesDecimal(bytes) => {
                let bytes_per_second = *bytes;
                let (denominator, unit) = if bytes_per_second < 1000 {
                    (1.0, "events/byte")
                } else if bytes_per_second < 1000 * 1000 {
                    (1000.0, "events/kilobyte")
                } else if bytes_per_second < 1000 * 1000 * 1000 {
                    (1000.0 * 1000.0, "events/megabyte")
                } else {
                    (1000.0 * 1000.0 * 1000.0, "events/gigabyte")
                };

                for val in values {
                    *val /= *bytes as f64;
                    *val /= denominator;
                }

                unit
            }
            Throughput::Elements(els) => {
                for val in values {
                    *val /= *els as f64;
                }
                "events/element"
            }
        }
    }

    fn scale_for_machines(&self, _values: &mut [f64]) -> &'static str {
        self.ration_units
    }
}

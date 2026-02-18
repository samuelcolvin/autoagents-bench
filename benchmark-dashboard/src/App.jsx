import { useState } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  Legend,
  Cell,
  CartesianGrid,
} from "recharts";
import benchmarkData from "../../benchmark_results_tool.json";

const TYPE_MAP = {
  AutoAgents: "rust",
  Rig: "rust",
  CrewAI: "python",
  GraphBit: "python",
  LangChain: "python",
  LangGraph: "python",
  LlamaIndex: "python",
  PydanticAI: "python",
};

const FRAMEWORKS = Object.fromEntries(
  Object.entries(benchmarkData).map(([name, data]) => [
    name,
    { ...data, type: TYPE_MAP[name] ?? "unknown" },
  ]),
);

const fwNames = Object.keys(FRAMEWORKS);
const PYTHON_FRAMEWORKS = fwNames.filter(
  (n) => FRAMEWORKS[n].type === "python",
);
const RUST_FRAMEWORKS = fwNames.filter((n) => FRAMEWORKS[n].type === "rust");

const PALETTE = [
  "#2563EB",
  "#7C3AED",
  "#DC2626",
  "#D97706",
  "#059669",
  "#0891B2",
  "#9333EA",
  "#16A34A",
  "#EA580C",
  "#0284C7",
];
const COLORS = Object.fromEntries(
  fwNames.map((n, i) => [n, PALETTE[i % PALETTE.length]]),
);

// Metadata derived from JSON
const sampleFw = FRAMEWORKS[fwNames[0]];
const totalRequests = sampleFw.total_requests;
const concurrency = sampleFw.concurrency;
const PRODUCTION_TARGET = 500; // target concurrent sessions for cost projection
const SCALE = PRODUCTION_TARGET / concurrency; // = 500/10 = 50, derived from JSON
const successRate = Math.round(
  (fwNames.reduce((s, n) => s + FRAMEWORKS[n].total_success, 0) /
    fwNames.reduce((s, n) => s + FRAMEWORKS[n].total_requests, 0)) *
    100,
);

// https://instances.vantage.sh/aws/ec2/r7g.2xlarge?currency=USD
const EC2 = [
  { name: "r7g.xlarge", ram: 32, priceHr: 0.214 },
  { name: "r7g.2xlarge", ram: 64, priceHr: 0.428 },
  { name: "r7g.4xlarge", ram: 128, priceHr: 0.857 },
  { name: "r7g.8xlarge", ram: 256, priceHr: 1.714 },
  { name: "r7g.12xlarge", ram: 384, priceHr: 2.57 },
  { name: "r7g.16xlarge", ram: 512, priceHr: 3.427 },
];

function bestEC2(memGB) {
  for (const i of EC2)
    if (i.ram >= memGB)
      return { ...i, count: 1, totalHr: i.priceHr, totalMo: i.priceHr * 730 };
  const l = EC2[EC2.length - 1];
  const c = Math.ceil(memGB / l.ram);
  return {
    ...l,
    count: c,
    totalHr: l.priceHr * c,
    totalMo: l.priceHr * c * 730,
  };
}


// Use LLM-only latency (total minus framework overhead) for the latency dimension so
// that framework overhead is not double-counted: once in latency and once in the
// overhead dimension. Both terms will always be zero for the worst framework,
// but this decomposition correctly attributes slow end-to-end time to its root cause.
const pureLat = (n) =>
  FRAMEWORKS[n].average_latency_ms -
  FRAMEWORKS[n].average_framework_overhead_corrected_ms;

// Min-max helpers for normalising to [0, 1]: best=1, worst=0.
// When all values are equal (range=0) every framework deserves full marks.
const mmHigh = (v, mn, mx) => (mx === mn ? 1 : (v - mn) / (mx - mn)); // higher is better
const mmLow = (v, mn, mx) => (mx === mn ? 1 : (mx - v) / (mx - mn)); // lower is better

const maxLat = Math.max(...fwNames.map(pureLat));
const minLat = Math.min(...fwNames.map(pureLat));
const maxMem = Math.max(...fwNames.map((n) => FRAMEWORKS[n].memory_peak_mb));
const minMem = Math.min(...fwNames.map((n) => FRAMEWORKS[n].memory_peak_mb));
const maxTp = Math.max(...fwNames.map((n) => FRAMEWORKS[n].throughput_rps));
const minTp = Math.min(...fwNames.map((n) => FRAMEWORKS[n].throughput_rps));
// CPU efficiency: throughput delivered per unit of CPU% — higher is better.
// A framework burning more CPU but delivering proportionally more throughput
// should not be penalised vs one that is merely idle.
// Guard against cpu_usage_percent === 0 to avoid silent NaN propagation.
const cpuEfficiency = Object.fromEntries(
  fwNames.map((n) => [
    n,
    FRAMEWORKS[n].throughput_rps /
      Math.max(FRAMEWORKS[n].cpu_usage_percent, 0.001),
  ]),
);
const maxCpuEfficiency = Math.max(...fwNames.map((n) => cpuEfficiency[n]));
const minCpuEfficiency = Math.min(...fwNames.map((n) => cpuEfficiency[n]));
const maxOh = Math.max(
  ...fwNames.map((n) => FRAMEWORKS[n].average_framework_overhead_corrected_ms),
);
const minOh = Math.min(
  ...fwNames.map((n) => FRAMEWORKS[n].average_framework_overhead_corrected_ms),
);

// Overhead dimension is only meaningful when frameworks differ in framework overhead.
// When all values are equal (e.g. all 0), the dimension is excluded and its 10-point
// weight is redistributed proportionally to the remaining four so the max is still 100.
const includeOverhead = maxOh > minOh;
// Base weights (without overhead): 25+20+30+15 = 90
const _BASE = { latency: 25, memory: 20, throughput: 30, cpu: 15 };
const WEIGHTS = includeOverhead
  ? { ..._BASE, overhead: 10 }
  : {
      latency: (100 * 25) / 90, // 27.78
      memory: (100 * 20) / 90, // 22.22
      throughput: (100 * 30) / 90, // 33.33
      cpu: (100 * 15) / 90, // 16.67
    };

const scores = {};
fwNames.forEach((n) => {
  const f = FRAMEWORKS[n];
  scores[n] =
    mmLow(pureLat(n), minLat, maxLat) * WEIGHTS.latency +
    mmLow(f.memory_peak_mb, minMem, maxMem) * WEIGHTS.memory +
    mmHigh(f.throughput_rps, minTp, maxTp) * WEIGHTS.throughput +
    mmHigh(cpuEfficiency[n], minCpuEfficiency, maxCpuEfficiency) * WEIGHTS.cpu +
    (includeOverhead
      ? mmLow(f.average_framework_overhead_corrected_ms, minOh, maxOh) *
        WEIGHTS.overhead
      : 0);
});
const ranked = Object.entries(scores).sort((a, b) => b[1] - a[1]);

// Top-ranked rust framework for cost comparison
const topRustFw =
  ranked.find(([n]) => FRAMEWORKS[n].type === "rust")?.[0] ??
  RUST_FRAMEWORKS[0];
const topRustMem = FRAMEWORKS[topRustFw].memory_peak_mb * SCALE;
const topRustCost = bestEC2(topRustMem / 1024);

// Best Python alternative by composite score (most performance-equivalent comparison)
const bestPythonFw =
  ranked.find(([n]) => FRAMEWORKS[n].type === "python")?.[0] ??
  PYTHON_FRAMEWORKS[0];
const bestPythonMem = FRAMEWORKS[bestPythonFw].memory_peak_mb * SCALE;
const bestPythonCost = bestEC2(bestPythonMem / 1024);

// Average Python (industry-average comparison)
const avgPythonMem =
  (PYTHON_FRAMEWORKS.reduce((s, f) => s + FRAMEWORKS[f].memory_peak_mb, 0) /
    PYTHON_FRAMEWORKS.length) *
  SCALE;
const avgPythonCost = bestEC2(avgPythonMem / 1024);

const Tip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div
      style={{
        background: "#fff",
        border: "1px solid #E5E7EB",
        borderRadius: 6,
        padding: "8px 12px",
        boxShadow: "0 2px 8px rgba(0,0,0,0.06)",
        fontFamily: "'IBM Plex Sans', sans-serif",
        fontSize: 12,
      }}
    >
      <p style={{ color: "#1F2937", fontWeight: 600, marginBottom: 4 }}>
        {label}
      </p>
      {payload.map((p, i) => (
        <p key={i} style={{ color: p.color || p.fill, margin: "2px 0" }}>
          {p.name}:{" "}
          <span style={{ color: "#1F2937", fontWeight: 600 }}>
            {typeof p.value === "number"
              ? p.value.toLocaleString(undefined, { maximumFractionDigits: 1 })
              : p.value}
          </span>
        </p>
      ))}
    </div>
  );
};

const Card = ({ children, accent, style: s }) => (
  <div
    style={{
      background: "#FFFFFF",
      border: "1px solid #D1D5DB",
      borderRadius: 6,
      borderLeft: accent ? `3px solid ${accent}` : undefined,
      ...s,
    }}
  >
    {children}
  </div>
);

const Badge = ({ type }) => (
  <span
    style={{
      padding: "2px 6px",
      borderRadius: 3,
      fontSize: 10,
      fontWeight: 500,
      letterSpacing: 0.3,
      fontFamily: "'IBM Plex Mono', monospace",
      background: type === "rust" ? "#FFFBEB" : "#EEF2FF",
      color: type === "rust" ? "#92400E" : "#4338CA",
      border: `1px solid ${type === "rust" ? "#FDE68A" : "#C7D2FE"}`,
    }}
  >
    {type}
  </span>
);

const ScoringNote = () => (
  <Card style={{ padding: "16px 20px", marginTop: 24, background: "#F9FAFB" }}>
    <div
      style={{
        display: "flex",
        gap: 8,
        alignItems: "flex-start",
        marginBottom: 10,
      }}
    >
      <svg
        width="16"
        height="16"
        viewBox="0 0 16 16"
        fill="none"
        style={{ marginTop: 1, flexShrink: 0 }}
      >
        <circle
          cx="8"
          cy="8"
          r="7"
          stroke="#9CA3AF"
          strokeWidth="1.5"
          fill="none"
        />
        <path
          d="M8 7v4M8 5h.01"
          stroke="#9CA3AF"
          strokeWidth="1.5"
          strokeLinecap="round"
        />
      </svg>
      <div style={{ fontSize: 13, fontWeight: 600, color: "#374151" }}>
        Scoring Methodology
      </div>
    </div>
    <p
      style={{
        fontSize: 12,
        color: "#6B7280",
        lineHeight: 1.7,
        marginBottom: 10,
      }}
    >
      The composite score (0–100) is a weighted sum of{" "}
      {includeOverhead ? "five" : "four"} normalized metrics. Each metric uses
      min-max normalisation: the best framework scores its full weight; the
      worst scores 0. Weights reflect enterprise priorities: throughput capacity
      is most critical, followed by latency, memory cost, and CPU efficiency.
      {includeOverhead &&
        " Framework overhead is included when frameworks differ."}
    </p>
    <table
      style={{
        width: "100%",
        borderCollapse: "collapse",
        fontSize: 12,
        fontFamily: "'IBM Plex Mono', monospace",
      }}
    >
      <thead>
        <tr style={{ borderBottom: "1px solid #E5E7EB" }}>
          <th
            style={{
              padding: "6px 8px",
              textAlign: "left",
              color: "#6B7280",
              fontWeight: 600,
              fontSize: 10,
              textTransform: "uppercase",
              letterSpacing: 0.5,
            }}
          >
            Metric
          </th>
          <th
            style={{
              padding: "6px 8px",
              textAlign: "left",
              color: "#6B7280",
              fontWeight: 600,
              fontSize: 10,
              textTransform: "uppercase",
              letterSpacing: 0.5,
            }}
          >
            Weight
          </th>
          <th
            style={{
              padding: "6px 8px",
              textAlign: "left",
              color: "#6B7280",
              fontWeight: 600,
              fontSize: 10,
              textTransform: "uppercase",
              letterSpacing: 0.5,
            }}
          >
            Formula
          </th>
        </tr>
      </thead>
      <tbody>
        {[
          [
            "Throughput (rps)",
            `${WEIGHTS.throughput.toFixed(1)}%`,
            "(value − min) / (max − min) × weight",
          ],
          [
            "LLM Latency",
            `${WEIGHTS.latency.toFixed(1)}%`,
            "(max − (latency − overhead)) / (max − min) × weight",
          ],
          [
            "Peak Memory",
            `${WEIGHTS.memory.toFixed(1)}%`,
            "(max − value) / (max − min) × weight",
          ],
          [
            "CPU Efficiency",
            `${WEIGHTS.cpu.toFixed(1)}%`,
            "(rps÷cpu% − min) / (max − min) × weight",
          ],
          ...(includeOverhead
            ? [
                [
                  "Framework Overhead",
                  `${WEIGHTS.overhead.toFixed(1)}%`,
                  "(max − overhead) / (max − min) × weight",
                ],
              ]
            : []),
        ].map(([m, w, f]) => (
          <tr key={m} style={{ borderBottom: "1px solid #E5E7EB" }}>
            <td style={{ padding: "6px 8px", color: "#374151" }}>{m}</td>
            <td
              style={{ padding: "6px 8px", color: "#2563EB", fontWeight: 600 }}
            >
              {w}
            </td>
            <td style={{ padding: "6px 8px", color: "#6B7280", fontSize: 11 }}>
              {f}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
    <p
      style={{ fontSize: 11, color: "#9CA3AF", lineHeight: 1.6, marginTop: 10 }}
    >
      All active dimensions use min-max normalisation: the best-performing
      framework scores its full weight; the worst scores 0. Latency is scored on
      LLM-only time (total latency minus framework overhead) to avoid
      double-penalising frameworks with high overhead — the overhead dimension
      captures that separately when it is active. CPU efficiency (rps ÷ cpu%)
      avoids penalising frameworks that burn more CPU but deliver proportionally
      higher throughput. Framework overhead is only shown as a scoring dimension
      when at least one framework reports non-zero overhead; otherwise its 10%
      weight is redistributed proportionally across the remaining four. A
      perfect score of 100 means best-in-class across every active dimension.
    </p>
  </Card>
);

export default function App() {
  const [tab, setTab] = useState("overview");

  const latData = fwNames
    .map((n) => ({
      name: n,
      avg: Math.round(FRAMEWORKS[n].average_latency_ms),
      p95: Math.round(FRAMEWORKS[n].p95_latency_ms),
      p99: Math.round(FRAMEWORKS[n].p99_latency_ms),
    }))
    .sort((a, b) => a.avg - b.avg);
  const tpData = fwNames
    .map((n) => ({ name: n, rps: FRAMEWORKS[n].throughput_rps }))
    .sort((a, b) => b.rps - a.rps);
  const memData = fwNames
    .map((n) => ({ name: n, mb: Math.round(FRAMEWORKS[n].memory_peak_mb) }))
    .sort((a, b) => a.mb - b.mb);
  const cpuData = fwNames
    .map((n) => ({ name: n, cpu: FRAMEWORKS[n].cpu_usage_percent }))
    .sort((a, b) => a.cpu - b.cpu);

  const top4 = ranked.slice(0, 4).map(([n]) => n);
  const radar = [
    {
      m: "Speed",
      ...Object.fromEntries(
        top4.map((n) => [
          n,
          Math.round(mmLow(pureLat(n), minLat, maxLat) * 100),
        ]),
      ),
    },
    {
      m: "Memory",
      ...Object.fromEntries(
        top4.map((n) => [
          n,
          Math.round(mmLow(FRAMEWORKS[n].memory_peak_mb, minMem, maxMem) * 100),
        ]),
      ),
    },
    {
      m: "Throughput",
      ...Object.fromEntries(
        top4.map((n) => [
          n,
          Math.round(mmHigh(FRAMEWORKS[n].throughput_rps, minTp, maxTp) * 100),
        ]),
      ),
    },
    {
      m: "CPU Eff.",
      ...Object.fromEntries(
        top4.map((n) => [
          n,
          Math.round(
            mmHigh(cpuEfficiency[n], minCpuEfficiency, maxCpuEfficiency) * 100,
          ),
        ]),
      ),
    },
    ...(includeOverhead
      ? [
          {
            m: "Low Overhead",
            ...Object.fromEntries(
              top4.map((n) => [
                n,
                Math.round(
                  mmLow(
                    FRAMEWORKS[n].average_framework_overhead_corrected_ms,
                    minOh,
                    maxOh,
                  ) * 100,
                ),
              ]),
            ),
          },
        ]
      : []),
  ];

  // Stat cards — all winners derived from JSON
  const bestLatFw = fwNames.reduce((b, n) =>
    FRAMEWORKS[n].average_latency_ms < FRAMEWORKS[b].average_latency_ms ? n : b,
  );
  const bestTpFw = fwNames.reduce((b, n) =>
    FRAMEWORKS[n].throughput_rps > FRAMEWORKS[b].throughput_rps ? n : b,
  );
  const bestMemFw = fwNames.reduce((b, n) =>
    FRAMEWORKS[n].memory_peak_mb < FRAMEWORKS[b].memory_peak_mb ? n : b,
  );
  const bestCpuFw = fwNames.reduce((b, n) =>
    cpuEfficiency[n] > cpuEfficiency[b] ? n : b,
  );
  const bestOhFw = fwNames.reduce((b, n) =>
    FRAMEWORKS[n].average_framework_overhead_corrected_ms <
    FRAMEWORKS[b].average_framework_overhead_corrected_ms
      ? n
      : b,
  );
  const bestP99Fw = fwNames.reduce((b, n) =>
    FRAMEWORKS[n].p99_latency_ms < FRAMEWORKS[b].p99_latency_ms ? n : b,
  );
  const statCards = [
    {
      l: "Fastest Latency",
      v: Math.round(FRAMEWORKS[bestLatFw].average_latency_ms).toLocaleString(),
      u: "ms",
      s: bestLatFw,
      c: COLORS[bestLatFw],
    },
    {
      l: "Best Throughput",
      v: FRAMEWORKS[bestTpFw].throughput_rps.toFixed(2),
      u: "rps",
      s: bestTpFw,
      c: COLORS[bestTpFw],
    },
    {
      l: "Lowest Memory",
      v: Math.round(FRAMEWORKS[bestMemFw].memory_peak_mb).toLocaleString(),
      u: "MB",
      s: bestMemFw,
      c: COLORS[bestMemFw],
    },
    {
      l: "Best CPU Efficiency",
      v: (
        mmHigh(cpuEfficiency[bestCpuFw], minCpuEfficiency, maxCpuEfficiency) *
        100
      ).toFixed(1),
      u: "/ 100",
      s: bestCpuFw,
      c: COLORS[bestCpuFw],
    },
    {
      l: "Best P99",
      v: (FRAMEWORKS[bestP99Fw].p99_latency_ms / 1000).toFixed(1),
      u: "s",
      s: bestP99Fw,
      c: COLORS[bestP99Fw],
    },
  ];

  const costData = [...RUST_FRAMEWORKS, ...PYTHON_FRAMEWORKS]
    .map((n) => {
      const mem = FRAMEWORKS[n].memory_peak_mb * SCALE;
      const c = bestEC2(mem / 1024);
      // Daily capacity ceiling: how many requests this deployment can sustain
      const dailyCapM = (FRAMEWORKS[n].throughput_rps * SCALE * 86_400 / 1_000_000).toFixed(1);
      return {
        name: n,
        monthly: Math.round(c.totalMo),
        memGB: Math.round(mem / 1024),
        dailyCapM,
      };
    })
    .sort((a, b) => a.monthly - b.monthly);

  const th = {
    padding: "10px 14px",
    textAlign: "left",
    color: "#6B7280",
    fontWeight: 600,
    fontSize: 10,
    textTransform: "uppercase",
    letterSpacing: 0.5,
    borderBottom: "1px solid #D1D5DB",
    fontFamily: "'IBM Plex Mono', monospace",
    background: "#F9FAFB",
  };
  const td = {
    padding: "11px 14px",
    borderBottom: "1px solid #E5E7EB",
    fontSize: 13,
    color: "#4B5563",
  };

  const tabs = [
    { id: "overview", label: "Overview" },
    { id: "latency", label: "Latency & Throughput" },
    { id: "resources", label: "Resource Usage" },
    { id: "cost", label: "Cost Analysis" },
  ];

  const axisStyle = { stroke: "#D1D5DB" };
  const gridStyle = { strokeDasharray: "3 3", stroke: "#E5E7EB" };
  const xTickStyle = {
    fill: "#6B7280",
    fontSize: 11,
    fontFamily: "'IBM Plex Mono', monospace",
  };
  const yTickStyle = { fill: "#6B7280", fontSize: 11 };

  return (
    <div
      style={{
        minHeight: "100vh",
        background: "#EBEEF3",
        fontFamily: "'IBM Plex Sans', sans-serif",
        color: "#4B5563",
      }}
    >
      <style>{`@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600;700&display=swap'); * { box-sizing: border-box; margin: 0; padding: 0; } table { border-spacing: 0; }`}</style>

      {/* Nav */}
      <div style={{ background: "#fff", borderBottom: "1px solid #D1D5DB" }}>
        <div
          style={{
            maxWidth: 1080,
            margin: "0 auto",
            padding: "0 20px",
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            height: 50,
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
              <rect width="20" height="20" rx="4" fill="#2563EB" />
              <path
                d="M5 14L10 6l5 8"
                stroke="#fff"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
            <span style={{ fontWeight: 600, fontSize: 14, color: "#1F2937" }}>
              {ranked[0][0]}
            </span>
            <span style={{ color: "#D1D5DB", fontSize: 14 }}>·</span>
            <span style={{ fontWeight: 500, fontSize: 13, color: "#6B7280" }}>
              Benchmark Report
            </span>
          </div>
          <span
            style={{
              fontSize: 11,
              fontFamily: "'IBM Plex Mono', monospace",
              color: "#6B7280",
            }}
          >
            {fwNames.length} Frameworks · {totalRequests} Requests ·{" "}
            {concurrency} Concurrency
          </span>
        </div>
      </div>

      <div
        style={{ maxWidth: 1080, margin: "0 auto", padding: "24px 20px 48px" }}
      >
        <h1
          style={{
            fontSize: 24,
            fontWeight: 700,
            color: "#1F2937",
            letterSpacing: -0.3,
            marginBottom: 6,
          }}
        >
          AI Agent Framework Performance Benchmark
        </h1>
        <p
          style={{
            fontSize: 14,
            color: "#6B7280",
            maxWidth: 600,
            lineHeight: 1.6,
            marginBottom: 22,
          }}
        >
          Comparative analysis across latency, throughput, memory, CPU
          utilization, and framework overhead with AWS EC2 cost projections at
          production scale.
        </p>

        {/* Winner */}
        <Card
          accent="#2563EB"
          style={{
            padding: "16px 20px",
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            flexWrap: "wrap",
            gap: 14,
            marginBottom: 24,
          }}
        >
          <div>
            <div
              style={{
                fontSize: 10,
                textTransform: "uppercase",
                letterSpacing: 1,
                color: "#2563EB",
                fontFamily: "'IBM Plex Mono', monospace",
                fontWeight: 600,
                marginBottom: 2,
              }}
            >
              Overall Best Framework
            </div>
            <div style={{ fontSize: 22, fontWeight: 700, color: "#1F2937" }}>
              {ranked[0][0]}
            </div>
            <div style={{ fontSize: 12, color: "#6B7280", marginTop: 2 }}>
              Composite score:{" "}
              <span style={{ color: "#2563EB", fontWeight: 600 }}>
                {ranked[0][1].toFixed(1)}
              </span>{" "}
              / 100 — lowest latency, smallest memory footprint, highest
              throughput
            </div>
          </div>
          <div style={{ display: "flex", gap: 6 }}>
            {ranked.slice(0, 4).map(([n, s], i) => (
              <div
                key={n}
                style={{
                  background: i === 0 ? "#DBEAFE" : "#F9FAFB",
                  border: `1px solid ${i === 0 ? "#BFDBFE" : "#D1D5DB"}`,
                  borderRadius: 5,
                  padding: "7px 11px",
                  textAlign: "center",
                  minWidth: 68,
                }}
              >
                <div
                  style={{
                    fontSize: 10,
                    color: "#6B7280",
                    fontFamily: "'IBM Plex Mono', monospace",
                    fontWeight: 500,
                  }}
                >
                  #{i + 1}
                </div>
                <div
                  style={{
                    fontSize: 12,
                    fontWeight: 600,
                    color: i === 0 ? "#2563EB" : "#1F2937",
                    marginTop: 1,
                  }}
                >
                  {n}
                </div>
                <div style={{ fontSize: 11, color: "#6B7280" }}>
                  {s.toFixed(1)}
                </div>
              </div>
            ))}
          </div>
        </Card>

        {/* Tabs */}
        <div
          style={{
            display: "flex",
            gap: 0,
            borderBottom: "1px solid #D1D5DB",
            marginBottom: 24,
          }}
        >
          {tabs.map((t) => (
            <button
              key={t.id}
              onClick={() => setTab(t.id)}
              style={{
                padding: "9px 16px",
                border: "none",
                borderBottom:
                  tab === t.id ? "2px solid #2563EB" : "2px solid transparent",
                background: "transparent",
                color: tab === t.id ? "#2563EB" : "#6B7280",
                fontFamily: "'IBM Plex Sans', sans-serif",
                fontSize: 13,
                cursor: "pointer",
                fontWeight: tab === t.id ? 600 : 500,
                marginBottom: -1,
                transition: "all 0.15s",
              }}
            >
              {t.label}
            </button>
          ))}
        </div>

        {/* ---- OVERVIEW ---- */}
        {tab === "overview" && (
          <>
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(auto-fill, minmax(180px, 1fr))",
                gap: 10,
                marginBottom: 28,
              }}
            >
              {statCards.map(({ l, v, u, s, c }) => (
                <Card key={l} accent={c} style={{ padding: "14px 14px" }}>
                  <div
                    style={{
                      fontSize: 10,
                      textTransform: "uppercase",
                      letterSpacing: 0.8,
                      color: "#6B7280",
                      fontFamily: "'IBM Plex Mono', monospace",
                      fontWeight: 500,
                      marginBottom: 6,
                    }}
                  >
                    {l}
                  </div>
                  <div
                    style={{ fontSize: 22, fontWeight: 700, color: "#1F2937" }}
                  >
                    {v}
                    <span
                      style={{
                        fontSize: 11,
                        color: c,
                        marginLeft: 3,
                        fontWeight: 600,
                      }}
                    >
                      {u}
                    </span>
                  </div>
                  <div style={{ fontSize: 11, color: "#6B7280", marginTop: 4 }}>
                    {s}
                  </div>
                </Card>
              ))}
            </div>

            <h3
              style={{
                fontSize: 15,
                fontWeight: 600,
                color: "#1F2937",
                marginBottom: 10,
              }}
            >
              Composite Radar — Top 4 Frameworks
            </h3>
            <Card style={{ padding: 16, marginBottom: 28 }}>
              <ResponsiveContainer width="100%" height={320}>
                <RadarChart data={radar}>
                  <PolarGrid stroke="#D1D5DB" />
                  <PolarAngleAxis
                    dataKey="m"
                    tick={{
                      fill: "#6B7280",
                      fontSize: 11,
                      fontFamily: "'IBM Plex Mono', monospace",
                    }}
                  />
                  <PolarRadiusAxis
                    tick={false}
                    axisLine={false}
                    domain={[0, 100]}
                  />
                  {top4.map((n) => (
                    <Radar
                      key={n}
                      name={n}
                      dataKey={n}
                      stroke={COLORS[n]}
                      fill={COLORS[n]}
                      fillOpacity={0.08}
                      strokeWidth={1.5}
                    />
                  ))}
                  <Legend
                    wrapperStyle={{
                      fontFamily: "'IBM Plex Sans', sans-serif",
                      fontSize: 12,
                      color: "#6B7280",
                    }}
                  />
                  <Tooltip content={<Tip />} />
                </RadarChart>
              </ResponsiveContainer>
            </Card>

            <h3
              style={{
                fontSize: 15,
                fontWeight: 600,
                color: "#1F2937",
                marginBottom: 10,
              }}
            >
              Full Rankings
            </h3>
            <Card style={{ overflowX: "auto" }}>
              <table style={{ width: "100%", borderCollapse: "collapse" }}>
                <thead>
                  <tr>
                    {[
                      "Rank",
                      "Framework",
                      "Type",
                      "Score",
                      "Avg Latency",
                      "Throughput",
                      "Memory",
                      "CPU",
                      "CPU Efficiency",
                      "Overhead",
                    ].map((h) => (
                      <th key={h} style={th}>
                        {h}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {ranked.map(([n, sc], i) => {
                    const f = FRAMEWORKS[n];
                    return (
                      <tr
                        key={n}
                        style={{
                          background:
                            i === 0
                              ? "#F4F7FB"
                              : i % 2 === 1
                                ? "#F9FAFB"
                                : "#fff",
                        }}
                      >
                        <td
                          style={{
                            ...td,
                            fontWeight: 600,
                            color: i === 0 ? "#2563EB" : "#1F2937",
                          }}
                        >
                          #{i + 1}
                        </td>
                        <td
                          style={{ ...td, fontWeight: 600, color: COLORS[n] }}
                        >
                          {n}
                        </td>
                        <td style={td}>
                          <Badge type={f.type} />
                        </td>
                        <td
                          style={{ ...td, fontWeight: 600, color: "#1F2937" }}
                        >
                          {sc.toFixed(1)}
                        </td>
                        <td style={td}>
                          {f.average_latency_ms.toLocaleString(undefined, {
                            maximumFractionDigits: 0,
                          })}{" "}
                          ms
                        </td>
                        <td style={td}>{f.throughput_rps.toFixed(2)} rps</td>
                        <td style={td}>
                          {f.memory_peak_mb.toLocaleString(undefined, {
                            maximumFractionDigits: 0,
                          })}{" "}
                          MB
                        </td>
                        <td style={td}>{f.cpu_usage_percent.toFixed(1)}%</td>
                        <td style={td}>
                          {(
                            mmHigh(
                              cpuEfficiency[n],
                              minCpuEfficiency,
                              maxCpuEfficiency,
                            ) * 100
                          ).toFixed(1)}
                        </td>
                        <td style={td}>
                          {f.average_framework_overhead_corrected_ms.toFixed(1)}{" "}
                          ms
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </Card>

            <ScoringNote />
          </>
        )}

        {/* ---- LATENCY ---- */}
        {tab === "latency" && (
          <>
            <h3
              style={{
                fontSize: 15,
                fontWeight: 600,
                color: "#1F2937",
                marginBottom: 10,
              }}
            >
              Average Latency (ms)
            </h3>
            <Card style={{ padding: 16, marginBottom: 28 }}>
              <ResponsiveContainer width="100%" height={320}>
                <BarChart data={latData} barCategoryGap="18%">
                  <CartesianGrid {...gridStyle} />
                  <XAxis
                    dataKey="name"
                    tick={xTickStyle}
                    axisLine={axisStyle}
                    tickLine={false}
                  />
                  <YAxis
                    tick={yTickStyle}
                    axisLine={axisStyle}
                    tickLine={false}
                  />
                  <Tooltip content={<Tip />} />
                  <Bar
                    dataKey="avg"
                    name="Avg Latency (ms)"
                    radius={[3, 3, 0, 0]}
                  >
                    {latData.map((d) => (
                      <Cell key={d.name} fill={COLORS[d.name]} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </Card>

            <h3
              style={{
                fontSize: 15,
                fontWeight: 600,
                color: "#1F2937",
                marginBottom: 10,
              }}
            >
              P95 vs P99 Tail Latency (ms)
            </h3>
            <Card style={{ padding: 16, marginBottom: 28 }}>
              <ResponsiveContainer width="100%" height={320}>
                <BarChart data={latData} barCategoryGap="18%">
                  <CartesianGrid {...gridStyle} />
                  <XAxis
                    dataKey="name"
                    tick={xTickStyle}
                    axisLine={axisStyle}
                    tickLine={false}
                  />
                  <YAxis
                    tick={yTickStyle}
                    axisLine={axisStyle}
                    tickLine={false}
                  />
                  <Tooltip content={<Tip />} />
                  <Bar
                    dataKey="p95"
                    name="P95"
                    fill="#2563EB"
                    fillOpacity={0.55}
                    radius={[3, 3, 0, 0]}
                  />
                  <Bar
                    dataKey="p99"
                    name="P99"
                    fill="#A8C4DA"
                    radius={[3, 3, 0, 0]}
                  />
                  <Legend
                    wrapperStyle={{
                      fontFamily: "'IBM Plex Sans', sans-serif",
                      fontSize: 12,
                    }}
                  />
                </BarChart>
              </ResponsiveContainer>
            </Card>

            <h3
              style={{
                fontSize: 15,
                fontWeight: 600,
                color: "#1F2937",
                marginBottom: 10,
              }}
            >
              Throughput (Requests/sec)
            </h3>
            <Card style={{ padding: 16 }}>
              <ResponsiveContainer width="100%" height={270}>
                <BarChart data={tpData} layout="vertical" barCategoryGap="16%">
                  <CartesianGrid {...gridStyle} horizontal={false} />
                  <XAxis
                    type="number"
                    tick={yTickStyle}
                    axisLine={axisStyle}
                    tickLine={false}
                  />
                  <YAxis
                    type="category"
                    dataKey="name"
                    tick={xTickStyle}
                    axisLine={axisStyle}
                    tickLine={false}
                    width={85}
                  />
                  <Tooltip content={<Tip />} />
                  <Bar dataKey="rps" name="Requests/s" radius={[0, 3, 3, 0]}>
                    {tpData.map((d) => (
                      <Cell key={d.name} fill={COLORS[d.name]} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </Card>
          </>
        )}

        {/* ---- RESOURCES ---- */}
        {tab === "resources" && (
          <>
            <h3
              style={{
                fontSize: 15,
                fontWeight: 600,
                color: "#1F2937",
                marginBottom: 10,
              }}
            >
              Peak Memory Usage (MB)
            </h3>
            <Card style={{ padding: 16, marginBottom: 28 }}>
              <ResponsiveContainer width="100%" height={320}>
                <BarChart data={memData} barCategoryGap="18%">
                  <CartesianGrid {...gridStyle} />
                  <XAxis
                    dataKey="name"
                    tick={xTickStyle}
                    axisLine={axisStyle}
                    tickLine={false}
                  />
                  <YAxis
                    tick={yTickStyle}
                    axisLine={axisStyle}
                    tickLine={false}
                  />
                  <Tooltip content={<Tip />} />
                  <Bar
                    dataKey="mb"
                    name="Peak Memory (MB)"
                    radius={[3, 3, 0, 0]}
                  >
                    {memData.map((d) => (
                      <Cell key={d.name} fill={COLORS[d.name]} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </Card>

            <h3
              style={{
                fontSize: 15,
                fontWeight: 600,
                color: "#1F2937",
                marginBottom: 10,
              }}
            >
              CPU Utilization (%)
            </h3>
            <Card style={{ padding: 16, marginBottom: 28 }}>
              <ResponsiveContainer width="100%" height={270}>
                <BarChart data={cpuData} layout="vertical" barCategoryGap="16%">
                  <CartesianGrid {...gridStyle} horizontal={false} />
                  <XAxis
                    type="number"
                    tick={yTickStyle}
                    axisLine={axisStyle}
                    tickLine={false}
                    domain={[0, 100]}
                  />
                  <YAxis
                    type="category"
                    dataKey="name"
                    tick={xTickStyle}
                    axisLine={axisStyle}
                    tickLine={false}
                    width={85}
                  />
                  <Tooltip content={<Tip />} />
                  <Bar dataKey="cpu" name="CPU %" radius={[0, 3, 3, 0]}>
                    {cpuData.map((d) => (
                      <Cell key={d.name} fill={COLORS[d.name]} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </Card>

          </>
        )}

        {/* ---- COST ---- */}
        {tab === "cost" && (
          <>
            <p
              style={{
                color: "#6B7280",
                fontSize: 13,
                lineHeight: 1.7,
                marginBottom: 22,
                maxWidth: 680,
              }}
            >
              Benchmark sample: {totalRequests} requests at{" "}
              <strong style={{ color: "#374151" }}>
                {concurrency} concurrent sessions
              </strong>{" "}
              (source: JSON). Scale factor:{" "}
              <strong style={{ color: "#374151" }}>
                {PRODUCTION_TARGET} ÷ {concurrency} = ×{SCALE}
              </strong>{" "}
              — peak memory is multiplied by ×{SCALE} to project the footprint
              at{" "}
              <strong style={{ color: "#374151" }}>
                {PRODUCTION_TARGET} concurrent sessions
              </strong>
              . Mapped to the smallest fitting AWS r7g (Graviton3) instance,
              US-East-1 on-demand.
            </p>

            <div
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))",
                gap: 12,
                marginBottom: 22,
              }}
            >
              {/* Top Rust framework */}
              <Card accent={COLORS[topRustFw]} style={{ padding: "18px 20px" }}>
                <div
                  style={{
                    fontSize: 10,
                    textTransform: "uppercase",
                    letterSpacing: 0.8,
                    color: COLORS[topRustFw],
                    fontFamily: "'IBM Plex Mono', monospace",
                    fontWeight: 600,
                  }}
                >
                  {topRustFw} ({FRAMEWORKS[topRustFw].type})
                </div>
                <div
                  style={{
                    fontSize: 32,
                    fontWeight: 700,
                    color: "#1F2937",
                    margin: "8px 0 2px",
                  }}
                >
                  $
                  {topRustCost.totalMo.toLocaleString(undefined, {
                    maximumFractionDigits: 0,
                  })}
                  <span
                    style={{ fontSize: 13, color: "#6B7280", fontWeight: 500 }}
                  >
                    /mo
                  </span>
                </div>
                <div
                  style={{ fontSize: 12, color: "#6B7280", marginBottom: 12 }}
                >
                  {topRustCost.count}× {topRustCost.name} · {topRustCost.ram} GB
                  RAM
                </div>
                <div
                  style={{
                    display: "grid",
                    gridTemplateColumns: "1fr 1fr",
                    gap: 8,
                  }}
                >
                  <div
                    style={{
                      background: "#F9FAFB",
                      borderRadius: 5,
                      padding: "8px 10px",
                      border: "1px solid #D1D5DB",
                    }}
                  >
                    <div
                      style={{
                        fontSize: 10,
                        color: "#6B7280",
                        fontFamily: "'IBM Plex Mono', monospace",
                        fontWeight: 500,
                      }}
                    >
                      MEMORY
                    </div>
                    <div
                      style={{
                        fontSize: 15,
                        fontWeight: 600,
                        color: "#1F2937",
                        marginTop: 1,
                      }}
                    >
                      {(topRustMem / 1024).toFixed(1)} GB
                    </div>
                  </div>
                  <div
                    style={{
                      background: "#F9FAFB",
                      borderRadius: 5,
                      padding: "8px 10px",
                      border: "1px solid #D1D5DB",
                    }}
                  >
                    <div
                      style={{
                        fontSize: 10,
                        color: "#6B7280",
                        fontFamily: "'IBM Plex Mono', monospace",
                        fontWeight: 500,
                      }}
                    >
                      HOURLY
                    </div>
                    <div
                      style={{
                        fontSize: 15,
                        fontWeight: 600,
                        color: "#1F2937",
                        marginTop: 1,
                      }}
                    >
                      ${topRustCost.totalHr.toFixed(2)}
                    </div>
                  </div>
                </div>
              </Card>

              {/* Avg Python */}
              <Card accent="#DC2626" style={{ padding: "18px 20px" }}>
                <div
                  style={{
                    fontSize: 10,
                    textTransform: "uppercase",
                    letterSpacing: 0.8,
                    color: "#DC2626",
                    fontFamily: "'IBM Plex Mono', monospace",
                    fontWeight: 600,
                  }}
                >
                  Avg Python Framework
                </div>
                <div
                  style={{
                    fontSize: 32,
                    fontWeight: 700,
                    color: "#1F2937",
                    margin: "8px 0 2px",
                  }}
                >
                  $
                  {avgPythonCost.totalMo.toLocaleString(undefined, {
                    maximumFractionDigits: 0,
                  })}
                  <span
                    style={{ fontSize: 13, color: "#6B7280", fontWeight: 500 }}
                  >
                    /mo
                  </span>
                </div>
                <div
                  style={{ fontSize: 12, color: "#6B7280", marginBottom: 12 }}
                >
                  {avgPythonCost.count}× {avgPythonCost.name} ·{" "}
                  {avgPythonCost.ram} GB RAM
                </div>
                <div
                  style={{
                    display: "grid",
                    gridTemplateColumns: "1fr 1fr",
                    gap: 8,
                  }}
                >
                  <div
                    style={{
                      background: "#F9FAFB",
                      borderRadius: 5,
                      padding: "8px 10px",
                      border: "1px solid #D1D5DB",
                    }}
                  >
                    <div
                      style={{
                        fontSize: 10,
                        color: "#6B7280",
                        fontFamily: "'IBM Plex Mono', monospace",
                        fontWeight: 500,
                      }}
                    >
                      MEMORY
                    </div>
                    <div
                      style={{
                        fontSize: 15,
                        fontWeight: 600,
                        color: "#1F2937",
                        marginTop: 1,
                      }}
                    >
                      {(avgPythonMem / 1024).toFixed(1)} GB
                    </div>
                  </div>
                  <div
                    style={{
                      background: "#F9FAFB",
                      borderRadius: 5,
                      padding: "8px 10px",
                      border: "1px solid #D1D5DB",
                    }}
                  >
                    <div
                      style={{
                        fontSize: 10,
                        color: "#6B7280",
                        fontFamily: "'IBM Plex Mono', monospace",
                        fontWeight: 500,
                      }}
                    >
                      HOURLY
                    </div>
                    <div
                      style={{
                        fontSize: 15,
                        fontWeight: 600,
                        color: "#1F2937",
                        marginTop: 1,
                      }}
                    >
                      ${avgPythonCost.totalHr.toFixed(2)}
                    </div>
                  </div>
                </div>
              </Card>
            </div>

            {/* Savings — two comparison points */}
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))",
                gap: 10,
                marginBottom: 24,
              }}
            >
              {/* vs best Python (most performance-equivalent) */}
              <div
                style={{
                  background: "#ECFDF5",
                  border: "1px solid #A7F3D0",
                  borderRadius: 6,
                  padding: "14px 18px",
                  display: "flex",
                  alignItems: "center",
                  gap: 14,
                  flexWrap: "wrap",
                }}
              >
                <div
                  style={{ fontSize: 34, fontWeight: 700, color: "#047857" }}
                >
                  {(
                    (1 - topRustCost.totalMo / bestPythonCost.totalMo) *
                    100
                  ).toFixed(0)}
                  %
                </div>
                <div>
                  <div
                    style={{ fontSize: 14, fontWeight: 600, color: "#1F2937" }}
                  >
                    {topRustFw} vs {bestPythonFw}
                  </div>
                  <div style={{ fontSize: 11, color: "#6B7280", marginTop: 1 }}>
                    Best Python by score · ~$
                    {(
                      bestPythonCost.totalMo - topRustCost.totalMo
                    ).toLocaleString(undefined, { maximumFractionDigits: 0 })}
                    /mo · ~$
                    {(
                      (bestPythonCost.totalMo - topRustCost.totalMo) *
                      12
                    ).toLocaleString(undefined, { maximumFractionDigits: 0 })}
                    /yr
                  </div>
                </div>
              </div>
              {/* vs avg Python (industry-average baseline) */}
              <div
                style={{
                  background: "#F0FDF4",
                  border: "1px solid #BBF7D0",
                  borderRadius: 6,
                  padding: "14px 18px",
                  display: "flex",
                  alignItems: "center",
                  gap: 14,
                  flexWrap: "wrap",
                }}
              >
                <div
                  style={{ fontSize: 34, fontWeight: 700, color: "#15803D" }}
                >
                  {(
                    (1 - topRustCost.totalMo / avgPythonCost.totalMo) *
                    100
                  ).toFixed(0)}
                  %
                </div>
                <div>
                  <div
                    style={{ fontSize: 14, fontWeight: 600, color: "#1F2937" }}
                  >
                    {topRustFw} vs Avg Python
                  </div>
                  <div style={{ fontSize: 11, color: "#6B7280", marginTop: 1 }}>
                    Industry-average baseline · ~$
                    {(
                      avgPythonCost.totalMo - topRustCost.totalMo
                    ).toLocaleString(undefined, { maximumFractionDigits: 0 })}
                    /mo · ~$
                    {(
                      (avgPythonCost.totalMo - topRustCost.totalMo) *
                      12
                    ).toLocaleString(undefined, { maximumFractionDigits: 0 })}
                    /yr
                  </div>
                </div>
              </div>
            </div>

            <h3
              style={{
                fontSize: 15,
                fontWeight: 600,
                color: "#1F2937",
                marginBottom: 10,
              }}
            >
              Monthly Cost by Framework
            </h3>
            <Card style={{ padding: 16, marginBottom: 24 }}>
              <ResponsiveContainer width="100%" height={320}>
                <BarChart data={costData} barCategoryGap="18%">
                  <CartesianGrid {...gridStyle} />
                  <XAxis
                    dataKey="name"
                    tick={xTickStyle}
                    axisLine={axisStyle}
                    tickLine={false}
                  />
                  <YAxis
                    tick={yTickStyle}
                    axisLine={axisStyle}
                    tickLine={false}
                    tickFormatter={(v) => `$${v.toLocaleString()}`}
                  />
                  <Tooltip content={<Tip />} />
                  <Bar
                    dataKey="monthly"
                    name="Monthly Cost ($)"
                    radius={[3, 3, 0, 0]}
                  >
                    {costData.map((d) => (
                      <Cell key={d.name} fill={COLORS[d.name]} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </Card>

            <h3
              style={{
                fontSize: 15,
                fontWeight: 600,
                color: "#1F2937",
                marginBottom: 10,
              }}
            >
              Instance Breakdown
            </h3>
            <Card style={{ overflowX: "auto" }}>
              <table style={{ width: "100%", borderCollapse: "collapse" }}>
                <thead>
                  <tr>
                    {[
                      "Framework",
                      "Type",
                      "Mem @10",
                      "Proj. Mem",
                      "Instance",
                      "Count",
                      "$/hr",
                      "$/mo",
                      "TP Cap. (RPS)",
                      "Daily Cap. (M req)",
                    ].map((h) => (
                      <th key={h} style={th}>
                        {h}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {[...RUST_FRAMEWORKS, ...PYTHON_FRAMEWORKS].map((n, i) => {
                    const f = FRAMEWORKS[n];
                    const ms = f.memory_peak_mb * SCALE;
                    const inst = bestEC2(ms / 1024);
                    return (
                      <tr
                        key={n}
                        style={{
                          background:
                            n === topRustFw
                              ? "#F4F7FB"
                              : i % 2 === 1
                                ? "#F9FAFB"
                                : "#fff",
                        }}
                      >
                        <td
                          style={{ ...td, fontWeight: 600, color: COLORS[n] }}
                        >
                          {n}
                        </td>
                        <td style={td}>
                          <Badge type={f.type} />
                        </td>
                        <td style={td}>
                          {f.memory_peak_mb.toLocaleString(undefined, {
                            maximumFractionDigits: 0,
                          })}{" "}
                          MB
                        </td>
                        <td style={td}>{(ms / 1024).toFixed(1)} GB</td>
                        <td
                          style={{
                            ...td,
                            fontFamily: "'IBM Plex Mono', monospace",
                            fontSize: 12,
                          }}
                        >
                          {inst.name}
                        </td>
                        <td style={td}>{inst.count}×</td>
                        <td style={td}>${inst.totalHr.toFixed(2)}</td>
                        <td
                          style={{ ...td, fontWeight: 600, color: "#1F2937" }}
                        >
                          $
                          {inst.totalMo.toLocaleString(undefined, {
                            maximumFractionDigits: 0,
                          })}
                        </td>
                        <td
                          style={{
                            ...td,
                            fontFamily: "'IBM Plex Mono', monospace",
                            fontSize: 12,
                          }}
                        >
                          {(f.throughput_rps * SCALE).toFixed(1)}
                        </td>
                        <td
                          style={{
                            ...td,
                            fontFamily: "'IBM Plex Mono', monospace",
                            fontSize: 12,
                            color: "#6B7280",
                          }}
                        >
                          {(
                            (f.throughput_rps * SCALE * 86_400) /
                            1_000_000
                          ).toFixed(1)}
                          M
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </Card>

            <div
              style={{
                marginTop: 12,
                padding: "10px 14px",
                background: "#F9FAFB",
                borderRadius: 5,
                border: "1px solid #D1D5DB",
                fontSize: 11,
                color: "#6B7280",
                fontFamily: "'IBM Plex Mono', monospace",
                lineHeight: 1.7,
              }}
            >
              * Proj. Mem = peak_mb × {SCALE} where {SCALE} ={" "}
              {PRODUCTION_TARGET} sessions ÷ {concurrency} concurrency. TP
              Cap. = throughput_rps × {SCALE} (sustained RPS at{" "}
              {PRODUCTION_TARGET} sessions). Daily Cap. = TP Cap. × 86,400
              (maximum requests per day this deployment can serve). EC2 r7g
              Graviton3 on-demand, US-East-1. Reserved instances reduce costs
              30–60%.
            </div>
          </>
        )}

        <div
          style={{
            marginTop: 36,
            paddingTop: 14,
            borderTop: "1px solid #D1D5DB",
            textAlign: "center",
          }}
        >
          <span
            style={{
              fontSize: 11,
              color: "#6B7280",
              fontFamily: "'IBM Plex Mono', monospace",
            }}
          >
            AI Agent Framework Benchmark · {fwNames.length} frameworks ·{" "}
            {totalRequests} requests × {concurrency} concurrency · {successRate}
            % success rate
          </span>
        </div>
      </div>
    </div>
  );
}

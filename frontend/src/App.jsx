import { useState, useEffect, useCallback } from "react";
import { useDropzone } from "react-dropzone";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  ScatterChart, Scatter, Cell, PieChart, Pie, Legend,
} from "recharts";

const API = "http://localhost:5000/api";

// ── Colour tokens ─────────────────────────────────────────────────────────────
const C = {
  bg: "#040d1a",
  surface: "#071426",
  border: "#0e2744",
  accent: "#00e5ff",
  high: "#ff3b5c",
  medium: "#f59e0b",
  low: "#10d97e",
  text: "#e2f0ff",
  muted: "#4a6785",
};

const riskColor = (r) =>
  r === "HIGH" ? C.high : r === "MEDIUM" ? C.medium : C.low;

// ── Subcomponents ────────────────────────────────────────────────────────────
function StatCard({ label, value, color, sub }) {
  return (
    <div style={{
      background: C.surface, border: `1px solid ${C.border}`,
      borderTop: `3px solid ${color}`, borderRadius: 10,
      padding: "20px 24px", flex: 1,
    }}>
      <div style={{ fontSize: 13, color: C.muted, letterSpacing: "0.08em", textTransform: "uppercase" }}>{label}</div>
      <div style={{ fontSize: 40, fontWeight: 700, color, fontFamily: "monospace", marginTop: 4 }}>{value}</div>
      {sub && <div style={{ fontSize: 12, color: C.muted, marginTop: 4 }}>{sub}</div>}
    </div>
  );
}

function RiskBadge({ level }) {
  return (
    <span style={{
      background: riskColor(level) + "22", color: riskColor(level),
      border: `1px solid ${riskColor(level)}44`,
      borderRadius: 4, padding: "2px 10px", fontSize: 11,
      fontWeight: 700, letterSpacing: "0.06em",
    }}>{level}</span>
  );
}

function CustomerRow({ c, rank }) {
  const pct = Math.round(c.churn_probability * 100);
  return (
    <tr style={{ borderBottom: `1px solid ${C.border}` }}>
      <td style={{ padding: "12px 16px", color: C.muted, fontFamily: "monospace" }}>#{rank}</td>
      <td style={{ padding: "12px 16px", fontFamily: "monospace", color: C.text }}>{c.customer_id}</td>
      <td style={{ padding: "12px 16px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <div style={{
            width: 100, height: 6, background: C.border, borderRadius: 3, overflow: "hidden"
          }}>
            <div style={{
              width: `${pct}%`, height: "100%",
              background: `linear-gradient(90deg, ${riskColor(c.risk_level)}, ${riskColor(c.risk_level)}aa)`,
              borderRadius: 3,
            }} />
          </div>
          <span style={{ fontFamily: "monospace", fontWeight: 700, color: riskColor(c.risk_level) }}>
            {pct}%
          </span>
        </div>
      </td>
      <td style={{ padding: "12px 16px" }}><RiskBadge level={c.risk_level} /></td>
      <td style={{ padding: "12px 16px", color: C.muted, fontSize: 13 }}>{c.top_shap_feature}</td>
    </tr>
  );
}

// ── Upload zone ───────────────────────────────────────────────────────────────
function UploadZone({ onData }) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const onDrop = useCallback(async (files) => {
    const file = files[0];
    setLoading(true); setError(null);
    const fd = new FormData();
    fd.append("file", file);
    try {
      const res = await fetch(`${API}/predict`, { method: "POST", body: fd });
      const data = await res.json();
      if (data.error) throw new Error(data.error);
      onData(data);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, [onData]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop, accept: { "text/csv": [".csv"] }, multiple: false,
  });

  return (
    <div {...getRootProps()} style={{
      border: `2px dashed ${isDragActive ? C.accent : C.border}`,
      borderRadius: 12, padding: "40px 32px", textAlign: "center",
      cursor: "pointer", background: isDragActive ? "#00e5ff08" : C.surface,
      transition: "all 0.2s",
    }}>
      <input {...getInputProps()} />
      <div style={{ fontSize: 32, marginBottom: 12 }}>📂</div>
      {loading ? (
        <div style={{ color: C.accent }}>Analysing customers…</div>
      ) : (
        <>
          <div style={{ color: C.text, fontWeight: 600 }}>
            {isDragActive ? "Drop your CSV here" : "Upload customer CSV"}
          </div>
          <div style={{ color: C.muted, fontSize: 13, marginTop: 6 }}>
            Must match Telco Churn schema (Kaggle format)
          </div>
        </>
      )}
      {error && <div style={{ color: C.high, marginTop: 10, fontSize: 13 }}>{error}</div>}
    </div>
  );
}

// ── Main dashboard ────────────────────────────────────────────────────────────
export default function App() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [tab, setTab] = useState("overview");

  // Load demo data on mount
  useEffect(() => {
    fetch(`${API}/demo`)
      .then(r => r.json())
      .then(d => { setData(d); setLoading(false); })
      .catch(() => setLoading(false));
  }, []);

  const handleExportPDF = async () => {
    const res = await fetch(`${API}/export-pdf`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ customers: data.customers }),
    });
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url; a.download = "churn_report.pdf"; a.click();
  };

  const shapData = data
    ? Object.entries(data.global_shap_importance)
        .slice(0, 8)
        .map(([name, value]) => ({ name: name.split(" ")[0], fullName: name, value }))
    : [];

  const pieData = data
    ? [
        { name: "High Risk", value: data.high_risk, fill: C.high },
        { name: "Medium Risk", value: data.medium_risk, fill: C.medium },
        { name: "Low Risk", value: data.low_risk, fill: C.low },
      ]
    : [];

  return (
    <div style={{
      minHeight: "100vh", background: C.bg, color: C.text,
      fontFamily: "'IBM Plex Mono', 'Fira Code', monospace",
      padding: "0 0 60px",
    }}>
      {/* Header */}
      <div style={{
        borderBottom: `1px solid ${C.border}`,
        padding: "20px 40px", display: "flex",
        alignItems: "center", justifyContent: "space-between",
      }}>
        <div>
          <span style={{
            fontSize: 22, fontWeight: 700, letterSpacing: "0.04em",
            color: C.accent,
          }}>RetainIQ</span>
          <span style={{ fontSize: 13, color: C.muted, marginLeft: 14 }}>
            Churn Prediction Engine
          </span>
        </div>
        <div style={{ display: "flex", gap: 12, alignItems: "center" }}>
          {data && (
            <span style={{
              background: C.low + "22", color: C.low, border: `1px solid ${C.low}44`,
              borderRadius: 6, padding: "4px 14px", fontSize: 12,
            }}>
              Model accuracy: {(data.model_accuracy * 100).toFixed(1)}%
            </span>
          )}
          {data && (
            <button onClick={handleExportPDF} style={{
              background: C.accent + "18", color: C.accent,
              border: `1px solid ${C.accent}44`, borderRadius: 6,
              padding: "6px 18px", cursor: "pointer", fontSize: 13,
              fontFamily: "inherit",
            }}>
              Export PDF Report
            </button>
          )}
        </div>
      </div>

      <div style={{ padding: "32px 40px" }}>
        {/* Upload zone */}
        <UploadZone onData={setData} />

        {loading && (
          <div style={{ textAlign: "center", color: C.muted, marginTop: 40 }}>
            Loading demo data…
          </div>
        )}

        {data && (
          <>
            {/* Stat cards */}
            <div style={{ display: "flex", gap: 16, marginTop: 32 }}>
              <StatCard label="Total Customers" value={data.total_customers}
                color={C.accent} sub="analysed in this batch" />
              <StatCard label="High Risk" value={data.high_risk}
                color={C.high} sub="churn probability > 70%" />
              <StatCard label="Medium Risk" value={data.medium_risk}
                color={C.medium} sub="40–70% churn probability" />
              <StatCard label="Low Risk" value={data.low_risk}
                color={C.low} sub="below 40% probability" />
            </div>

            {/* Tab nav */}
            <div style={{ display: "flex", gap: 0, marginTop: 32, borderBottom: `1px solid ${C.border}` }}>
              {["overview", "at-risk", "shap"].map(t => (
                <button key={t} onClick={() => setTab(t)} style={{
                  background: "none", border: "none", cursor: "pointer",
                  padding: "10px 24px", fontFamily: "inherit",
                  color: tab === t ? C.accent : C.muted,
                  borderBottom: tab === t ? `2px solid ${C.accent}` : "2px solid transparent",
                  fontSize: 13, letterSpacing: "0.06em", textTransform: "capitalize",
                  marginBottom: -1,
                }}>
                  {t === "shap" ? "SHAP Explainability" : t === "at-risk" ? "Top At-Risk" : "Overview"}
                </button>
              ))}
            </div>

            {/* Overview tab */}
            {tab === "overview" && (
              <div style={{ display: "flex", gap: 24, marginTop: 28 }}>
                {/* Risk distribution pie */}
                <div style={{
                  flex: 1, background: C.surface, border: `1px solid ${C.border}`,
                  borderRadius: 10, padding: 24,
                }}>
                  <div style={{ fontSize: 14, color: C.muted, marginBottom: 16 }}>
                    Risk Distribution
                  </div>
                  <ResponsiveContainer width="100%" height={220}>
                    <PieChart>
                      <Pie data={pieData} dataKey="value" nameKey="name"
                        cx="50%" cy="50%" outerRadius={85} strokeWidth={0}>
                        {pieData.map((entry, i) => (
                          <Cell key={i} fill={entry.fill} />
                        ))}
                      </Pie>
                      <Tooltip
                        contentStyle={{ background: C.surface, border: `1px solid ${C.border}`, borderRadius: 6 }}
                        labelStyle={{ color: C.text }}
                      />
                      <Legend formatter={(v) => <span style={{ color: C.muted, fontSize: 12 }}>{v}</span>} />
                    </PieChart>
                  </ResponsiveContainer>
                </div>

                {/* Churn probability histogram */}
                <div style={{
                  flex: 2, background: C.surface, border: `1px solid ${C.border}`,
                  borderRadius: 10, padding: 24,
                }}>
                  <div style={{ fontSize: 14, color: C.muted, marginBottom: 16 }}>
                    Churn Score Distribution
                  </div>
                  <ResponsiveContainer width="100%" height={220}>
                    <BarChart data={(() => {
                      const buckets = Array.from({ length: 10 }, (_, i) => ({
                        range: `${i * 10}–${i * 10 + 10}%`,
                        count: 0,
                        fill: i >= 7 ? C.high : i >= 4 ? C.medium : C.low,
                      }));
                      data.customers.forEach(c => {
                        const b = Math.min(Math.floor(c.churn_probability * 10), 9);
                        buckets[b].count++;
                      });
                      return buckets;
                    })()}>
                      <XAxis dataKey="range" tick={{ fill: C.muted, fontSize: 10 }} />
                      <YAxis tick={{ fill: C.muted, fontSize: 10 }} />
                      <Tooltip
                        contentStyle={{ background: C.surface, border: `1px solid ${C.border}`, borderRadius: 6 }}
                        cursor={{ fill: C.border }}
                      />
                      <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                        {Array.from({ length: 10 }, (_, i) => (
                          <Cell key={i} fill={i >= 7 ? C.high : i >= 4 ? C.medium : C.low} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            )}

            {/* At-risk customers tab */}
            {tab === "at-risk" && (
              <div style={{
                marginTop: 28, background: C.surface,
                border: `1px solid ${C.border}`, borderRadius: 10, overflow: "hidden",
              }}>
                <div style={{ padding: "16px 20px", borderBottom: `1px solid ${C.border}` }}>
                  <span style={{ fontSize: 14, color: C.muted }}>
                    Top 20 highest churn risk customers
                  </span>
                </div>
                <table style={{ width: "100%", borderCollapse: "collapse" }}>
                  <thead>
                    <tr style={{ background: C.bg }}>
                      {["#", "Customer ID", "Churn Risk", "Level", "Top Risk Factor"].map(h => (
                        <th key={h} style={{
                          padding: "10px 16px", textAlign: "left",
                          fontSize: 11, color: C.muted, letterSpacing: "0.07em",
                          textTransform: "uppercase", fontWeight: 500,
                        }}>{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {data.customers.slice(0, 20).map((c, i) => (
                      <CustomerRow key={c.customer_id} c={c} rank={i + 1} />
                    ))}
                  </tbody>
                </table>
              </div>
            )}

            {/* SHAP tab */}
            {tab === "shap" && (
              <div style={{ marginTop: 28 }}>
                <div style={{ marginBottom: 14, color: C.muted, fontSize: 13 }}>
                  Global feature importance — average |SHAP value| across all customers.
                  Higher = stronger influence on churn prediction.
                </div>
                <div style={{
                  background: C.surface, border: `1px solid ${C.border}`,
                  borderRadius: 10, padding: 24,
                }}>
                  <ResponsiveContainer width="100%" height={340}>
                    <BarChart data={shapData} layout="vertical" margin={{ left: 20 }}>
                      <XAxis type="number" tick={{ fill: C.muted, fontSize: 11 }} />
                      <YAxis type="category" dataKey="name" tick={{ fill: C.text, fontSize: 12 }} width={120} />
                      <Tooltip
                        formatter={(v, _, p) => [v.toFixed(4), p.payload.fullName]}
                        contentStyle={{ background: C.surface, border: `1px solid ${C.border}`, borderRadius: 6 }}
                        cursor={{ fill: C.border }}
                      />
                      <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                        {shapData.map((_, i) => (
                          <Cell key={i} fill={`hsl(${195 - i * 14}, 90%, 60%)`} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>

                {/* Per-customer SHAP for top at-risk */}
                <div style={{ marginTop: 24 }}>
                  <div style={{ fontSize: 14, color: C.muted, marginBottom: 14 }}>
                    Top at-risk customer — individual SHAP breakdown
                  </div>
                  {data.customers[0] && (
                    <div style={{
                      background: C.surface, border: `1px solid ${C.border}`,
                      borderRadius: 10, padding: 24,
                    }}>
                      <div style={{ marginBottom: 12 }}>
                        <span style={{ fontFamily: "monospace", color: C.accent }}>
                          {data.customers[0].customer_id}
                        </span>
                        <span style={{ color: C.muted, marginLeft: 12, fontSize: 13 }}>
                          Churn probability: {(data.customers[0].churn_probability * 100).toFixed(1)}%
                        </span>
                      </div>
                      <ResponsiveContainer width="100%" height={260}>
                        <BarChart
                          layout="vertical"
                          data={Object.entries(data.customers[0].shap_values)
                            .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
                            .slice(0, 8)
                            .map(([name, value]) => ({ name: name.split(" ")[0], value }))}
                          margin={{ left: 20 }}
                        >
                          <XAxis type="number" tick={{ fill: C.muted, fontSize: 11 }} />
                          <YAxis type="category" dataKey="name" tick={{ fill: C.text, fontSize: 12 }} width={120} />
                          <Tooltip
                            contentStyle={{ background: C.surface, border: `1px solid ${C.border}`, borderRadius: 6 }}
                            cursor={{ fill: C.border }}
                          />
                          <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                            {Object.entries(data.customers[0].shap_values)
                              .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
                              .slice(0, 8)
                              .map(([, v], i) => (
                                <Cell key={i} fill={v > 0 ? C.high : C.low} />
                              ))}
                          </Bar>
                        </BarChart>
                      </ResponsiveContainer>
                      <div style={{ fontSize: 11, color: C.muted, marginTop: 8 }}>
                        🔴 Positive SHAP = increases churn risk &nbsp;|&nbsp; 🟢 Negative = decreases churn risk
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}

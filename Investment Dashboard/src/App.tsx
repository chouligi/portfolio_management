import { useEffect, useMemo, useState } from "react";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Pie,
  PieChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  ArrowDownRight,
  ArrowUpRight,
  CheckCircle2,
  CircleDollarSign,
  Clock3,
  Database,
  Layers3,
  RefreshCw,
  Search,
  ShieldCheck,
  WalletCards,
} from "lucide-react";
import type { Broker, DashboardData, Holding } from "./types";

const colors = ["#8ca58d", "#c4a47d", "#788ca0", "#b38f8c", "#a7a287", "#718c84", "#9d91a9"];
const euro = new Intl.NumberFormat("en-IE", {
  style: "currency",
  currency: "EUR",
  maximumFractionDigits: 0,
});
const euroPrecise = new Intl.NumberFormat("en-IE", {
  style: "currency",
  currency: "EUR",
  minimumFractionDigits: 2,
  maximumFractionDigits: 2,
});
const percent = new Intl.NumberFormat("en-IE", { style: "percent", maximumFractionDigits: 2 });
const compact = new Intl.NumberFormat("en-IE", { notation: "compact", maximumFractionDigits: 1 });
const tooltipNumber = (value: unknown) => Number(Array.isArray(value) ? value[0] : value ?? 0);
const tooltipEuro = (value: unknown) => euro.format(tooltipNumber(value));
const tooltipPercent = (value: unknown) => percent.format(tooltipNumber(value));
const tooltipSignedPercent = (value: unknown) => {
  const number = tooltipNumber(value);
  return `${number > 0 ? "+" : ""}${percent.format(number)}`;
};

function metricTone(value: number) {
  return value >= 0 ? "positive" : "negative";
}

function Kpi({
  icon,
  label,
  value,
  detail,
  tone,
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
  detail: string;
  tone?: "positive" | "negative";
}) {
  return (
    <article className="kpi-card">
      <div className="kpi-top">
        <span className="icon-box">{icon}</span>
        <span className="eyebrow">{label}</span>
      </div>
      <strong className={tone ? `kpi-value ${tone}` : "kpi-value"}>{value}</strong>
      <span className="muted">{detail}</span>
    </article>
  );
}

function StatusPill({ children, warning = false }: { children: React.ReactNode; warning?: boolean }) {
  return <span className={warning ? "status warning" : "status"}>{children}</span>;
}

export default function App() {
  const [data, setData] = useState<DashboardData | null>(null);
  const [broker, setBroker] = useState<Broker>("all");
  const [query, setQuery] = useState("");
  const [activePage, setActivePage] = useState("Overview");

  const navigate = (page: string) => {
    setActivePage(page);
    document.getElementById(page.toLowerCase())?.scrollIntoView({ behavior: "smooth", block: "start" });
  };

  useEffect(() => {
    fetch("/data/dashboard.json")
      .then((response) => {
        if (!response.ok) throw new Error("Dashboard data has not been generated.");
        return response.json();
      })
      .then(setData)
      .catch(() => setData(null));
  }, []);

  const holdings = useMemo(() => {
    if (!data) return [];
    const selected = data.holdings.map((holding) => {
      const quantity =
        broker === "ibkr"
          ? holding.ibkrQuantity
          : broker === "degiro"
            ? holding.degiroQuantity
            : holding.quantity;
      return { ...holding, quantity, value: quantity * holding.price };
    });
    const total = selected.reduce((sum, holding) => sum + holding.value, 0);
    return selected
      .map((holding) => ({ ...holding, weight: total ? holding.value / total : 0 }))
      .filter(
        (holding) =>
          holding.quantity > 0 &&
          `${holding.ticker} ${holding.name} ${holding.category}`.toLowerCase().includes(query.toLowerCase()),
      );
  }, [broker, data, query]);

  if (!data) {
    return (
      <main className="empty-state">
        <Database size={32} />
        <h1>Generate dashboard data</h1>
        <p>Run <code>npm run data</code> from the Investment Dashboard directory, then refresh.</p>
      </main>
    );
  }

  const filteredValue = holdings.reduce((sum, holding) => sum + holding.value, 0);
  const contributionChart = data.contributionHistory.map((item) => ({
    ...item,
    label: new Date(item.date).toLocaleDateString("en-GB", { month: "short", year: "2-digit" }),
  }));
  const brokersReconciled = Boolean(data.ibkr.flex?.positionsReconciled && data.degiro?.positionsReconciled);

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark">GC</span>
          <div>
            <strong>Portfolio Monitor</strong>
            <span>Global ETF strategy</span>
          </div>
        </div>
        <nav>
          {["Overview", "Holdings", "Performance"].map((page) => (
            <button key={page} className={activePage === page ? "nav-link active" : "nav-link"} onClick={() => navigate(page)}>
              {page}
            </button>
          ))}
        </nav>
        <div className="top-actions">
          <StatusPill>
            <ShieldCheck size={14} /> Local & private
          </StatusPill>
          <span className="updated">
            <Clock3 size={15} /> {new Date(data.generatedAt).toLocaleString("en-GB", { dateStyle: "medium", timeStyle: "short" })}
          </span>
        </div>
      </header>

      <main>
        <section className="hero" id="overview">
          <div>
            <span className="overline">Combined ETF portfolio</span>
            <h1>A calm view of long-term capital.</h1>
            <p>Monitoring allocation, contributions and performance across Interactive Brokers and Degiro.</p>
          </div>
          <div className="hero-status">
            {brokersReconciled ? <CheckCircle2 size={18} /> : <RefreshCw size={18} />}
            <div>
              <strong>{brokersReconciled ? "Broker records verified" : "Position data available"}</strong>
              <span>
                {brokersReconciled
                  ? "IBKR and Degiro agree with portfolio records"
                  : data.ibkr.flex
                    ? `IBKR statement through ${data.ibkr.flex.reportDate.slice(0, 4)}-${data.ibkr.flex.reportDate.slice(4, 6)}-${data.ibkr.flex.reportDate.slice(6, 8)}`
                  : `Spreadsheet updated ${data.sourceUpdatedAt}`}
              </span>
            </div>
          </div>
        </section>

        <section className="kpi-grid">
          <Kpi icon={<WalletCards size={18} />} label="Portfolio value" value={euro.format(data.summary.currentValue)} detail="Combined current holdings" />
          <Kpi icon={<CircleDollarSign size={18} />} label="Invested capital" value={euro.format(data.summary.investedCapital)} detail="Recorded contributions" />
          <Kpi
            icon={data.summary.profitLoss >= 0 ? <ArrowUpRight size={18} /> : <ArrowDownRight size={18} />}
            label="Total gain"
            value={euro.format(data.summary.profitLoss)}
            detail={`${percent.format(data.summary.profitLossPercent)} simple return`}
            tone={metricTone(data.summary.profitLoss)}
          />
          <Kpi icon={<Layers3 size={18} />} label="Portfolio structure" value={`${percent.format(data.summary.equityValue / data.summary.currentValue)} equities`} detail={`${percent.format(data.summary.bondValue / data.summary.currentValue)} bonds`} />
        </section>

        <section className="dashboard-grid">
          <article className="panel allocation-panel">
            <div className="panel-heading">
              <div>
                <span className="eyebrow">Allocation</span>
                <h2>Current mix</h2>
              </div>
              <StatusPill>Combined portfolio</StatusPill>
            </div>
            <div className="allocation-content">
              <div className="donut-wrap">
                <ResponsiveContainer width="100%" height={280}>
                  <PieChart>
                    <Pie data={holdings} dataKey="value" nameKey="ticker" innerRadius={78} outerRadius={112} paddingAngle={2}>
                      {holdings.map((holding, index) => <Cell key={holding.ticker} fill={colors[index % colors.length]} />)}
                    </Pie>
                    <Tooltip formatter={tooltipEuro} />
                  </PieChart>
                </ResponsiveContainer>
                <div className="donut-center">
                  <strong>{euro.format(filteredValue)}</strong>
                  <span>{broker === "all" ? "All brokers" : broker === "ibkr" ? "IBKR" : "Degiro"}</span>
                </div>
              </div>
              <div className="allocation-list">
                {holdings.map((holding, index) => (
                  <div className="allocation-row" key={holding.ticker}>
                    <span className="dot" style={{ background: colors[index % colors.length] }} />
                    <strong>{holding.ticker}</strong>
                    <span>{holding.category}</span>
                    <b>{percent.format(holding.weight)}</b>
                  </div>
                ))}
              </div>
            </div>
          </article>

          <article className="panel target-panel">
            <div className="panel-heading">
              <div>
                <span className="eyebrow">Strategy alignment</span>
                <h2>Difference from target</h2>
              </div>
              <StatusPill>Overweight / underweight</StatusPill>
            </div>
            <ResponsiveContainer width="100%" height={325}>
              <BarChart data={data.holdings} layout="vertical" margin={{ left: 0, right: 20 }}>
                <CartesianGrid stroke="#e9e6df" horizontal={false} />
                <XAxis
                  type="number"
                  domain={[-0.02, 0.02]}
                  ticks={[-0.02, -0.01, 0, 0.01, 0.02]}
                  tickFormatter={(value) => `${value > 0 ? "+" : ""}${(value * 100).toFixed(0)}%`}
                  axisLine={false}
                  tickLine={false}
                />
                <YAxis type="category" dataKey="ticker" axisLine={false} tickLine={false} width={48} />
                <Tooltip formatter={tooltipSignedPercent} />
                <ReferenceLine x={0} stroke="#8f938f" strokeWidth={1.5} />
                <Bar dataKey="divergence" name="Difference from target" radius={[4, 4, 4, 4]}>
                  {data.holdings.map((holding) => (
                    <Cell
                      key={holding.ticker}
                      fill={Math.abs(holding.divergence) < 0.005 ? "#718c84" : holding.divergence > 0 ? "#b38f8c" : "#c4a47d"}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
            <div className="divergence-legend">
              <span><i className="underweight-dot" /> Underweight</span>
              <span><i className="target-dot" /> Near target</span>
              <span><i className="overweight-dot" /> Overweight</span>
            </div>
          </article>
        </section>

        <section className="panel holdings-panel" id="holdings">
          <div className="panel-heading holdings-heading">
            <div>
              <span className="eyebrow">Positions</span>
              <h2>ETF holdings</h2>
            </div>
            <div className="table-controls">
              <label className="search">
                <Search size={16} />
                <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search holdings" />
              </label>
              <div className="segmented">
                {(["all", "ibkr", "degiro"] as Broker[]).map((item) => (
                  <button key={item} className={broker === item ? "active" : ""} onClick={() => setBroker(item)}>
                    {item === "all" ? "All brokers" : item === "ibkr" ? "IBKR" : "Degiro"}
                  </button>
                ))}
              </div>
            </div>
          </div>
          <div className="table-scroll">
            <table>
              <thead>
                <tr>
                  <th>Fund</th>
                  <th>Category</th>
                  <th className="numeric">Price</th>
                  <th className="numeric">Quantity</th>
                  <th className="numeric">Value</th>
                  <th className="numeric">Weight</th>
                  <th className="numeric">Target</th>
                  <th>Alignment</th>
                </tr>
              </thead>
              <tbody>
                {holdings.map((holding) => <HoldingRow key={holding.ticker} holding={holding} />)}
              </tbody>
            </table>
          </div>
        </section>

        <section className="dashboard-grid performance-grid" id="performance">
          <article className="panel">
            <div className="panel-heading">
              <div>
                <span className="eyebrow">Capital deployed</span>
                <h2>Cumulative contributions</h2>
              </div>
              <StatusPill>Contribution history</StatusPill>
            </div>
            <ResponsiveContainer width="100%" height={300}>
              <AreaChart data={contributionChart}>
                <defs>
                  <linearGradient id="capital" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#607e73" stopOpacity={0.35} />
                    <stop offset="95%" stopColor="#607e73" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid stroke="#e9e6df" vertical={false} />
                <XAxis dataKey="label" minTickGap={34} axisLine={false} tickLine={false} />
                <YAxis tickFormatter={(value) => compact.format(value)} axisLine={false} tickLine={false} width={54} />
                <Tooltip formatter={tooltipEuro} />
                <Area type="monotone" dataKey="cumulative" name="Contributed" stroke="#263a34" strokeWidth={2} fill="url(#capital)" />
              </AreaChart>
            </ResponsiveContainer>
          </article>

          <article className="panel">
            <div className="panel-heading">
              <div>
                <span className="eyebrow">Broker history</span>
                <h2>Annual capital activity</h2>
              </div>
              <StatusPill>Both brokers</StatusPill>
            </div>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={data.brokerHistory}>
                <CartesianGrid stroke="#e9e6df" vertical={false} />
                <XAxis dataKey="year" axisLine={false} tickLine={false} />
                <YAxis tickFormatter={(value) => compact.format(value)} axisLine={false} tickLine={false} width={54} />
                <Tooltip formatter={tooltipEuro} />
                <Legend />
                <Bar dataKey="ibkrEndingValue" name="IBKR year-end value" fill="#263a34" radius={[5, 5, 0, 0]} />
                <Bar dataKey="ibkrDeposits" name="IBKR deposits" fill="#c7b89d" radius={[5, 5, 0, 0]} />
                <Bar dataKey="degiroPurchases" name="Degiro purchases" fill="#788ca0" radius={[5, 5, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </article>
        </section>

        <section className="quality-strip">
          <div>
            <RefreshCw size={18} />
            <div><strong>Valuation data</strong><span>Portfolio prices and holdings</span></div>
          </div>
          <div>
            <Database size={18} />
            <div>
              <strong>Interactive Brokers</strong>
              <span>{data.ibkr.flex?.positionsReconciled ? "Positions verified" : "Position review required"}</span>
            </div>
          </div>
          <div>
            <Clock3 size={18} />
            <div>
              <strong>Degiro</strong>
              <span>
                {data.degiro?.positionsReconciled
                  ? "Transaction history verified"
                  : "Position review required"}
              </span>
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}

function HoldingRow({ holding }: { holding: Holding }) {
  const gap = holding.weight - holding.targetWeight;
  const aligned = Math.abs(gap) < 0.005;
  return (
    <tr>
      <td>
        <div className="fund-cell">
          <span className="ticker">{holding.ticker}</span>
          <span>{holding.name}</span>
        </div>
      </td>
      <td><span className="category">{holding.category}</span></td>
      <td className="numeric">{euroPrecise.format(holding.price)}</td>
      <td className="numeric">{holding.quantity.toLocaleString("en-IE")}</td>
      <td className="numeric strong">{euro.format(holding.value)}</td>
      <td className="numeric">{percent.format(holding.weight)}</td>
      <td className="numeric muted-cell">{percent.format(holding.targetWeight)}</td>
      <td>
        <span className={aligned ? "alignment aligned" : gap > 0 ? "alignment over" : "alignment under"}>
          {aligned ? "On target" : `${gap > 0 ? "+" : ""}${percent.format(gap)}`}
        </span>
      </td>
    </tr>
  );
}

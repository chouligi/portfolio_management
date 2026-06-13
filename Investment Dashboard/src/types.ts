export type Broker = "all" | "ibkr" | "degiro";

export interface Holding {
  ticker: string;
  name: string;
  category: string;
  price: number;
  degiroQuantity: number;
  ibkrQuantity: number;
  quantity: number;
  value: number;
  weight: number;
  targetWeight: number;
  divergence: number;
}

export interface DashboardData {
  generatedAt: string;
  sourceUpdatedAt: string;
  summary: {
    currentValue: number;
    investedCapital: number;
    profitLoss: number;
    profitLossPercent: number;
    ibkrValue: number;
    degiroValue: number;
    equityValue: number;
    bondValue: number;
  };
  holdings: Holding[];
  contributions: { date: string; amount: number }[];
  contributionHistory: { date: string; amount: number; cumulative: number }[];
  brokerHistory: {
    year: number;
    ibkrEndingValue: number;
    ibkrDeposits: number;
    degiroPurchases: number;
  }[];
  ibkr: {
    annual: {
      year: number;
      startingValue: number;
      endingValue: number;
      deposits: number;
      dividends: number;
      fees: number;
      netChange: number;
    }[];
    historyCompleteThrough: number;
    flex: {
      reportDate: string;
      generatedAt: string;
      endingValue: number;
      positionsReconciled: boolean;
      positionMismatches: {
        ticker: string;
        sheetQuantity: number | null;
        flexQuantity: number;
      }[];
    } | null;
  };
  degiro: {
    firstTradeDate: string;
    lastTradeDate: string;
    transactionCount: number;
    totalPurchases: number;
    totalFees: number;
    positionsReconciled: boolean;
    positionMismatches: {
      ticker: string;
      sheetQuantity: number | null;
      transactionQuantity: number;
    }[];
    annual: {
      year: number;
      purchases: number;
      fees: number;
      transactions: number;
    }[];
  } | null;
  dataQuality: Record<string, string>;
}

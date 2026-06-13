# Investment Dashboard

Local-only monitoring dashboard for the combined IBKR and Degiro ETF portfolio.

## Current Data Sources

- Google Sheets: current holdings, target allocations, total invested capital, and contribution history
- IBKR annual activity statements: historical account values, deposits, trades, dividends, and fees
- Degiro transaction export: historical purchases, fees, cost basis, and position reconciliation

Raw transactions and generated dashboard data are ignored by Git.

## Configure Automatic IBKR Flex Updates

The annual CSV files are the one-time historical source. The `Codex - Dashboard`
Year-to-Date Flex Query supplies ongoing IBKR activity automatically.

1. In IBKR Client Portal, open **Performance & Reports → Flex Queries**.
2. Open the information popover for `Codex - Dashboard` and note its Query ID.
3. Enable **Flex Web Service Configuration** and generate a token.
4. Configure the local secrets:

```bash
cp .env.example .env
```

Edit `.env` and fill in:

```text
IBKR_FLEX_QUERY_ID=your-query-id
IBKR_FLEX_TOKEN=your-secret-token
GOOGLE_SHEET_ID=your-private-spreadsheet-id
```

`.env`, the private spreadsheet ID, and downloaded Flex reports are ignored by
Git. The token provides reporting access, not trading access.

Test the Flex download independently:

```bash
npm run flex
```

Activity Flex data is updated by IBKR once daily after close of business.

## Run Locally

```bash
cd "Investment Dashboard"
npm run data
npm run dev
```

Open the local URL printed by Vite, normally `http://localhost:5173`.

`npm run data`:

1. Downloads the current YTD IBKR Flex XML when `.env` is configured.
2. Reads the private Google Sheet.
3. Parses the historical IBKR annual CSV files.
4. Parses and reconciles the Degiro transaction export.
5. Regenerates the ignored local dashboard payload.

When Flex credentials are not configured, the command skips that download and
continues using existing private statement files.

## Verify Production Build

```bash
npm run build
```

## Current Limitations

- Prices currently come from the Google Sheet; automatic market-price fetching is the next data-pipeline milestone.
- The Flex report verifies IBKR quantities and provides the latest IBKR account
  value. Current combined holdings continue to come from the Google Sheet.
- Exact daily combined portfolio performance is not yet reconstructed; the current
  dashboard shows contribution history and annual broker activity.

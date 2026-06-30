/*
 * Pharma1 analysis runner
 * -----------------------
 * Uses data/Pharma1.xlsx to estimate pharmaceutical tariff exposure, production
 * cost transmission, drug-price pressure, and limited short-run substitution.
 *
 * This script intentionally uses only Node built-ins so it can run in the
 * current project environment even when Python packages are unavailable.
 */

const fs = require('fs');
const path = require('path');
const zlib = require('zlib');

const REPO_ROOT = path.resolve(__dirname, '..', '..');
const DATA_DIR = path.join(REPO_ROOT, 'data');
const OUTPUT_DIR = path.join(REPO_ROOT, 'python_output');

const PHARMA1_PATH = path.join(DATA_DIR, 'Pharma1.xlsx');
const COUNTRY_LABELS_PATH = path.join(DATA_DIR, 'base_data', 'country_labels.csv');
const TARIFFS_PATH = path.join(DATA_DIR, 'base_data', 'tariffs.csv');

const PASS_THROUGH = 0.88;
const EPS_PHARMA = 2.3;
const IO_MULTIPLIER = 1.067;
const ICIO_INTERMEDIATE_IMPORT_SHARE = 0.1236;
const PROJECT_IMPORT_DEPENDENCY = 0.54;
const PROJECT_REFERENCE_IMPORTS_BN = 220.0;

const DRUG_SPEND_QUINTILES = [
  { group: 'Q1 (Lowest 20%)', baseSpend: 595, referenceExtra: 63, referenceBurdenPctIncome: 0.370 },
  { group: 'Q2', baseSpend: 946, referenceExtra: 100, referenceBurdenPctIncome: 0.232 },
  { group: 'Q3 (Middle)', baseSpend: 1080, referenceExtra: 114, referenceBurdenPctIncome: 0.158 },
  { group: 'Q4', baseSpend: 1008, referenceExtra: 106, referenceBurdenPctIncome: 0.095 },
  { group: 'Q5 (Top 20%)', baseSpend: 1190, referenceExtra: 126, referenceBurdenPctIncome: 0.053 },
];

const MONTH_COLS = [
  'January', 'February', 'March', 'April', 'May', 'June',
  'July', 'August', 'September', 'October', 'November', 'December',
];

const COUNTRY_ALIASES = new Map([
  ['czechia czech republic', 'czechia'],
  ['russia', 'russian federation'],
  ['slovakia', 'slovak republic'],
  ['saint lucia', 'st lucia'],
  ['myanmar burma', 'myanmar'],
]);

function ensureOutputDir() {
  fs.mkdirSync(OUTPUT_DIR, { recursive: true });
}

function xmlUnescape(s) {
  return String(s || '')
    .replace(/&amp;/g, '&')
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&quot;/g, '"')
    .replace(/&apos;/g, "'");
}

function normalizeName(s) {
  const key = String(s || '')
    .toLowerCase()
    .replace(/&/g, 'and')
    .replace(/[^a-z0-9]+/g, ' ')
    .trim();
  return COUNTRY_ALIASES.get(key) || key;
}

function parseCsv(text) {
  const rows = [];
  let row = [];
  let cell = '';
  let inQuotes = false;

  for (let i = 0; i < text.length; i += 1) {
    const ch = text[i];
    const next = text[i + 1];

    if (ch === '"' && inQuotes && next === '"') {
      cell += '"';
      i += 1;
    } else if (ch === '"') {
      inQuotes = !inQuotes;
    } else if (ch === ',' && !inQuotes) {
      row.push(cell);
      cell = '';
    } else if ((ch === '\n' || ch === '\r') && !inQuotes) {
      if (ch === '\r' && next === '\n') i += 1;
      row.push(cell);
      if (row.some((v) => v !== '')) rows.push(row);
      row = [];
      cell = '';
    } else {
      cell += ch;
    }
  }
  row.push(cell);
  if (row.some((v) => v !== '')) rows.push(row);
  return rows;
}

function readZipEntries(buffer) {
  let eocd = -1;
  for (let i = buffer.length - 22; i >= 0; i -= 1) {
    if (buffer.readUInt32LE(i) === 0x06054b50) {
      eocd = i;
      break;
    }
  }
  if (eocd < 0) throw new Error('Could not find XLSX central directory');

  const entryCount = buffer.readUInt16LE(eocd + 10);
  let cursor = buffer.readUInt32LE(eocd + 16);
  const out = new Map();

  for (let i = 0; i < entryCount; i += 1) {
    if (buffer.readUInt32LE(cursor) !== 0x02014b50) {
      throw new Error('Malformed XLSX central directory');
    }

    const method = buffer.readUInt16LE(cursor + 10);
    const compressedSize = buffer.readUInt32LE(cursor + 20);
    const nameLen = buffer.readUInt16LE(cursor + 28);
    const extraLen = buffer.readUInt16LE(cursor + 30);
    const commentLen = buffer.readUInt16LE(cursor + 32);
    const localHeaderOffset = buffer.readUInt32LE(cursor + 42);
    const name = buffer.slice(cursor + 46, cursor + 46 + nameLen).toString('utf8');

    const localNameLen = buffer.readUInt16LE(localHeaderOffset + 26);
    const localExtraLen = buffer.readUInt16LE(localHeaderOffset + 28);
    const start = localHeaderOffset + 30 + localNameLen + localExtraLen;
    let data = buffer.slice(start, start + compressedSize);
    if (method === 8) data = zlib.inflateRawSync(data);
    if (method !== 0 && method !== 8) throw new Error(`Unsupported ZIP compression method ${method}`);

    out.set(name, data.toString('utf8'));
    cursor += 46 + nameLen + extraLen + commentLen;
  }

  return out;
}

function colIndex(cellRef) {
  const letters = cellRef.match(/[A-Z]+/)[0];
  let n = 0;
  for (const ch of letters) n = n * 26 + ch.charCodeAt(0) - 64;
  return n - 1;
}

function cellValue(cellXml) {
  const type = (cellXml.match(/t="([^"]+)"/) || [])[1];
  if (type === 'inlineStr') {
    return xmlUnescape((cellXml.match(/<t[^>]*>([\s\S]*?)<\/t>/) || [])[1] || '');
  }
  const value = (cellXml.match(/<v>([\s\S]*?)<\/v>/) || [])[1];
  if (value == null) return '';
  return Number(value);
}

function readPharma1Rows() {
  const zip = readZipEntries(fs.readFileSync(PHARMA1_PATH));
  const sheet = zip.get('xl/worksheets/sheet2.xml');
  if (!sheet) throw new Error('Pharma1.xlsx is missing xl/worksheets/sheet2.xml');

  const table = [];
  for (const rowMatch of sheet.matchAll(/<row[^>]*r="(\d+)"[^>]*>([\s\S]*?)<\/row>/g)) {
    const row = [];
    for (const cellMatch of rowMatch[2].matchAll(/<c[^>]*r="([A-Z]+\d+)"[^>]*>[\s\S]*?<\/c>/g)) {
      row[colIndex(cellMatch[1])] = cellValue(cellMatch[0]);
    }
    table[Number(rowMatch[1]) - 1] = row;
  }

  return table
    .slice(3)
    .filter((row) => row && row[0] === 'Customs Value')
    .map((row) => ({
      country: row[1],
      hts: String(row[2]),
      description: row[3],
      year: Number(row[4]),
      total: MONTH_COLS.reduce((sum, _month, idx) => sum + (Number(row[5 + idx]) || 0), 0),
    }));
}

function readTariffMap() {
  const labels = parseCsv(fs.readFileSync(COUNTRY_LABELS_PATH, 'utf8')).slice(1);
  const tariffs = parseCsv(fs.readFileSync(TARIFFS_PATH, 'utf8')).slice(1);
  const map = new Map();

  labels.forEach((row, idx) => {
    map.set(normalizeName(row[2]), {
      iso3: row[0],
      countryName: row[2],
      tariff: Number(tariffs[idx]?.[0]),
    });
  });

  return map;
}

function groupBy(rows, keyFn) {
  const groups = new Map();
  for (const row of rows) {
    const key = keyFn(row);
    groups.set(key, (groups.get(key) || 0) + row.total);
  }
  return groups;
}

function hhi(shares) {
  return shares.reduce((sum, s) => sum + (s * 100) ** 2, 0);
}

function csvEscape(value) {
  const s = String(value ?? '');
  return /[",\n\r]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

function writeCsv(filePath, rows, headers) {
  const lines = [headers.join(',')];
  for (const row of rows) {
    lines.push(headers.map((h) => csvEscape(row[h])).join(','));
  }
  fs.writeFileSync(filePath, `${lines.join('\n')}\n`);
}

function svgEscape(value) {
  return String(value ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function pct(x, digits = 1) {
  return `${(x * 100).toFixed(digits)}%`;
}

function moneyBn(x, digits = 1) {
  return `$${(x / 1e9).toFixed(digits)}B`;
}

function run() {
  ensureOutputDir();
  const rows = readPharma1Rows();
  const tariffMap = readTariffMap();

  const byYear = [...groupBy(rows, (r) => r.year).entries()]
    .sort(([a], [b]) => a - b)
    .map(([year, total]) => ({ year, total }));

  const year = 2024;
  const rows2024 = rows.filter((r) => r.year === year);
  const total2024 = rows2024.reduce((sum, r) => sum + r.total, 0);
  const importDependency = PROJECT_IMPORT_DEPENDENCY * (total2024 / (PROJECT_REFERENCE_IMPORTS_BN * 1e9));

  const countryExposure = [...groupBy(rows2024, (r) => r.country).entries()]
    .map(([country, value]) => {
      const match = tariffMap.get(normalizeName(country));
      return {
        country,
        iso3: match?.iso3 || '',
        value,
        share: value / total2024,
        tariff: Number.isFinite(match?.tariff) ? match.tariff : null,
        matched: Boolean(match && Number.isFinite(match.tariff)),
      };
    })
    .sort((a, b) => b.value - a.value);

  const matchedValue = countryExposure
    .filter((r) => r.matched)
    .reduce((sum, r) => sum + r.value, 0);
  const tauEff = countryExposure
    .filter((r) => r.matched)
    .reduce((sum, r) => sum + r.value * r.tariff, 0) / matchedValue;

  const postDenom = countryExposure
    .filter((r) => r.matched)
    .reduce((sum, r) => sum + r.share * (1 + r.tariff) ** (-EPS_PHARMA), 0);
  for (const row of countryExposure) {
    if (!row.matched) {
      row.share_post = null;
      row.delta_share = null;
    } else {
      row.share_post = (row.share * (1 + row.tariff) ** (-EPS_PHARMA)) / postDenom;
      row.delta_share = row.share_post - row.share;
    }
  }

  const preHhi = hhi(countryExposure.filter((r) => r.matched).map((r) => r.share));
  const postHhi = hhi(countryExposure.filter((r) => r.matched).map((r) => r.share_post));

  const inputCostDrugPrice = tauEff * ICIO_INTERMEDIATE_IMPORT_SHARE * PASS_THROUGH * IO_MULTIPLIER * 100;
  const importDependencyDrugPrice = tauEff * importDependency * PASS_THROUGH * IO_MULTIPLIER * 100;
  const importVolumeChange = -EPS_PHARMA * tauEff / (1 + tauEff) * 100;
  const lowTariffShare = countryExposure
    .filter((r) => r.matched && r.tariff <= 0.10)
    .reduce((sum, r) => sum + r.share, 0);

  const htsExposure = [...groupBy(rows2024, (r) => r.hts).entries()]
    .sort((a, b) => b[1] - a[1])
    .map(([hts, value]) => ({ hts, value, share: value / total2024 }));

  const countryCsv = countryExposure.map((r, idx) => ({
    rank: idx + 1,
    country: r.country,
    iso3: r.iso3,
    import_value_usd: Math.round(r.value),
    import_share_pct: (r.share * 100).toFixed(4),
    tariff_pct: r.tariff == null ? '' : (r.tariff * 100).toFixed(2),
    post_tariff_share_pct: r.share_post == null ? '' : (r.share_post * 100).toFixed(4),
    delta_share_pp: r.delta_share == null ? '' : (r.delta_share * 100).toFixed(4),
  }));

  writeCsv(
    path.join(OUTPUT_DIR, 'pharma1_country_exposure_2024.csv'),
    countryCsv,
    ['rank', 'country', 'iso3', 'import_value_usd', 'import_share_pct', 'tariff_pct', 'post_tariff_share_pct', 'delta_share_pp'],
  );

  writeCsv(
    path.join(OUTPUT_DIR, 'pharma1_hts_exposure_2024.csv'),
    htsExposure.map((r) => ({
      hts: r.hts,
      import_value_usd: Math.round(r.value),
      import_value_bn: (r.value / 1e9).toFixed(3),
      import_share_pct: (r.share * 100).toFixed(4),
    })),
    ['hts', 'import_value_usd', 'import_value_bn', 'import_share_pct'],
  );

  const topCountries = countryExposure.slice(0, 15);
  const unmatched = countryExposure.filter((r) => !r.matched);

  const md = [
    '# Pharma1 Tariff Cost And Drug Price Analysis',
    '',
    'Objective: analyze how tariffs on imported pharmaceutical inputs increase production costs and drug prices when short-run substitution is limited.',
    '',
    '## Data',
    '',
    `- Source workbook: \`data/Pharma1.xlsx\`, sheet \`Query Results\`.`,
    `- Import measure: Customs Value, Imports For Consumption.`,
    `- Main baseline year: ${year}, the latest complete full year in the workbook.`,
    `- Rows used: ${rows2024.length.toLocaleString()} country-HTS-year rows.`,
    `- Total ${year} pharma imports: ${moneyBn(total2024)}.`,
    `- Tariff matching coverage: ${pct(matchedValue / total2024, 2)} of import value.`,
    unmatched.length
      ? `- Unmatched countries excluded from tariff weighting: ${unmatched.map((r) => `${r.country} (${moneyBn(r.value, 3)})`).join(', ')}.`
      : '- All countries matched to the tariff schedule.',
    '',
    '## Key Results',
    '',
    `- Trade-weighted Liberation Day tariff: ${pct(tauEff, 2)}.`,
    `- Production/input-cost route: +${inputCostDrugPrice.toFixed(2)}% drug price pressure, using ICIO imported intermediate input share ${pct(ICIO_INTERMEDIATE_IMPORT_SHARE, 2)}, pass-through ${pct(PASS_THROUGH, 0)}, and IO multiplier ${IO_MULTIPLIER.toFixed(3)}x.`,
    `- Broader import-dependence route: +${importDependencyDrugPrice.toFixed(2)}% drug price pressure, using Pharma1 imports scaled to the project import-dependency benchmark (${pct(importDependency, 2)}).`,
    `- Implied import volume response: ${importVolumeChange.toFixed(1)}%, using elasticity ${EPS_PHARMA}.`,
    `- Supplier HHI: ${preHhi.toFixed(0)} pre-tariff to ${postHhi.toFixed(0)} post-tariff (${(postHhi - preHhi).toFixed(0)} change).`,
    `- Low-tariff supplier share (tariff <= 10%): ${pct(lowTariffShare, 1)} of 2024 import value.`,
    '',
    '## Top Suppliers, 2024',
    '',
    '| Rank | Country | Imports | Share | Tariff | Post-Tariff Share | Shift |',
    '|---:|---|---:|---:|---:|---:|---:|',
    ...topCountries.map((r, idx) => (
      `| ${idx + 1} | ${r.country} | ${moneyBn(r.value)} | ${pct(r.share, 1)} | ${r.tariff == null ? 'n/a' : pct(r.tariff, 0)} | ${r.share_post == null ? 'n/a' : pct(r.share_post, 1)} | ${r.delta_share == null ? 'n/a' : `${(r.delta_share * 100).toFixed(2)}pp`} |`
    )),
    '',
    '## HTS Composition, 2024',
    '',
    '| HTS | Imports | Share |',
    '|---|---:|---:|',
    ...htsExposure.map((r) => `| ${r.hts} | ${moneyBn(r.value)} | ${pct(r.share, 1)} |`),
    '',
    '## Year Totals In Pharma1',
    '',
    '| Year | Imports |',
    '|---:|---:|',
    ...byYear.map((r) => `| ${r.year} | ${moneyBn(r.total)} |`),
    '',
    '## Interpretation',
    '',
    'The Pharma1 data show a broad, high-value import base with no large zero-tariff escape valve in the matched tariff schedule. The model allows some sourcing reallocation toward lower-tariff suppliers, but the HHI barely moves and the import-volume response is strongly negative. In the short run, that means tariff pressure mainly transmits through imported input and finished-drug costs rather than being neutralized by rapid supplier switching.',
    '',
  ].join('\n');
  fs.writeFileSync(path.join(OUTPUT_DIR, 'pharma1_analysis_summary.md'), md);

  const tex = [
    '\\begin{table}[H]',
    '\\centering',
    '\\caption{Pharma1 Pharmaceutical Tariff Cost Transmission}',
    '\\label{tab:pharma1_price}',
    '\\begin{tabular}{lc}',
    '\\toprule',
    'Metric & Estimate \\\\',
    '\\midrule',
    `2024 Pharma1 imports & \\$${(total2024 / 1e9).toFixed(1)}B \\\\`,
    `Trade-weighted tariff & ${(tauEff * 100).toFixed(2)}\\% \\\\`,
    `Matched import value & ${(matchedValue / total2024 * 100).toFixed(2)}\\% \\\\`,
    `Production/input-cost drug price pressure & +${inputCostDrugPrice.toFixed(2)}\\% \\\\`,
    `Broader import-dependence drug price pressure & +${importDependencyDrugPrice.toFixed(2)}\\% \\\\`,
    `Import volume change & ${importVolumeChange.toFixed(1)}\\% \\\\`,
    `Supplier HHI, pre-tariff & ${preHhi.toFixed(0)} \\\\`,
    `Supplier HHI, post-tariff & ${postHhi.toFixed(0)} \\\\`,
    '\\bottomrule',
    '\\end{tabular}',
    '\\end{table}',
    '',
  ].join('\n');
  fs.writeFileSync(path.join(OUTPUT_DIR, 'Table_S3_pharma1_price.tex'), tex);

  const riskRows = topCountries.map((r, idx) => {
    let riskTier = 'Low';
    if ((r.share >= 0.05 && r.tariff >= 0.20) || r.tariff >= 0.30) riskTier = 'High';
    if (r.share >= 0.10 && r.tariff >= 0.20) riskTier = 'Very high';
    return {
      rank: idx + 1,
      country: r.country,
      import_value_bn: (r.value / 1e9).toFixed(2),
      pre_tariff_share_pct: (r.share * 100).toFixed(2),
      tariff_pct: r.tariff == null ? '' : (r.tariff * 100).toFixed(2),
      post_tariff_share_pct: r.share_post == null ? '' : (r.share_post * 100).toFixed(2),
      share_shift_pp: r.delta_share == null ? '' : (r.delta_share * 100).toFixed(2),
      risk_tier: riskTier,
    };
  });

  writeCsv(
    path.join(OUTPUT_DIR, 'pharma1_supply_chain_risk_2024.csv'),
    riskRows,
    ['rank', 'country', 'import_value_bn', 'pre_tariff_share_pct', 'tariff_pct', 'post_tariff_share_pct', 'share_shift_pp', 'risk_tier'],
  );

  const top5Share = countryExposure.slice(0, 5).reduce((sum, r) => sum + r.share, 0);
  const top10Share = countryExposure.slice(0, 10).reduce((sum, r) => sum + r.share, 0);
  const highTariffShare = countryExposure
    .filter((r) => r.matched && r.tariff >= 0.20)
    .reduce((sum, r) => sum + r.share, 0);
  const severeTariffShare = countryExposure
    .filter((r) => r.matched && r.tariff >= 0.30)
    .reduce((sum, r) => sum + r.share, 0);
  const biggestGainers = countryExposure
    .filter((r) => r.matched && r.delta_share > 0)
    .sort((a, b) => b.delta_share - a.delta_share)
    .slice(0, 5);
  const biggestLosers = countryExposure
    .filter((r) => r.matched && r.delta_share < 0)
    .sort((a, b) => a.delta_share - b.delta_share)
    .slice(0, 5);

  const riskMd = [
    '# Pharma1 Supply Chain Risk Analysis',
    '',
    'Objective: evaluate how tariffs affect pharmaceutical supply-chain risk by analyzing dependence on key importing countries, changes in sourcing patterns, and supplier concentration.',
    '',
    '## Supply Dependence',
    '',
    `- 2024 Pharma1 import base: ${moneyBn(total2024)} across ${countryExposure.length} supplier countries.`,
    `- Top 5 suppliers account for ${pct(top5Share, 1)} of imports.`,
    `- Top 10 suppliers account for ${pct(top10Share, 1)} of imports.`,
    `- Ireland alone supplies ${pct(countryExposure[0].share, 1)} of Pharma1 imports, creating a single-country dependence point even though the broader supplier base is diversified.`,
    `- ${pct(highTariffShare, 1)} of imports come from countries facing tariffs of at least 20%.`,
    `- ${pct(severeTariffShare, 1)} of imports come from countries facing tariffs of at least 30%.`,
    '',
    '## Sourcing Pattern Changes',
    '',
    `Post-tariff sourcing is modeled as share_post[j] proportional to share_pre[j] x (1 + tariff[j])^(-${EPS_PHARMA}).`,
    '',
    '| Direction | Country | Tariff | Pre Share | Post Share | Shift |',
    '|---|---|---:|---:|---:|---:|',
    ...biggestGainers.map((r) => `| Gain | ${r.country} | ${pct(r.tariff, 0)} | ${pct(r.share, 1)} | ${pct(r.share_post, 1)} | +${(r.delta_share * 100).toFixed(2)}pp |`),
    ...biggestLosers.map((r) => `| Loss | ${r.country} | ${pct(r.tariff, 0)} | ${pct(r.share, 1)} | ${pct(r.share_post, 1)} | ${(r.delta_share * 100).toFixed(2)}pp |`),
    '',
    '## Supplier Concentration',
    '',
    `- HHI before tariffs: ${preHhi.toFixed(0)}.`,
    `- HHI after tariff-induced reallocation: ${postHhi.toFixed(0)}.`,
    `- HHI change: ${(postHhi - preHhi).toFixed(0)} points.`,
    '',
    'The HHI remains in the unconcentrated range, so the main risk is not monopoly-style concentration. The risk is tariff-correlated dependence: several large suppliers face 20% or higher tariffs, while low-tariff alternatives account for only a limited share of the baseline supply base.',
    '',
    '## Risk Takeaway',
    '',
    'Tariffs raise pharmaceutical supply-chain risk by making the existing import network more expensive and pushing sourcing toward a small set of lower-tariff alternatives. Because those alternatives start from a modest import share, substitution is partial rather than complete. The result is a supply chain that remains diversified on paper but becomes more fragile in practice: high-tariff key suppliers lose share, lower-tariff suppliers gain share, and the overall concentration metric moves only slightly.',
    '',
  ].join('\n');
  fs.writeFileSync(path.join(OUTPUT_DIR, 'pharma1_supply_chain_risk.md'), riskMd);

  function buildCountryExposureForYear(targetYear) {
    const yearRows = rows.filter((r) => r.year === targetYear);
    const yearTotal = yearRows.reduce((sum, r) => sum + r.total, 0);
    const grouped = [...groupBy(yearRows, (r) => r.country).entries()]
      .map(([country, value]) => {
        const match = tariffMap.get(normalizeName(country));
        return {
          country,
          iso3: match?.iso3 || '',
          value,
          share: value / yearTotal,
          tariff: Number.isFinite(match?.tariff) ? match.tariff : null,
          matched: Boolean(match && Number.isFinite(match.tariff)),
        };
      })
      .sort((a, b) => b.value - a.value);

    const matchedRows = grouped.filter((r) => r.matched);
    const matchedTotal = matchedRows.reduce((sum, r) => sum + r.value, 0);
    const weightedTariff = matchedRows.reduce((sum, r) => sum + r.value * r.tariff, 0) / matchedTotal;
    const supplierHhi = hhi(matchedRows.map((r) => r.share));
    const top5 = grouped.slice(0, 5);
    const top10 = grouped.slice(0, 10);

    return {
      year: targetYear,
      total: yearTotal,
      nCountries: grouped.length,
      matchedShare: matchedTotal / yearTotal,
      weightedTariff,
      hhi: supplierHhi,
      top5Share: top5.reduce((sum, r) => sum + r.share, 0),
      top10Share: top10.reduce((sum, r) => sum + r.share, 0),
      rows: grouped,
      top5,
    };
  }

  const dep2024 = buildCountryExposureForYear(2024);
  const dep2025 = buildCountryExposureForYear(2025);
  const rank2024 = new Map(dep2024.rows.map((r, idx) => [r.country, idx + 1]));
  const rank2025 = new Map(dep2025.rows.map((r, idx) => [r.country, idx + 1]));

  const objectiveRows = [...new Set([...dep2024.top5, ...dep2025.top5].map((r) => r.country))]
    .map((country) => {
      const r2024 = dep2024.rows.find((r) => r.country === country);
      const r2025 = dep2025.rows.find((r) => r.country === country);
      return {
        country,
        rank_2024: rank2024.get(country) || '',
        imports_2024_bn: r2024 ? (r2024.value / 1e9).toFixed(2) : '',
        share_2024_pct: r2024 ? (r2024.share * 100).toFixed(2) : '',
        rank_2025: rank2025.get(country) || '',
        imports_2025_bn: r2025 ? (r2025.value / 1e9).toFixed(2) : '',
        share_2025_pct: r2025 ? (r2025.share * 100).toFixed(2) : '',
        rank_change: rank2024.has(country) && rank2025.has(country)
          ? rank2024.get(country) - rank2025.get(country)
          : '',
      };
    })
    .sort((a, b) => {
      const ar = a.rank_2025 || 999;
      const br = b.rank_2025 || 999;
      return ar - br;
    });

  writeCsv(
    path.join(OUTPUT_DIR, 'pharma1_objective1_dependence_top_suppliers.csv'),
    objectiveRows,
    ['country', 'rank_2024', 'imports_2024_bn', 'share_2024_pct', 'rank_2025', 'imports_2025_bn', 'share_2025_pct', 'rank_change'],
  );

  const objectiveMd = [
    '# Objective 1: U.S. Dependence on Key Pharmaceutical Supplier Countries',
    '',
    'Objective: measure how dependent the U.S. pharmaceutical supply chain is on foreign supplier countries by identifying the largest sources of pharmaceutical imports, measuring their import shares, and assessing supplier concentration.',
    '',
    '## 2024 Top 5 Supplier Countries',
    '',
    '| Rank | Country | Imports | Share | Tariff |',
    '|---:|---|---:|---:|---:|',
    ...dep2024.top5.map((r, idx) => `| ${idx + 1} | ${r.country} | ${moneyBn(r.value)} | ${pct(r.share, 2)} | ${r.tariff == null ? 'n/a' : pct(r.tariff, 0)} |`),
    '',
    '## 2025 Top 5 Supplier Countries',
    '',
    '| Rank | Country | Imports | Share | Tariff |',
    '|---:|---|---:|---:|---:|',
    ...dep2025.top5.map((r, idx) => `| ${idx + 1} | ${r.country} | ${moneyBn(r.value)} | ${pct(r.share, 2)} | ${r.tariff == null ? 'n/a' : pct(r.tariff, 0)} |`),
    '',
    '## Dependence Metrics',
    '',
    '| Metric | 2024 | 2025 |',
    '|---|---:|---:|',
    `| Total Pharma1 imports | ${moneyBn(dep2024.total)} | ${moneyBn(dep2025.total)} |`,
    `| Supplier countries | ${dep2024.nCountries} | ${dep2025.nCountries} |`,
    `| Top 5 supplier share | ${pct(dep2024.top5Share, 1)} | ${pct(dep2025.top5Share, 1)} |`,
    `| Top 10 supplier share | ${pct(dep2024.top10Share, 1)} | ${pct(dep2025.top10Share, 1)} |`,
    `| Supplier HHI | ${dep2024.hhi.toFixed(0)} | ${dep2025.hhi.toFixed(0)} |`,
    `| Trade-weighted tariff | ${pct(dep2024.weightedTariff, 2)} | ${pct(dep2025.weightedTariff, 2)} |`,
    '',
    '## Ranking Changes',
    '',
    '| Country | 2024 Rank | 2025 Rank | Change |',
    '|---|---:|---:|---:|',
    ...objectiveRows.map((r) => `| ${r.country} | ${r.rank_2024 || '-'} | ${r.rank_2025 || '-'} | ${r.rank_change === '' ? '-' : r.rank_change} |`),
    '',
    '## Interpretation',
    '',
    `The U.S. pharmaceutical import base remains highly dependent on a small group of supplier countries. The top 5 suppliers account for ${pct(dep2024.top5Share, 1)} of imports in 2024 and ${pct(dep2025.top5Share, 1)} in 2025. Ireland is the largest source in both years, though its share falls from ${pct(dep2024.top5[0].share, 1)} to ${pct(dep2025.top5[0].share, 1)}. Germany rises to the second position in 2025, increasing its role as a key supplier. HHI falls from ${dep2024.hhi.toFixed(0)} to ${dep2025.hhi.toFixed(0)}, meaning supplier concentration decreases somewhat, but dependence on major foreign suppliers remains substantial.`,
    '',
  ].join('\n');
  fs.writeFileSync(path.join(OUTPUT_DIR, 'pharma1_objective1_dependence.md'), objectiveMd);

  const objective2025Rows = dep2025.rows.slice(0, 10).map((r, idx) => ({
    rank: idx + 1,
    country: r.country,
    imports_2025_bn: (r.value / 1e9).toFixed(2),
    share_2025_pct: (r.share * 100).toFixed(2),
    tariff_pct: r.tariff == null ? '' : (r.tariff * 100).toFixed(2),
    cumulative_share_pct: (dep2025.rows.slice(0, idx + 1).reduce((sum, row) => sum + row.share, 0) * 100).toFixed(2),
  }));

  writeCsv(
    path.join(OUTPUT_DIR, 'pharma1_objective1_dependence_2025.csv'),
    objective2025Rows,
    ['rank', 'country', 'imports_2025_bn', 'share_2025_pct', 'tariff_pct', 'cumulative_share_pct'],
  );

  const top5HighTariffShare2025 = dep2025.top5
    .filter((r) => r.tariff != null && r.tariff >= 0.20)
    .reduce((sum, r) => sum + r.share, 0);
  const top10HighTariffShare2025 = dep2025.rows.slice(0, 10)
    .filter((r) => r.tariff != null && r.tariff >= 0.20)
    .reduce((sum, r) => sum + r.share, 0);

  const allHighTariffShare2025 = dep2025.rows
    .filter((r) => r.tariff != null && r.tariff >= 0.20)
    .reduce((sum, r) => sum + r.share, 0);
  const severeTariffShare2025 = dep2025.rows
    .filter((r) => r.tariff != null && r.tariff >= 0.30)
    .reduce((sum, r) => sum + r.share, 0);
  const lowTariffShare2025 = dep2025.rows
    .filter((r) => r.tariff != null && r.tariff <= 0.10)
    .reduce((sum, r) => sum + r.share, 0);

  const objective2025Md = [
    '# Objective 1: U.S. Dependence on Key Pharmaceutical Supplier Countries and Tariff-Exposed Supply Risk, 2025',
    '',
    'Objective: evaluate U.S. pharmaceutical import dependence by identifying the top supplier countries, calculating supplier concentration, and measuring how much of the import base is exposed to Liberation Day tariffs.',
    '',
    'This objective asks whether the U.S. pharmaceutical supply chain is vulnerable not only because it relies on a small group of foreign suppliers, but also because those key suppliers face substantial tariff exposure.',
    '',
    '## 2025 Top 5 Supplier Countries',
    '',
    '| Rank | Country | Imports | Share of U.S. Pharma Imports | Liberation Day Tariff |',
    '|---:|---|---:|---:|---:|',
    ...dep2025.top5.map((r, idx) => `| ${idx + 1} | ${r.country} | ${moneyBn(r.value)} | ${pct(r.share, 2)} | ${r.tariff == null ? 'n/a' : pct(r.tariff, 0)} |`),
    '',
    '## 2025 Dependence Metrics',
    '',
    '| Metric | Value |',
    '|---|---:|',
    `| Total Pharma1 imports | ${moneyBn(dep2025.total)} |`,
    `| Supplier countries | ${dep2025.nCountries} |`,
    `| Top 5 supplier share | ${pct(dep2025.top5Share, 1)} |`,
    `| Top 10 supplier share | ${pct(dep2025.top10Share, 1)} |`,
    `| Supplier HHI | ${dep2025.hhi.toFixed(0)} |`,
    `| Trade-weighted tariff | ${pct(dep2025.weightedTariff, 2)} |`,
    `| Total import share facing tariffs >= 20% | ${pct(allHighTariffShare2025, 1)} |`,
    `| Total import share facing tariffs >= 30% | ${pct(severeTariffShare2025, 1)} |`,
    `| Total import share facing tariffs <= 10% | ${pct(lowTariffShare2025, 1)} |`,
    `| Top 5 share facing tariffs >= 20% | ${pct(top5HighTariffShare2025, 1)} |`,
    `| Top 10 share facing tariffs >= 20% | ${pct(top10HighTariffShare2025, 1)} |`,
    '',
    '## Tariff-Exposed Supply Risk',
    '',
    '| Risk Channel | Evidence from 2025 Pharma1 Data |',
    '|---|---|',
    `| Key-supplier dependence | Top 5 countries supply ${pct(dep2025.top5Share, 1)} of U.S. pharmaceutical imports. |`,
    `| Broad concentration | Top 10 countries supply ${pct(dep2025.top10Share, 1)} of imports. |`,
    `| Tariff-exposed dependence | ${pct(allHighTariffShare2025, 1)} of all imports come from countries facing tariffs of at least 20%. |`,
    `| Limited low-tariff alternatives | Only ${pct(lowTariffShare2025, 1)} of imports come from countries facing tariffs of 10% or less. |`,
    `| Severe tariff exposure | ${pct(severeTariffShare2025, 1)} of imports come from countries facing tariffs of at least 30%. |`,
    '',
    '## Interpretation',
    '',
    `In 2025, U.S. pharmaceutical imports remain highly dependent on a small group of countries. The top 5 suppliers account for ${pct(dep2025.top5Share, 1)} of imports, and the top 10 account for ${pct(dep2025.top10Share, 1)}. Ireland is the largest supplier, providing ${pct(dep2025.top5[0].share, 2)} of imports, followed by Germany at ${pct(dep2025.top5[1].share, 2)} and Switzerland at ${pct(dep2025.top5[2].share, 2)}.`,
    '',
    `The tariff exposure is substantial: ${pct(allHighTariffShare2025, 1)} of all 2025 pharmaceutical imports come from countries facing Liberation Day tariffs of at least 20%, and ${pct(top5HighTariffShare2025, 1)} comes from the top-5 suppliers alone. This means the tariff shock falls directly on the countries that matter most for U.S. pharmaceutical supply.`,
    '',
    `The HHI of ${dep2025.hhi.toFixed(0)} suggests the market is not extremely concentrated in a monopoly sense. However, supplier risk remains important because the largest countries are heavily tariff-exposed and low-tariff alternatives represent only ${pct(lowTariffShare2025, 1)} of imports. The main vulnerability is therefore tariff-exposed dependence: the U.S. relies on a small group of major pharmaceutical suppliers, and most of that import base faces meaningful tariff pressure.`,
    '',
  ].join('\n');
  fs.writeFileSync(path.join(OUTPUT_DIR, 'pharma1_objective1_dependence_2025.md'), objective2025Md);

  function addPostTariffShares(exposure) {
    const validRows = exposure.rows.filter((r) => r.tariff != null);
    const denominator = validRows.reduce((sum, r) => sum + r.share * (1 + r.tariff) ** (-EPS_PHARMA), 0);
    return exposure.rows.map((r) => {
      if (r.tariff == null) return { ...r, sharePost: null, deltaShare: null };
      const sharePost = (r.share * (1 + r.tariff) ** (-EPS_PHARMA)) / denominator;
      return { ...r, sharePost, deltaShare: sharePost - r.share };
    });
  }

  const dep2025Post = addPostTariffShares(dep2025);
  const gainers2025 = dep2025Post
    .filter((r) => r.deltaShare != null && r.deltaShare > 0)
    .sort((a, b) => b.deltaShare - a.deltaShare)
    .slice(0, 10);
  const losers2025 = dep2025Post
    .filter((r) => r.deltaShare != null && r.deltaShare < 0)
    .sort((a, b) => a.deltaShare - b.deltaShare)
    .slice(0, 10);
  const top5PostShare2025 = dep2025Post
    .sort((a, b) => (b.sharePost ?? -1) - (a.sharePost ?? -1))
    .slice(0, 5)
    .reduce((sum, r) => sum + r.sharePost, 0);
  const postHhi2025 = hhi(dep2025Post.filter((r) => r.sharePost != null).map((r) => r.sharePost));

  writeCsv(
    path.join(OUTPUT_DIR, 'pharma1_objective2_sourcing_shifts_2025.csv'),
    dep2025Post.slice(0, 25).map((r, idx) => ({
      rank_pre_tariff: idx + 1,
      country: r.country,
      imports_2025_bn: (r.value / 1e9).toFixed(2),
      tariff_pct: r.tariff == null ? '' : (r.tariff * 100).toFixed(2),
      pre_tariff_share_pct: (r.share * 100).toFixed(3),
      post_tariff_share_pct: r.sharePost == null ? '' : (r.sharePost * 100).toFixed(3),
      change_pp: r.deltaShare == null ? '' : (r.deltaShare * 100).toFixed(3),
    })),
    ['rank_pre_tariff', 'country', 'imports_2025_bn', 'tariff_pct', 'pre_tariff_share_pct', 'post_tariff_share_pct', 'change_pp'],
  );

  const objective2Md = [
    '# Objective 2: Tariff-Driven Pharmaceutical Sourcing Shifts, 2025',
    '',
    'Objective: analyze how tariff exposure alters pharmaceutical sourcing patterns by identifying supplier countries that gain or lose U.S. import share after tariffs.',
    '',
    `Post-tariff shares are estimated with a gravity-style sourcing response: share_post[j] is proportional to share_pre[j] x (1 + tariff[j])^(-${EPS_PHARMA}). This captures limited substitution: lower-tariff suppliers gain share, but the response is partial rather than a full replacement of high-tariff suppliers.`,
    '',
    '## Largest Estimated Share Gainers',
    '',
    '| Rank | Country | Tariff | Pre Share | Post Share | Change |',
    '|---:|---|---:|---:|---:|---:|',
    ...gainers2025.slice(0, 5).map((r, idx) => `| ${idx + 1} | ${r.country} | ${pct(r.tariff, 0)} | ${pct(r.share, 2)} | ${pct(r.sharePost, 2)} | +${(r.deltaShare * 100).toFixed(2)}pp |`),
    '',
    '## Largest Estimated Share Losers',
    '',
    '| Rank | Country | Tariff | Pre Share | Post Share | Change |',
    '|---:|---|---:|---:|---:|---:|',
    ...losers2025.slice(0, 5).map((r, idx) => `| ${idx + 1} | ${r.country} | ${pct(r.tariff, 0)} | ${pct(r.share, 2)} | ${pct(r.sharePost, 2)} | ${(r.deltaShare * 100).toFixed(2)}pp |`),
    '',
    '## Sourcing Concentration Before and After Tariffs',
    '',
    '| Metric | Pre-Tariff | Post-Tariff Estimate |',
    '|---|---:|---:|',
    `| Top 5 supplier share | ${pct(dep2025.top5Share, 1)} | ${pct(top5PostShare2025, 1)} |`,
    `| Supplier HHI | ${dep2025.hhi.toFixed(0)} | ${postHhi2025.toFixed(0)} |`,
    '',
    '## Interpretation',
    '',
    `Tariffs shift estimated sourcing toward lower-tariff suppliers such as Singapore, the United Kingdom, Canada, Ireland, and Australia. Higher-tariff suppliers such as Switzerland, China, India, Japan, and South Korea lose share. However, the shift is limited: the top-5 supplier share changes from ${pct(dep2025.top5Share, 1)} before tariffs to ${pct(top5PostShare2025, 1)} after tariffs, and HHI changes from ${dep2025.hhi.toFixed(0)} to ${postHhi2025.toFixed(0)}.`,
    '',
    'This means tariffs do not meaningfully diversify the U.S. pharmaceutical supply base. Instead, they partially reallocate sourcing toward a small set of lower-tariff countries while leaving overall dependence on major foreign suppliers intact.',
    '',
  ].join('\n');
  fs.writeFileSync(path.join(OUTPUT_DIR, 'pharma1_objective2_sourcing_shifts_2025.md'), objective2Md);

  const objective3Rows = DRUG_SPEND_QUINTILES.map((q) => {
    const inferredIncome = q.referenceExtra / (q.referenceBurdenPctIncome / 100);
    const extraCost = q.baseSpend * (importDependencyDrugPrice / 100);
    const burdenPctIncome = extraCost / inferredIncome * 100;
    return {
      group: q.group,
      base_spend_yr: q.baseSpend,
      extra_cost_yr: extraCost,
      burden_pct_income: burdenPctIncome,
      inferred_income: inferredIncome,
    };
  });
  const maxSpendGroup = objective3Rows.reduce((best, row) => (
    row.base_spend_yr > best.base_spend_yr ? row : best
  ), objective3Rows[0]);
  const maxBurdenGroup = objective3Rows.reduce((best, row) => (
    row.burden_pct_income > best.burden_pct_income ? row : best
  ), objective3Rows[0]);
  const q1Burden = objective3Rows[0].burden_pct_income;
  const q5Burden = objective3Rows[4].burden_pct_income;
  const regressivityRatio = q1Burden / q5Burden;

  writeCsv(
    path.join(OUTPUT_DIR, 'pharma1_objective3_consumer_burden_2025.csv'),
    objective3Rows.map((r) => ({
      group: r.group,
      annual_drug_spending_usd: r.base_spend_yr.toFixed(0),
      extra_cost_from_tariffs_usd: r.extra_cost_yr.toFixed(2),
      burden_pct_income: r.burden_pct_income.toFixed(3),
    })),
    ['group', 'annual_drug_spending_usd', 'extra_cost_from_tariffs_usd', 'burden_pct_income'],
  );

  const objective3Md = [
    '# Objective 3: Consumer Burden from Tariff-Driven Pharmaceutical Price Increases, 2025',
    '',
    'Objective: assess the distributional impact of tariff-driven pharmaceutical price increases by identifying which consumer groups spend the most on prescription drugs and which groups face the greatest financial burden.',
    '',
    `This analysis applies the Pharma1 2025 tariff-driven drug-price pressure estimate of +${importDependencyDrugPrice.toFixed(2)}% to baseline annual drug spending by income quintile.`,
    '',
    '## Spending and Burden by Income Group',
    '',
    '| Group | Annual Drug Spending | Extra Cost from Tariffs | Extra Cost as % of Income |',
    '|---|---:|---:|---:|',
    ...objective3Rows.map((r) => `| ${r.group} | $${r.base_spend_yr.toFixed(0)} | +$${r.extra_cost_yr.toFixed(0)} | ${r.burden_pct_income.toFixed(3)}% |`),
    '',
    '## Key Findings',
    '',
    `- The group spending the most in dollar terms is ${maxSpendGroup.group}, with annual drug spending of $${maxSpendGroup.base_spend_yr.toFixed(0)}.`,
    `- The group facing the largest burden relative to income is ${maxBurdenGroup.group}, with tariff-driven extra costs equal to ${maxBurdenGroup.burden_pct_income.toFixed(3)}% of income.`,
    `- The lowest-income group faces a burden ${regressivityRatio.toFixed(1)}x larger than the highest-income group as a share of income.`,
    '',
    '## Interpretation',
    '',
    `Higher-income households spend the most on drugs in dollar terms, but lower-income households face the greatest financial burden. Under the Pharma1 tariff-price estimate, Q5 households spend the most annually on drugs, while Q1 households experience the highest cost burden relative to income. This means tariff-driven pharmaceutical price increases are regressive: the dollar increase may be larger for some higher-spending groups, but the affordability impact is greatest for lower-income consumers.`,
    '',
  ].join('\n');
  fs.writeFileSync(path.join(OUTPUT_DIR, 'pharma1_objective3_consumer_burden_2025.md'), objective3Md);

  const barMax = Math.max(...dep2025.top5.map((r) => r.share));
  const barX = 235;
  const barW = 430;
  const rowY = 142;
  const rowGap = 56;
  const top5Bars = dep2025.top5.map((r, idx) => {
    const y = rowY + idx * rowGap;
    const w = (r.share / barMax) * barW;
    const color = r.tariff >= 0.30 ? '#b91c1c' : r.tariff >= 0.20 ? '#dc6b2f' : '#2f8f6b';
    return `
      <text x="52" y="${y + 19}" class="country">${idx + 1}. ${svgEscape(r.country)}</text>
      <rect x="${barX}" y="${y}" width="${w.toFixed(1)}" height="28" rx="4" fill="${color}"/>
      <text x="${barX + w + 10}" y="${y + 19}" class="value">${(r.share * 100).toFixed(2)}% | $${(r.value / 1e9).toFixed(1)}B | tariff ${(r.tariff * 100).toFixed(0)}%</text>`;
  }).join('\n');

  const metricCards = [
    ['Top 5 Share', pct(dep2025.top5Share, 1), 'More than half of imports'],
    ['Top 10 Share', pct(dep2025.top10Share, 1), 'Broad foreign dependence'],
    ['Tariff >= 20%', pct(allHighTariffShare2025, 1), 'Most import value exposed'],
    ['Low Tariff <= 10%', pct(lowTariffShare2025, 1), 'Limited alternatives'],
  ].map((m, idx) => {
    const x = 52 + idx * 214;
    return `
      <rect x="${x}" y="462" width="190" height="104" rx="8" fill="#f7f9fb" stroke="#d6dde5"/>
      <text x="${x + 16}" y="492" class="metric-label">${svgEscape(m[0])}</text>
      <text x="${x + 16}" y="526" class="metric-value">${svgEscape(m[1])}</text>
      <text x="${x + 16}" y="550" class="metric-note">${svgEscape(m[2])}</text>`;
  }).join('\n');

  const svg = `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="960" height="640" viewBox="0 0 960 640" role="img" aria-label="2025 U.S. pharmaceutical supplier dependence and tariff exposure">
  <style>
    .title { font: 700 25px Arial, sans-serif; fill: #1d2733; }
    .subtitle { font: 400 14px Arial, sans-serif; fill: #536273; }
    .section { font: 700 15px Arial, sans-serif; fill: #1d2733; }
    .country { font: 700 14px Arial, sans-serif; fill: #253244; }
    .value { font: 400 13px Arial, sans-serif; fill: #2d3b4f; }
    .metric-label { font: 700 13px Arial, sans-serif; fill: #536273; text-transform: uppercase; }
    .metric-value { font: 700 28px Arial, sans-serif; fill: #1d2733; }
    .metric-note { font: 400 12px Arial, sans-serif; fill: #66758a; }
    .footnote { font: 400 11px Arial, sans-serif; fill: #748297; }
  </style>
  <rect width="960" height="640" fill="#ffffff"/>
  <text x="52" y="52" class="title">U.S. Pharmaceutical Import Dependence, 2025</text>
  <text x="52" y="78" class="subtitle">Pharma1 imports for consumption: $${(dep2025.total / 1e9).toFixed(1)}B across ${dep2025.nCountries} supplier countries | Trade-weighted tariff: ${(dep2025.weightedTariff * 100).toFixed(2)}%</text>

  <text x="52" y="115" class="section">Top 5 supplier countries by import share</text>
  <text x="235" y="115" class="subtitle">Share of total 2025 U.S. pharmaceutical imports</text>
  ${top5Bars}

  <line x1="52" y1="430" x2="908" y2="430" stroke="#d6dde5"/>
  <text x="52" y="448" class="section">Tariff-exposed supply-risk indicators</text>
  ${metricCards}

  <circle cx="710" cy="156" r="6" fill="#dc6b2f"/>
  <text x="724" y="161" class="subtitle">20-29% tariff</text>
  <circle cx="810" cy="156" r="6" fill="#b91c1c"/>
  <text x="824" y="161" class="subtitle">30%+ tariff</text>

  <text x="52" y="606" class="footnote">Interpretation: supplier concentration is moderate by HHI (${dep2025.hhi.toFixed(0)}), but risk is high because ${pct(allHighTariffShare2025, 1)} of imports face tariffs of at least 20% and low-tariff alternatives are only ${pct(lowTariffShare2025, 1)}.</text>
</svg>
`;
  fs.writeFileSync(path.join(OUTPUT_DIR, 'fig_pharma1_objective1_2025.svg'), svg);

  const shiftRows = [...gainers2025.slice(0, 5), ...losers2025.slice(0, 5)];
  const maxAbsDelta = Math.max(...shiftRows.map((r) => Math.abs(r.deltaShare)));
  const shiftBars = shiftRows.map((r, idx) => {
    const y = 118 + idx * 42;
    const width = Math.abs(r.deltaShare) / maxAbsDelta * 250;
    const x = r.deltaShare >= 0 ? 480 : 480 - width;
    const color = r.deltaShare >= 0 ? '#2f8f6b' : '#b91c1c';
    const labelX = r.deltaShare >= 0 ? x + width + 8 : x - 8;
    const anchor = r.deltaShare >= 0 ? 'start' : 'end';
    return `
      <text x="52" y="${y + 17}" class="country">${svgEscape(r.country)}</text>
      <text x="245" y="${y + 17}" class="subtitle">tariff ${(r.tariff * 100).toFixed(0)}%</text>
      <rect x="${x.toFixed(1)}" y="${y}" width="${width.toFixed(1)}" height="24" rx="4" fill="${color}"/>
      <text x="${labelX.toFixed(1)}" y="${y + 17}" text-anchor="${anchor}" class="value">${r.deltaShare >= 0 ? '+' : ''}${(r.deltaShare * 100).toFixed(2)}pp</text>`;
  }).join('\n');

  const svg2 = `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="960" height="640" viewBox="0 0 960 640" role="img" aria-label="2025 pharmaceutical sourcing shifts after tariffs">
  <style>
    .title { font: 700 25px Arial, sans-serif; fill: #1d2733; }
    .subtitle { font: 400 13px Arial, sans-serif; fill: #536273; }
    .section { font: 700 15px Arial, sans-serif; fill: #1d2733; }
    .country { font: 700 14px Arial, sans-serif; fill: #253244; }
    .value { font: 400 13px Arial, sans-serif; fill: #2d3b4f; }
    .metric-label { font: 700 13px Arial, sans-serif; fill: #536273; }
    .metric-value { font: 700 28px Arial, sans-serif; fill: #1d2733; }
    .footnote { font: 400 11px Arial, sans-serif; fill: #748297; }
  </style>
  <rect width="960" height="640" fill="#ffffff"/>
  <text x="52" y="52" class="title">Estimated Pharmaceutical Sourcing Shifts After Tariffs, 2025</text>
  <text x="52" y="78" class="subtitle">Gravity-style reallocation with elasticity ${EPS_PHARMA}: lower-tariff suppliers gain share, high-tariff suppliers lose share.</text>
  <text x="52" y="104" class="section">Largest gains and losses in import share</text>
  <line x1="480" y1="110" x2="480" y2="540" stroke="#9aa8b8" stroke-dasharray="4 4"/>
  <text x="382" y="104" class="subtitle">Losses</text>
  <text x="500" y="104" class="subtitle">Gains</text>
  ${shiftBars}
  <rect x="52" y="562" width="250" height="52" rx="8" fill="#f7f9fb" stroke="#d6dde5"/>
  <text x="68" y="584" class="metric-label">Top 5 share</text>
  <text x="68" y="606" class="subtitle">${pct(dep2025.top5Share, 1)} pre-tariff -> ${pct(top5PostShare2025, 1)} post-tariff</text>
  <rect x="330" y="562" width="250" height="52" rx="8" fill="#f7f9fb" stroke="#d6dde5"/>
  <text x="346" y="584" class="metric-label">Supplier HHI</text>
  <text x="346" y="606" class="subtitle">${dep2025.hhi.toFixed(0)} pre-tariff -> ${postHhi2025.toFixed(0)} post-tariff</text>
  <text x="610" y="592" class="footnote">Interpretation: tariffs reallocate sourcing only partially; they do not materially diversify supplier dependence.</text>
</svg>
`;
  fs.writeFileSync(path.join(OUTPUT_DIR, 'fig_pharma1_objective2_sourcing_shifts_2025.svg'), svg2);

  const maxBaseSpend = Math.max(...objective3Rows.map((r) => r.base_spend_yr));
  const maxBurden = Math.max(...objective3Rows.map((r) => r.burden_pct_income));
  const spendBars = objective3Rows.map((r, idx) => {
    const y = 128 + idx * 58;
    const w = r.base_spend_yr / maxBaseSpend * 300;
    const burdenW = r.burden_pct_income / maxBurden * 300;
    return `
      <text x="54" y="${y + 19}" class="country">${svgEscape(r.group)}</text>
      <rect x="220" y="${y}" width="${w.toFixed(1)}" height="24" rx="4" fill="#2f8f6b"/>
      <text x="${230 + w}" y="${y + 18}" class="value">$${r.base_spend_yr.toFixed(0)}</text>
      <rect x="560" y="${y}" width="${burdenW.toFixed(1)}" height="24" rx="4" fill="#b91c1c"/>
      <text x="${570 + burdenW}" y="${y + 18}" class="value">${r.burden_pct_income.toFixed(3)}%</text>`;
  }).join('\n');

  const svg3 = `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="960" height="560" viewBox="0 0 960 560" role="img" aria-label="Consumer burden from pharmaceutical tariff price increases by income group">
  <style>
    .title { font: 700 24px Arial, sans-serif; fill: #1d2733; }
    .subtitle { font: 400 13px Arial, sans-serif; fill: #536273; }
    .section { font: 700 14px Arial, sans-serif; fill: #1d2733; }
    .country { font: 700 13px Arial, sans-serif; fill: #253244; }
    .value { font: 400 12px Arial, sans-serif; fill: #2d3b4f; }
    .metric { font: 700 22px Arial, sans-serif; fill: #1d2733; }
    .footnote { font: 400 11px Arial, sans-serif; fill: #748297; }
  </style>
  <rect width="960" height="560" fill="#ffffff"/>
  <text x="52" y="52" class="title">Who Bears Pharmaceutical Tariff Costs?</text>
  <text x="52" y="78" class="subtitle">Pharma1 2025 estimated drug-price pressure: +${importDependencyDrugPrice.toFixed(2)}%</text>
  <text x="220" y="112" class="section">Annual drug spending</text>
  <text x="560" y="112" class="section">Extra cost as % of income</text>
  ${spendBars}
  <rect x="52" y="440" width="250" height="70" rx="8" fill="#f7f9fb" stroke="#d6dde5"/>
  <text x="68" y="466" class="subtitle">Highest dollar spending</text>
  <text x="68" y="494" class="metric">${svgEscape(maxSpendGroup.group)}</text>
  <rect x="330" y="440" width="250" height="70" rx="8" fill="#f7f9fb" stroke="#d6dde5"/>
  <text x="346" y="466" class="subtitle">Largest income burden</text>
  <text x="346" y="494" class="metric">${svgEscape(maxBurdenGroup.group)}</text>
  <rect x="608" y="440" width="250" height="70" rx="8" fill="#f7f9fb" stroke="#d6dde5"/>
  <text x="624" y="466" class="subtitle">Regressivity ratio</text>
  <text x="624" y="494" class="metric">${regressivityRatio.toFixed(1)}x</text>
  <text x="52" y="536" class="footnote">Interpretation: higher-income consumers spend more dollars on drugs, but lower-income consumers bear the larger burden relative to income.</text>
</svg>
`;
  fs.writeFileSync(path.join(OUTPUT_DIR, 'fig_pharma1_objective3_consumer_burden_2025.svg'), svg3);

  console.log('Pharma1 analysis complete.');
  console.log(`2024 imports: ${moneyBn(total2024)}`);
  console.log(`Trade-weighted tariff: ${(tauEff * 100).toFixed(2)}%`);
  console.log(`Input-cost drug price pressure: +${inputCostDrugPrice.toFixed(2)}%`);
  console.log(`Import-dependence drug price pressure: +${importDependencyDrugPrice.toFixed(2)}%`);
  console.log(`Import volume response: ${importVolumeChange.toFixed(1)}%`);
  console.log(`HHI: ${preHhi.toFixed(0)} -> ${postHhi.toFixed(0)}`);
  console.log('Saved outputs to python_output/.');
}

run();

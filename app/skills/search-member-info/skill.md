---
name: bupa-member-products
description: Query and analyse the Bupa member-to-product table in SQL (member_number, product). Use this skill whenever the user asks about Bupa members, which product or cover a member holds or wants SQL written against the member products data, even if they don't name the table. Also use it for sanity checks on this data, such as duplicates, unknown products or malformed member numbers.
---

# Bupa Member Products – SQL Query Skill

Helps write, run and explain SQL against a table that maps each Bupa member number to the hospital product they hold. Product names come from the *Bupa Hospital covers at a glance* guide (effective 2 March 2026).

## Table

Table name: `member_info` (spelled exactly this way; if the user's table turns out to be named differently, use theirs and say so).

| Column          | Type          | Notes                                                                 |
|-----------------|---------------|-----------------------------------------------------------------------|
| `member_number` | INT | 8-digit identifier stored as an integer. Compare without quotes: <MEMBER_NO>. Never do arithmetic on it. Expect values from 10000000 to 99999999; an integer cannot hold leading zeros, so a number that should start with 0 would appear with only 7 digits. |
| `product`       | VARCHAR / TEXT | Full product name, including excess or level where applicable, e.g. `Silver Plus Prime $500 Excess`. |

Grain: one row per member. If duplicate `member_number` values exist, flag it rather than silently deduplicating.

Before writing queries against an unfamiliar load, confirm the real schema (`DESCRIBE member_info`, `\d member_info`, or `SELECT * FROM member_info LIMIT 5`).

## Workflow

1. Identify the question type: lookup, count/breakdown, data quality, or join with another table.
2. Write ANSI-style SQL first; adjust for the user's dialect (see Dialect notes).
3. Prefer `GROUP BY` plus `COUNT(*)` for breakdowns, and order results by count descending.
4. For exact-name filters use `=`. For families (e.g. all Gold products) use the derived-attribute patterns below, not hand-typed lists.
5. State any assumption (table name, case sensitivity, how a product was classified) alongside the query.

## Deriving attributes from the `product` text

The table holds only the product name, so tier, excess and segment are derived. Use `CASE` expressions. Order matters: check the most specific pattern first.

## Example queries

Look up one member:
```sql
SELECT member_number, product
FROM member_info
WHERE member_number = <MEMBER_NO>;
```

## Dialect notes

| Need                     | Postgres            | MySQL              | SQL Server           | SQLite              |
|--------------------------|---------------------|--------------------|----------------------|---------------------|
| Case-insensitive match   | `ILIKE`             | `LIKE` (default CI) | `LIKE` (collation)   | `LIKE` (ASCII CI)   |
| Regex                    | `~`                 | `REGEXP`           | not native           | `REGEXP` (extension) |
| Row limit                | `LIMIT n`           | `LIMIT n`          | `SELECT TOP n`       | `LIMIT n`           |

- `LIKE` patterns with `%` and `_` behave the same everywhere. Escape a literal `_` or `%` if needed.
- `$` in product names (e.g. `$500`) is plain text in SQL strings, with no escaping required, but watch for shell or templating layers that expand `$`.

## Things to keep in mind

- This table has **no personal information beyond a member number and product**. Don't invent columns such as name, DOB, premium or state; if the user wants them, ask for the schema or a joined table.
- The guide says an Eligibility Check should be done before admission to confirm cover. This table is not proof of current coverage, so don't present results as an eligibility confirmation.
- Product names are a snapshot from the 2 March 2026 guide and may change. If a product isn't recognised, report it rather than guessing its tier.
- The data may be mock/test data. Don't present counts as real membership figures unless the user says otherwise.

# BlackOut — Gameplay

## Overview

BlackOut is a 2-team competitive game. Each team controls 5 units on a procedurally generated 24×24 grid map. The goal is to accumulate more **power** (score) than the opponent by collecting Batteries and depositing them into your team's storage before a **storage absorption** event permanently locks in the score.

## Win Condition

The game ends when either:
- A team reaches **100 points**, or
- **7 minutes** have elapsed

The team with the higher score wins.

---

## Map

The map is a 24×24 tile grid, procedurally generated each episode from a seed. It contains the following zones:

### Research Base
Each team's spawn point. Enemies cannot enter. Contains one protected storage and a **base shrine** for Carrier transformation.

### Storage (Battery Cluster)
Items deposited here become **owned** by that team and apply their effects. Each team has **4 storage areas** — one inside the Research Base (protected) and 3 scattered across the open map (raidable).

Items are stacked automatically on deposit using a priority order based on distance from the spawn point (farthest tiles first). An item is only deposited if it fits entirely; otherwise the unit passes through without depositing.

### Central Shrine (장비창)
Located at the center of the map. A **Worker** that enters here transforms into a **Guard**.

### Base Shrine
Located inside the Research Base. A **Worker** that enters here transforms into a **Carrier**.

---

## Units

All units start as **Workers** and can transform by entering shrine zones. Transformation only works from the Worker class — Guards and Carriers that enter a shrine are unaffected. All units respawn as Workers after death, and this is the **only** way for a Guard or Carrier to revert to Worker.

| Class (asset name) | Collision Width | Base Speed | Can Collect | Kills on contact |
|---|---|---|---|---|
| Worker (Collector) | 0.55 | 4 | Yes | Carrier |
| Guard (Hunter) | 0.65 | 6 | No | Worker, Carrier; mutual kill vs Guard |
| Carrier | 0.45 | 6 | Yes | — |

### Worker
The default unit form. Can collect and deposit items but cannot defeat Guards. Transforms into Guard at the central shrine or into Carrier at the base shrine.

### Guard
A combat unit. Kills any enemy on contact (Workers, Carriers, and other Guards — though Guard vs Guard results in both dying). Cannot collect items. Use Guards to protect your own storage or raid the enemy's.

### Carrier
A fast, small specialist for collecting items. Dies on contact with **any** enemy unit. Only **1 Carrier per team** can exist at a time — the base shrine is disabled while one is alive.

---

## Items

All buff/debuff effects are **active only while the item sits in storage**. Removing or stealing the item immediately cancels the effect.

| Item | Effect | Target |
|---|---|---|
| **Battery** | Grants points equal to item amount on deposit; permanently locked in on absorption | — |
| **BuffSpeed** | Speed +50% while this item is in allied storage | All ally units |
| **DebuffSpeed** | Speed −90% while this item is in allied storage | Enemy Worker units only |
| **BuffSize** | Size +50% while this item is in allied storage | All ally units |
| **DebuffSize** | Size −30% while this item is in allied storage | All enemy units |

Batteries are present from the start of each episode (total ~200 score worth, spread across the map). Special items (BuffSpeed, DebuffSpeed, BuffSize, DebuffSize) spawn one at a time on neutral tiles, with a **10-second respawn cooldown** after the previous one is removed.

---

## Storage Absorption

Every **20 seconds**, a **storage absorption** event fires:

- All items in every storage are consumed.
- Non-battery items disappear (their effects end).
- Battery score is **permanently added** to the owning team's total and cannot be stolen afterward.

To steal batteries, you must raid the enemy's storage **before** the next absorption.

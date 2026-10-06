import os, json, warnings
from itertools import combinations
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from flask import Flask, jsonify, render_template, request
from fuzzywuzzy import process
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# ── CONFIG ──────────────────────────────────────────────────
DATA_CSV = "gameweek_32data.csv"
FIXTURES_CSV = "dgwfixtures2.csv"
STRENGTH_CSV = "team_difficulty.csv"

TEAM_MAP = {
    1: "Arsenal",
    2: "Aston Villa",
    3: "Burnley",
    4: "Bournemouth",
    5: "Brentford",
    6: "Brighton",
    7: "Chelsea",
    8: "Crystal Palace",
    9: "Everton",
    10: "Fulham",
    11: "Leeds",
    12: "Liverpool",
    13: "Man City",
    14: "Man Utd",
    15: "Newcastle",
    16: "Nottingham Forest",
    17: "Sunderland",
    18: "Tottenham Hotspurs",
    19: "West Ham",
    20: "Wolves"
}
POSITION_MAP = {1:"Goalkeeper", 2:"Defender", 3:"Midfielder", 4:"Forward"}
POS_ORDER = {"Goalkeeper":0, "Defender":1, "Midfielder":2, "Forward":3}

EXCLUDED_COLS = [
    'first_name','second_name','can_select','can_transact',
    'chance_of_playing_next_round','chance_of_playing_this_round','code',
    'corners_and_indirect_freekicks_text','cost_change_event','cost_change_event_fall',
    'cost_change_start','cost_change_start_fall','direct_freekicks_text',
    'has_temporary_code','news','news_added','penalties_text','photo',
    'removed','special','squad_number','status','transfers_in_event',
    'transfers_out_event','birth_date','team_join_date','opta_code','web_name'
]
TARGET_COL = "total_points"

# ── GLOBALS ──────────────────────────────────────────────────
_players_base = None
_fixtures = None
_team_strengths = None
_min_val = None
_max_val = None

app = Flask(__name__)


# ════════════════════════════════════════════════════════════
#  STARTUP  — load data & train models
# ════════════════════════════════════════════════════════════

def _normalize(value, invert=False, low=0.7, high=1.3):
    v = float(np.clip(value, _min_val, _max_val))
    z = (v - _min_val) / (_max_val - _min_val)
    if invert:
        z = 1 - z
    return low + z * (high - low)


def _train_position_model(position):
    df_pos = _players_base[_players_base["position"] == position].copy()

    feature_cols = [
        c for c in df_pos.columns
        if c not in EXCLUDED_COLS + [TARGET_COL, "position"]
    ]
    X = df_pos[feature_cols].copy()
    y = df_pos[TARGET_COL]

    cat_cols = X.select_dtypes(include=["object", "bool"]).columns.tolist()
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    for c in cat_cols: X[c] = X[c].fillna("Unknown").astype(str)
    for c in num_cols: X[c] = X[c].fillna(0)

    X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
    X_full = X.copy()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)
    mae = mean_absolute_error(y_te, preds)

    print(f"  🌲 {position:12s} RF MAE = {mae:.2f}")

    # Predict on full dataset
    _players_base.loc[df_pos.index, "predicted_points"] = model.predict(X_full)


def init_app():
    global _players_base, _fixtures, _team_strengths, _min_val, _max_val

    print("📂  Loading CSV files …")
    _players_base = pd.read_csv(DATA_CSV, encoding="utf-8")
    _fixtures = pd.read_csv(FIXTURES_CSV)
    _team_strengths = pd.read_csv(STRENGTH_CSV).set_index("id")

    scols = ["strength_attack_home","strength_attack_away",
             "strength_defence_home","strength_defence_away"]
    _min_val = float(_team_strengths[scols].min().min())
    _max_val = float(_team_strengths[scols].max().max())

    _players_base = _players_base.dropna(subset=[TARGET_COL])
    _players_base["position"]     = _players_base["element_type"].map(POSITION_MAP)
    _players_base["cost_million"] = _players_base["now_cost"] / 10.0
    _players_base["predicted_points"] = 0.0

    print("🧠  Training position models …")
    for pos in ["Goalkeeper", "Defender", "Midfielder", "Forward"]:
        _train_position_model(pos)

    print("✅  Ready — visit http://localhost:5000\n")


# ════════════════════════════════════════════════════════════
#  FIXTURE HELPERS
# ════════════════════════════════════════════════════════════

def _player_points_for_gw(player_row, gw):

    tid  = player_row["team"]
    gw_f = _fixtures[
        ((_fixtures["team_h"] == tid) | (_fixtures["team_a"] == tid)) &
        (_fixtures["event"] == gw)
    ]
    if gw_f.empty:
        return 0.0

    base = player_row["predicted_points"] / max(gw - 1, 1)
    total = 0.0
    for _, fix in gw_f.iterrows():
        is_home = fix["team_h"] == tid
        opp = fix["team_a"] if is_home else fix["team_h"]
        if is_home:
            opp_att = _team_strengths.loc[opp, "strength_attack_away"]
            opp_def = _team_strengths.loc[opp, "strength_defence_away"]
        else:
            opp_att = _team_strengths.loc[opp, "strength_attack_home"]
            opp_def = _team_strengths.loc[opp, "strength_defence_home"]

        if player_row["position"] in ("Goalkeeper", "Defender"):
            diff = _normalize(opp_att, invert=True)
        else:
            diff = _normalize(opp_def, invert=True)

        total += base * diff
    return total


def _gw_breakdown(player_row, current_gw, n=5):
    return [round(_player_points_for_gw(player_row, gw), 2)
            for gw in range(current_gw, current_gw + n)]


def _fixture_string(team_id, gw):
    rows = _fixtures[
        ((_fixtures["team_h"] == team_id) | (_fixtures["team_a"] == team_id)) &
        (_fixtures["event"] == gw)
    ]
    if rows.empty:
        return "—"
    parts = []
    for _, f in rows.iterrows():
        if f["team_h"] == team_id:
            parts.append(f"{TEAM_MAP.get(f['team_a'], '?')} (H)")
        else:
            parts.append(f"{TEAM_MAP.get(f['team_h'], '?')} (A)")
    return " + ".join(parts)


def _add_fixture_cols(df, current_gw):
    df = df.copy()
    df["predicted_points_fixture_adj_gw"]  = 0.0
    df["predicted_points_fixture_adj_2gw"] = 0.0
    df["predicted_points_fixture_adj_7gw"] = 0.0
    for idx, row in df.iterrows():
        df.at[idx, "predicted_points_fixture_adj_gw"]  = _player_points_for_gw(row, current_gw)
        df.at[idx, "predicted_points_fixture_adj_2gw"] = sum(
            _player_points_for_gw(row, gw) for gw in range(current_gw, current_gw + 2))
        df.at[idx, "predicted_points_fixture_adj_7gw"] = sum(
            _player_points_for_gw(row, gw) for gw in range(current_gw, current_gw + 7))
    return df

def apply_injuries(df, injured_short=None, injured_long=None):
    df = df.copy()

    injured_short = injured_short or []
    injured_long = injured_long or []

    if "full_name" not in df.columns:
        df["full_name"] = df["first_name"] + " " + df["second_name"]

    for name in injured_short:
        mask = (
            df["full_name"].str.lower().str.contains(name.lower(), na=False) |
            df["web_name"].str.lower().str.contains(name.lower(), na=False)
        )
        df.loc[mask, "predicted_points_fixture_adj_gw"] = 0

    for name in injured_long:
        mask = (
            df["full_name"].str.lower().str.contains(name.lower(), na=False) |
            df["web_name"].str.lower().str.contains(name.lower(), na=False)
        )
        df.loc[mask, "predicted_points_fixture_adj_gw"] = 0
        df.loc[mask, "predicted_points_fixture_adj_2gw"] = 0
        df.loc[mask, "predicted_points_fixture_adj_7gw"] = 0

    return df

# ════════════════════════════════════════════════════════════
#  TRANSFER OPTIMIZER
# ════════════════════════════════════════════════════════════

def _find_best_transfers(my_team, candidates, free_transfers, money_in_bank):
    base_2gw = my_team["predicted_points_fixture_adj_2gw"].sum()
    base_7gw = my_team["predicted_points_fixture_adj_7gw"].sum()
    base_pts = 0.5 * base_2gw + 0.5 * base_7gw
    base_cost = my_team["cost_million"].sum()

    c_cost = candidates["cost_million"]
    c_2gw = candidates["predicted_points_fixture_adj_2gw"]
    c_7gw = candidates["predicted_points_fixture_adj_7gw"]
    c_pos = candidates["position"]
    c_team = candidates["team"]

    moves = []

    for n_out in range(1, free_transfers + 1):
        out_pool = my_team.sort_values("predicted_points_fixture_adj_7gw").head(8)

        for out_idx in combinations(out_pool.index, n_out):
            out_cost = my_team.loc[list(out_idx), "cost_million"].sum()
            out_2gw = my_team.loc[list(out_idx), "predicted_points_fixture_adj_2gw"].sum()
            out_7gw = my_team.loc[list(out_idx), "predicted_points_fixture_adj_7gw"].sum()
            budget = money_in_bank + out_cost

            remain = my_team.drop(list(out_idx), errors="ignore")
            pos_after = remain["position"].value_counts().to_dict()
            clu_after = remain["team"].value_counts().to_dict()

            out_pos_cnt = my_team.loc[list(out_idx), "position"].value_counts().to_dict()

            for in_idx in combinations(candidates.index, n_out):
                if c_pos.loc[list(in_idx)].value_counts().to_dict() != out_pos_cnt:
                    continue
                if c_cost.loc[list(in_idx)].sum() > budget:
                    continue

                ok = True
                for p in in_idx:
                    pos = c_pos.loc[p]; tm = c_team.loc[p]
                    if pos == "Goalkeeper"  and pos_after.get(pos, 0) >= 2: ok=False; break
                    if pos == "Defender"   and pos_after.get(pos, 0) >= 5: ok=False; break
                    if pos == "Midfielder" and pos_after.get(pos, 0) >= 5: ok=False; break
                    if pos == "Forward"    and pos_after.get(pos, 0) >= 3: ok=False; break
                    if clu_after.get(tm, 0) >= 3: ok=False; break
                if not ok:
                    continue

                in_2gw = c_2gw.loc[list(in_idx)].sum()
                in_7gw = c_7gw.loc[list(in_idx)].sum()
                new_pts = 0.5 * (base_2gw - out_2gw + in_2gw) + 0.5 * (base_7gw - out_7gw + in_7gw)
                gain = new_pts - base_pts

                if gain > 0:
                    new_cost = base_cost - out_cost + c_cost.loc[list(in_idx)].sum()
                    moves.append((gain, out_idx, in_idx, new_pts, new_cost))

    return sorted(moves, key=lambda x: x[0], reverse=True)


# ════════════════════════════════════════════════════════════
#  CHIP STRATEGY ADVISOR
# ════════════════════════════════════════════════════════════

def _compute_chip_advice(my_team, all_players, budget, current_gw, chips_used):
    advice = []
    used = set(c.lower() for c in chips_used)

    # ── helpers ───────────────────────────────────────────
    sorted_team = my_team.sort_values("predicted_points_fixture_adj_gw", ascending=False)
    top_player = sorted_team.iloc[0] if len(sorted_team) else None
    bench_4 = sorted_team.iloc[11:15] if len(sorted_team) >= 15 else sorted_team.iloc[min(11, len(sorted_team)):]
    bench_score = float(bench_4["predicted_points_fixture_adj_gw"].sum()) if not bench_4.empty else 0

    # ── Triple Captain ────────────────────────────────────
    if "triple captain" in used:
        advice.append({"chip":"Triple Captain","action":"Already used","reason":"You've used this chip.","urgency":"used","icon":"3️⃣"})
    else:
        cap_pts = float(top_player["predicted_points_fixture_adj_gw"]) if top_player is not None else 0
        if cap_pts >= 7.5:
            advice.append({
                "chip":"Triple Captain","action":"USE NOW","urgency":"use_now","icon":"3️⃣",
                "reason":f"{top_player['web_name']} is projected {cap_pts:.1f} pts — tripling him returns ~{cap_pts*3:.1f} pts. Excellent candidate."
            })
        elif cap_pts >= 6:
            advice.append({
                "chip":"Triple Captain","action":"Consider using","urgency":"consider","icon":"3️⃣",
                "reason":f"{top_player['web_name']} at {cap_pts:.1f} projected pts is decent but wait for a bigger fixture if possible."
            })
        else:
            advice.append({
                "chip":"Triple Captain","action":"Save it","urgency":"save","icon":"3️⃣",
                "reason":f"Top captain ({top_player['web_name'] if top_player is not None else '?'}) only projects {cap_pts:.1f} pts. Hold for a better week."
            })

    # ── Bench Boost ───────────────────────────────────────
    if "bench boost" in used:
        advice.append({"chip":"Bench Boost","action":"Already used","reason":"You've used this chip.","urgency":"used","icon":"📦"})
    else:
        if bench_score >= 18:
            advice.append({
                "chip":"Bench Boost","action":"USE NOW","urgency":"use_now","icon":"📦",
                "reason":f"Bench projected {bench_score:.1f} pts — that's outstanding value. Activate this week."
            })
        elif bench_score >= 12:
            advice.append({
                "chip":"Bench Boost","action":"Consider using","urgency":"consider","icon":"📦",
                "reason":f"Bench at {bench_score:.1f} projected pts is solid. Worth using if your starters also have good fixtures."
            })
        else:
            advice.append({
                "chip":"Bench Boost","action":"Save it","urgency":"save","icon":"📦",
                "reason":f"Bench only projects {bench_score:.1f} pts. Wait until you have a stronger 4th–15th before activating."
            })

    # ── Free Hit ─────────────────────────────────────────
    if "free hit" in used:
        advice.append({"chip":"Free Hit","action":"Already used","reason":"You've used this chip.","urgency":"used","icon":"🆓"})
    else:
        team_gw = float(my_team["predicted_points_fixture_adj_gw"].sum())
        # Check if there's a blank/double gameweek indication — use fixture count
        teams_with_fixtures = set()
        if _fixtures is not None:
            gw_fix = _fixtures[_fixtures["event"] == current_gw]
            teams_with_fixtures = set(gw_fix["team_h"].tolist() + gw_fix["team_a"].tolist())
        my_team_ids = set(my_team["team"].tolist())
        blanking = len(my_team_ids - teams_with_fixtures)
        if blanking >= 3:
            advice.append({
                "chip":"Free Hit","action":"USE NOW","urgency":"use_now","icon":"🆓",
                "reason":f"{blanking} of your players have no fixture this GW. Free Hit lets you field a full team without long-term commitment."
            })
        elif team_gw < 45:
            advice.append({
                "chip":"Free Hit","action":"Consider using","urgency":"consider","icon":"🆓",
                "reason":f"Your team projects {team_gw:.1f} pts this GW — below average. A Free Hit could significantly improve this week."
            })
        else:
            advice.append({
                "chip":"Free Hit","action":"Save it","urgency":"save","icon":"🆓",
                "reason":f"Your team looks fine this GW ({team_gw:.1f} pts projected). Save for a blank GW or a week with 3+ of your players missing fixtures."
            })

    # ── Wildcard ─────────────────────────────────────────
    if "wildcard" in used:
        advice.append({"chip":"Wildcard","action":"Already used","reason":"You've used this chip.","urgency":"used","icon":"🃏"})
    else:
        # Count how many of your players are in the bottom 80% of the league by 7GW pts
        all_7gw = all_players["predicted_points_fixture_adj_7gw"].dropna()
        low_threshold = all_7gw.quantile(0.8)
        weak_players = int((my_team["predicted_points_fixture_adj_7gw"] < low_threshold).sum())
        team_7gw = float(my_team["predicted_points_fixture_adj_7gw"].sum())
        if weak_players >= 6:
            advice.append({
                "chip":"Wildcard","action":"USE NOW","urgency":"use_now","icon":"🃏",
                "reason":f"{weak_players} of your 15 players rank in the bottom 70% for next 7 GW points. A full rebuild is strongly recommended."
            })
        elif weak_players >= 4:
            advice.append({
                "chip":"Wildcard","action":"Consider using","urgency":"consider","icon":"🃏",
                "reason":f"{weak_players} players underperforming long-term. Consider wildcarding if you have 2+ expensive/difficult transfers queued up."
            })
        else:
            advice.append({
                "chip":"Wildcard","action":"Save it","urgency":"save","icon":"🃏",
                "reason":f"Only {weak_players} weak players in your squad — regular transfers should be enough. Keep the wildcard for a schedule crisis."
            })

    return advice


# ════════════════════════════════════════════════════════════
#  ROUTES
# ════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/get_players")
def get_players():
    if _players_base is None:
        return jsonify([])
    players = _players_base.copy()

    players["full_name"] = players["first_name"] + " " + players["second_name"]

    return jsonify([
        {
            "name": row["full_name"],
            "web_name": row["web_name"],
            "position": row["position"]
        }
        for _, row in players.iterrows()
    ])


# ════════════════════════════════════════════════════════════
#  TEAM PICKERS + ENHANCED OUTPUT
# ════════════════════════════════════════════════════════════

def pick_best_starting_11(team_df):
    team_df = team_df.sort_values("predicted_points_fixture_adj_gw", ascending=False)

    gks = team_df[team_df["position"] == "Goalkeeper"]
    defs = team_df[team_df["position"] == "Defender"]
    mids = team_df[team_df["position"] == "Midfielder"]
    fwds = team_df[team_df["position"] == "Forward"]

    if gks.empty or len(defs) < 3 or len(mids) < 3 or fwds.empty:
        return team_df.head(11)

    min_lineup = pd.concat([
        gks.iloc[[0]],
        defs.iloc[:3],
        mids.iloc[:3],
        fwds.iloc[[0]]
    ])

    counts = {"Goalkeeper": 1, "Defender": 3, "Midfielder": 3, "Forward": 1}
    max_counts = {"Goalkeeper": 1, "Defender": 5, "Midfielder": 5, "Forward": 3}


    remaining = team_df[~team_df.index.isin(min_lineup.index)]
    extras = []
    for _, row in remaining.iterrows():
        if len(extras) == 3:
            break
        pos = row["position"]
        if counts[pos] < max_counts[pos]:
            extras.append(row)
            counts[pos] += 1

    if extras:
        result = pd.concat([min_lineup, pd.DataFrame(extras)])
    else:
        result = min_lineup

    return result

def pick_bench(team_df, starting_df):
    bench = team_df.drop(starting_df.index)
    bench = bench.sort_values("predicted_points_fixture_adj_gw", ascending=False)
    return bench


def get_captains(df):
    df = df.sort_values("predicted_points_fixture_adj_gw", ascending=False)
    captain = df.iloc[0]
    vice = df.iloc[1] if len(df) > 1 else captain
    return captain, vice


def get_formation(df):
    counts = df["position"].value_counts()
    return f"{counts.get('Defender',0)}-{counts.get('Midfielder',0)}-{counts.get('Forward',0)}"


def build_best_team(players, budget, pts_col):
    players = players.sort_values(pts_col, ascending=False)

    squad = []
    cost = 0
    club_count = {}
    pos_limits = {"Goalkeeper":2,"Defender":5,"Midfielder":5,"Forward":3}
    pos_count = {k:0 for k in pos_limits}

    for _, p in players.iterrows():
        pos = p["position"]
        team = p["team"]
        price = p["cost_million"]

        if pos_count[pos] >= pos_limits[pos]:
            continue
        if club_count.get(team, 0) >= 3:
            continue
        if cost + price > budget:
            continue

        squad.append(p)
        pos_count[pos] += 1
        club_count[team] = club_count.get(team, 0) + 1
        cost += price

        if len(squad) == 15:
            break

    squad_df = pd.DataFrame(squad)

    xi = pick_best_starting_11(squad_df)
    bench = pick_bench(squad_df, xi)
    cap, vc = get_captains(xi)

    return {
        "squad": squad_df,
        "xi": xi,
        "bench": bench,
        "captain": cap,
        "vice": vc,
        "formation": get_formation(xi)
    }


def format_players(df, pts_col):
    return [
        {
            "name": row["first_name"] + " " + row["second_name"],
            "position": row["position"],
            "pts": round(float(row[pts_col]), 2),
            "cost": round(float(row["cost_million"]), 1)
        }
        for _, row in df.iterrows()
    ]

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    current_gw = int(data["current_gw"])
    free_xfer = int(data["free_transfers"])
    bank = float(data["money_in_bank"])
    raw_names = [n.strip() for n in data.get("team_names", []) if n.strip()]
    chips_used = [c.lower().strip() for c in data.get("chips_used", [])]
    injured_short = data.get("injured_short", [])
    injured_long = data.get("injured_long", [])

    # ── match names ──────────────────────────────────────────
    _players_base["full_name"] = _players_base["first_name"] + " " + _players_base["second_name"]
    all_names = _players_base["full_name"].tolist()
    matched = []
    match_log = []
    for raw in raw_names:
        result = process.extractOne(raw, all_names)
        if result and result[1] >= 65:
            matched.append(result[0])
            if result[0].lower() != raw.lower():
                match_log.append(f'"{raw}" → "{result[0]}"')
        else:
            match_log.append(f'"{raw}" — NOT FOUND (score {result[1] if result else 0})')

    # ── build fixture-adj dataset ────────────────────────────
    players = _add_fixture_cols(_players_base, current_gw)
    players["cost_million"] = players["now_cost"] / 10.0  # ensure fresh

    my_team = players[players["full_name"].isin(matched)].copy()
    my_team = apply_injuries(my_team, injured_short, injured_long)
    other_players = players[~players.index.isin(my_team.index)].copy()

    if my_team.empty:
        return jsonify({"error": "No players matched. Check spellings."})

    # de-dupe my_team index (safety)
    my_team.index = list(range(len(my_team)))
    players.index = list(range(len(players)))

    team_value = my_team["cost_million"].sum()
    budget = team_value + bank

    # ── candidates ───────────────────────────────────────────
    cands = pd.concat([
        other_players[other_players["position"] == pos]
        .nlargest(20, "predicted_points_fixture_adj_7gw")
        for pos in ["Goalkeeper", "Defender", "Midfielder", "Forward"]
    ])

    # ── transfers ────────────────────────────────────────────
    raw_moves = _find_best_transfers(my_team, cands, free_xfer, bank)

    gw_labels = [f"GW{current_gw + i}" for i in range(5)]

    def player_dict(row, df_source):
        breakdown = _gw_breakdown(row, current_gw, 5)
        full = row["first_name"] + " " + row["second_name"]
        web = row.get("web_name", "")

        def is_injured(lst):
            return any(n.lower() in full.lower() or n.lower() in web.lower() for n in lst)

        if is_injured(injured_long):
            breakdown = [0.0] * 5
        elif is_injured(injured_short):
            breakdown[0] = 0.0
        return {
            "name": full,
            "position": row["position"],
            "cost": round(float(row["cost_million"]), 1),
            "pred_7gw": round(float(row["predicted_points_fixture_adj_7gw"]), 2),
            "gw_breakdown": breakdown,
            "fixtures": [_fixture_string(row["team"], current_gw + i) for i in range(5)],
        }

    transfers = []
    for gain, out_idx, in_idx, new_pts, new_cost in raw_moves[:5]:
        out_list = [player_dict(my_team.loc[p], my_team) for p in out_idx]
        in_list = [player_dict(cands.loc[p],   cands)   for p in in_idx]
        transfers.append({
            "gain":     round(float(gain), 2),
            "out":      out_list,
            "in":       in_list,
            "new_total": round(float(new_pts), 2),
            "new_cost":  round(float(new_cost), 1),
        })

    # ── team overview (sorted by position then pts) ───────────
    pos_key = my_team["position"].map(POS_ORDER)
    team_sorted = my_team.assign(_pk=pos_key).sort_values(
        ["_pk", "predicted_points_fixture_adj_gw"], ascending=[True, False]
    )
    team_overview = []
    for _, row in team_sorted.iterrows():
        team_overview.append({
            "name":     row["web_name"],
            "position": row["position"],
            "cost":     round(float(row["cost_million"]), 1),
            "gw_pts":   round(float(row["predicted_points_fixture_adj_gw"]), 2),
            "fixture":  _fixture_string(row["team"], current_gw),
        })

    # ── captain ──────────────────────────────────────────────
    cap_idx = my_team["predicted_points_fixture_adj_gw"].idxmax()
    cap_row = my_team.loc[cap_idx]
    vc_df = my_team.drop(cap_idx)
    vc_idx = vc_df["predicted_points_fixture_adj_gw"].idxmax()
    vc_row = vc_df.loc[vc_idx]

    # ── chip strategy advice ─────────────────────────────────
    chip_advice = _compute_chip_advice(my_team, players, budget, current_gw, chips_used)

    # ── CURRENT TEAM XI ─────────────────────────────
    xi_df = pick_best_starting_11(my_team)
    bench_df = pick_bench(my_team, xi_df)
    cap, vc = get_captains(xi_df)

    best_xi = format_players(xi_df, "predicted_points_fixture_adj_gw")
    bench = format_players(bench_df, "predicted_points_fixture_adj_gw")

    # ── FREE HIT TEAM ───────────────────────────────
    fh = build_best_team(players, budget, "predicted_points_fixture_adj_gw")

    free_hit = {
        "xi": format_players(fh["xi"], "predicted_points_fixture_adj_gw"),
        "bench": format_players(fh["bench"], "predicted_points_fixture_adj_gw"),
        "captain": fh["captain"]["first_name"] + " " + fh["captain"]["second_name"],
        "vice": fh["vice"]["first_name"] + " " + fh["vice"]["second_name"],
        "formation": fh["formation"]
    }

    # ── WILDCARD TEAM ───────────────────────────────
    wc = build_best_team(players, budget, "predicted_points_fixture_adj_7gw")

    wildcard = {
        "xi": format_players(wc["xi"], "predicted_points_fixture_adj_7gw"),
        "bench": format_players(wc["bench"], "predicted_points_fixture_adj_7gw"),
        "captain": wc["captain"]["first_name"] + " " + wc["captain"]["second_name"],
        "vice": wc["vice"]["first_name"] + " " + wc["vice"]["second_name"],
        "formation": wc["formation"]
    }

    return jsonify({
        "transfers": transfers,
        "team_overview": team_overview,
        "captain": cap_row["web_name"],
        "captain_pts": round(float(cap_row["predicted_points_fixture_adj_gw"]), 2),
        "vice_captain": vc_row["web_name"],
        "vc_pts": round(float(vc_row["predicted_points_fixture_adj_gw"]), 2),
        "budget": round(float(budget), 1),
        "team_value": round(float(team_value), 1),
        "current_gw": current_gw,
        "matched": len(my_team),
        "match_log": match_log,
        "gw_labels": gw_labels,
        "chips_used": chips_used,
        "chip_advice": chip_advice,
        "best_xi": best_xi,
        "bench": bench,
        "captain_pick": cap["first_name"] + " " + cap["second_name"],
        "vice_pick": vc["first_name"] + " " + vc["second_name"],
        "formation": get_formation(xi_df),

        "free_hit_team": free_hit,
        "wildcard_team": wildcard,
    })


# ════════════════════════════════════════════════════════════
#  ENTRY POINT
# ════════════════════════════════════════════════════════════
if __name__ == "__main__":
    init_app()
    app.run(debug=False, port=5000, use_reloader=False)
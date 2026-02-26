import os
import sqlite3
from datetime import datetime
import pandas as pd
import streamlit as st

DB_PATH = os.environ.get("BETJOURNAL_DB", "betting_journal.sqlite3")
DATE_FMT = "%d.%m.%Y"
DT_FMT = "%d.%m.%Y %H:%M"
INITIAL_BANKROLL_DEFAULT = 20000.0

RESULTS = ["pending", "won", "lost", "void"]
BET_TYPES = ["single", "combo", "system"]
REASONS = ["analysis", "tip", "intuition"]
MENTAL_STATES = ["calm", "nervous", "emotional"]
EMOTIONAL_AFTER = ["", "calm", "relieved", "happy", "angry", "tilted", "disappointed", "neutral"]


# ---------- helpers ----------
def fmt_kc(x):
    try:
        s = f"{float(x):,.2f}".replace(",", "X").replace(".", ",").replace("X", " ")
        return f"{s} Kč"
    except Exception:
        return ""

def parse_float(s, field):
    s = str(s).strip()
    if s == "":
        raise ValueError(f"{field}: povinné")
    s2 = s.replace(" ", "").replace(",", ".")
    try:
        return float(s2)
    except ValueError:
        raise ValueError(f"{field}: neplatné číslo")

def parse_date(s):
    return datetime.strptime(str(s).strip(), DATE_FMT)

def parse_datetime(date_str, time_str):
    d = parse_date(date_str)
    t = str(time_str).strip()
    if t == "":
        now = datetime.now()
        return d.replace(hour=now.hour, minute=now.minute, second=0, microsecond=0)
    try:
        hh, mm = t.split(":")
        hh = int(hh); mm = int(mm)
        if not (0 <= hh <= 23 and 0 <= mm <= 59):
            raise ValueError
    except Exception:
        raise ValueError("Čas: použij HH:MM (např. 19:30)")
    return d.replace(hour=hh, minute=mm, second=0, microsecond=0)

def calc_profit_loss(result, odds, stake):
    if result == "won":
        return stake * (odds - 1.0)
    if result == "lost":
        return -stake
    return 0.0


# ---------- DB ----------
def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS settings(
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    );
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS bets(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        dt_utc TEXT NOT NULL,
        sport TEXT NOT NULL,
        competition TEXT NOT NULL,
        bet_type TEXT NOT NULL CHECK(bet_type IN ('single','combo','system')),
        odds REAL NOT NULL CHECK(odds > 0),
        stake REAL NOT NULL CHECK(stake >= 0),
        result TEXT NOT NULL CHECK(result IN ('pending','won','lost','void')),
        profit_loss REAL NOT NULL,
        reason TEXT NOT NULL CHECK(reason IN ('analysis','tip','intuition')),
        mental_before TEXT NOT NULL CHECK(mental_before IN ('calm','nervous','emotional')),
        emotional_after TEXT,
        bankroll_before REAL NOT NULL,
        bankroll_after REAL NOT NULL,
        pct_bankroll REAL NOT NULL,
        notes TEXT
    );
    """)
    conn.commit()

    cur.execute("SELECT value FROM settings WHERE key='initial_bankroll'")
    row = cur.fetchone()
    if not row:
        cur.execute("INSERT INTO settings(key,value) VALUES('initial_bankroll', ?)", (str(INITIAL_BANKROLL_DEFAULT),))
        conn.commit()
    conn.close()

def get_initial_bankroll():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT value FROM settings WHERE key='initial_bankroll'")
    row = cur.fetchone()
    conn.close()
    try:
        return float(row["value"]) if row else INITIAL_BANKROLL_DEFAULT
    except Exception:
        return INITIAL_BANKROLL_DEFAULT

def set_initial_bankroll(v: float):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
    INSERT INTO settings(key,value) VALUES('initial_bankroll',?)
    ON CONFLICT(key) DO UPDATE SET value=excluded.value
    """, (str(v),))
    conn.commit()
    conn.close()

def fetch_bets(filters=None):
    filters = filters or {}
    where = []
    params = []

    if filters.get("date_from"):
        dt_from = parse_date(filters["date_from"]).isoformat(timespec="seconds")
        where.append("dt_utc >= ?")
        params.append(dt_from)
    if filters.get("date_to"):
        dt_to = parse_date(filters["date_to"]).replace(hour=23, minute=59, second=59).isoformat(timespec="seconds")
        where.append("dt_utc <= ?")
        params.append(dt_to)

    if filters.get("sport") and filters["sport"] != "All":
        where.append("sport = ?")
        params.append(filters["sport"])
    if filters.get("result") and filters["result"] != "All":
        where.append("result = ?")
        params.append(filters["result"])
    if filters.get("bet_type") and filters["bet_type"] != "All":
        where.append("bet_type = ?")
        params.append(filters["bet_type"])

    if filters.get("search"):
        q = f"%{filters['search'].strip()}%"
        if q != "%%":
            where.append("(competition LIKE ? OR notes LIKE ?)")
            params.extend([q, q])

    sql = "SELECT * FROM bets"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY dt_utc ASC"

    conn = get_conn()
    df = pd.read_sql_query(sql, conn, params=params)
    conn.close()
    return df

def add_bet_row(data: dict):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
    INSERT INTO bets(
        dt_utc, sport, competition, bet_type, odds, stake, result, profit_loss,
        reason, mental_before, emotional_after, bankroll_before, bankroll_after, pct_bankroll, notes
    ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        data["dt_utc"], data["sport"], data["competition"], data["bet_type"],
        data["odds"], data["stake"], data["result"], data["profit_loss"],
        data["reason"], data["mental_before"], data.get("emotional_after"),
        data["bankroll_before"], data["bankroll_after"], data["pct_bankroll"],
        data.get("notes"),
    ))
    conn.commit()
    conn.close()

def update_result(bet_id: int, result: str, emotional_after: str):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM bets WHERE id=?", (bet_id,))
    r = cur.fetchone()
    if not r:
        conn.close()
        raise ValueError("Záznam nenalezen.")

    odds = float(r["odds"])
    stake = float(r["stake"])
    pl = calc_profit_loss(result, odds, stake)
    b_before = float(r["bankroll_before"])
    b_after = b_before + pl

    cur.execute("""
    UPDATE bets SET result=?, profit_loss=?, emotional_after=?, bankroll_after=?
    WHERE id=?
    """, (result, float(pl), emotional_after or None, float(b_after), bet_id))
    conn.commit()
    conn.close()

def delete_bet(bet_id: int):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM bets WHERE id=?", (bet_id,))
    conn.commit()
    conn.close()

def rebuild_chain():
    initial = get_initial_bankroll()
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM bets ORDER BY dt_utc ASC")
    rows = cur.fetchall()

    bankroll = initial
    for r in rows:
        b_before = bankroll
        b_after = bankroll + float(r["profit_loss"])
        pct = (float(r["stake"]) / b_before) if b_before > 0 else 0.0
        cur.execute("""
        UPDATE bets SET bankroll_before=?, bankroll_after=?, pct_bankroll=?
        WHERE id=?
        """, (float(b_before), float(b_after), float(pct), int(r["id"])))
        bankroll = b_after

    conn.commit()
    conn.close()


# ---------- metrics ----------
def compute_metrics(df: pd.DataFrame, initial_bankroll: float):
    if df.empty:
        return {
            "count": 0, "wins": 0, "losses": 0, "total_stake": 0.0, "total_pl": 0.0,
            "roi": 0.0, "strike": 0.0, "avg_odds": 0.0,
            "biggest_win": 0.0, "biggest_loss": 0.0,
            "best_win_streak": 0, "best_loss_streak": 0,
            "current_streak": ("none", 0),
            "current_bankroll": initial_bankroll
        }

    total_stake = df["stake"].sum()
    total_pl = df["profit_loss"].sum()

    wl = df[df["result"].isin(["won", "lost"])].copy()
    wins = int((wl["result"] == "won").sum())
    losses = int((wl["result"] == "lost").sum())

    roi = (total_pl / total_stake) if total_stake > 0 else 0.0
    strike = (wins / (wins + losses)) if (wins + losses) > 0 else 0.0
    avg_odds = float(df["odds"].mean())

    biggest_win = float(df["profit_loss"].max())
    biggest_loss = float(df["profit_loss"].min())

    # streaks
    best_win = best_loss = 0
    cur_win = cur_loss = 0
    for res in df["result"].tolist():
        if res == "won":
            cur_win += 1; cur_loss = 0
        elif res == "lost":
            cur_loss += 1; cur_win = 0
        else:
            continue
        best_win = max(best_win, cur_win)
        best_loss = max(best_loss, cur_loss)

    current_streak = ("none", 0)
    for res in reversed(df["result"].tolist()):
        if res not in ("won", "lost"):
            continue
        n = 0
        for rr in reversed(df["result"].tolist()):
            if rr == res:
                n += 1
            elif rr in ("won", "lost"):
                break
        current_streak = (res, n)
        break

    current_bankroll = initial_bankroll + total_pl

    return {
        "count": int(len(df)),
        "wins": wins,
        "losses": losses,
        "total_stake": float(total_stake),
        "total_pl": float(total_pl),
        "roi": float(roi),
        "strike": float(strike),
        "avg_odds": float(avg_odds),
        "biggest_win": biggest_win,
        "biggest_loss": biggest_loss,
        "best_win_streak": int(best_win),
        "best_loss_streak": int(best_loss),
        "current_streak": current_streak,
        "current_bankroll": float(current_bankroll),
    }


# ---------- Streamlit UI ----------
st.set_page_config(page_title="Betting Journal (CZK)", layout="wide")
init_db()

st.title("Betting Journal (CZK) – web app")
st.caption("SQLite + filtry + export + grafy. Datum: DD.MM.YYYY. Částky v Kč.")

with st.sidebar:
    st.header("Nastavení")
    init_br = st.number_input("Initial bankroll (Kč)", min_value=0.0, value=float(get_initial_bankroll()), step=100.0)
    if st.button("Uložit initial bankroll"):
        set_initial_bankroll(float(init_br))
        st.success("Uloženo.")

    st.divider()
    st.header("Filtry")
    f_date_from = st.text_input("Od (DD.MM.YYYY)", value="")
    f_date_to = st.text_input("Do (DD.MM.YYYY)", value="")
    f_sport = st.selectbox("Sport", ["All"], index=0)
    f_result = st.selectbox("Výsledek", ["All"] + RESULTS, index=0)
    f_bet_type = st.selectbox("Typ", ["All"] + BET_TYPES, index=0)
    f_search = st.text_input("Hledat (liga / poznámky)", value="")

    st.divider()
    if st.button("Přepočítat bankroll chain (override)", type="secondary"):
        rebuild_chain()
        st.success("Hotovo. (Přepočítá bankroll_before/after dle historie P/L)")

# Load data (first fetch without sport list)
filters = {}
try:
    if f_date_from.strip():
        parse_date(f_date_from.strip())
        filters["date_from"] = f_date_from.strip()
    if f_date_to.strip():
        parse_date(f_date_to.strip())
        filters["date_to"] = f_date_to.strip()
except Exception as e:
    st.error(f"Chyba ve filtru data: {e}")

filters["result"] = f_result
filters["bet_type"] = f_bet_type
filters["search"] = f_search.strip()

df_all = fetch_bets(filters=filters)

# fill sport select options from db + keep current selection
sports = sorted([s for s in df_all["sport"].dropna().unique().tolist()]) if not df_all.empty else []
with st.sidebar:
    cur = st.session_state.get("sport_choice", "All")
    sport_choice = st.selectbox("Sport (z DB)", ["All"] + sports, index=(["All"] + sports).index(cur) if cur in (["All"] + sports) else 0)
    st.session_state["sport_choice"] = sport_choice
filters["sport"] = sport_choice
df = fetch_bets(filters=filters)

initial_bankroll = float(get_initial_bankroll())
met = compute_metrics(df, initial_bankroll)

# --- dashboard ---
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Bankroll", fmt_kc(met["current_bankroll"]))
c2.metric("Total P/L", fmt_kc(met["total_pl"]))
c3.metric("ROI", f"{met['roi']*100:.2f}%")
c4.metric("Strike rate", f"{met['strike']*100:.2f}%")
c5.metric("Bets", str(met["count"]))

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Avg odds", f"{met['avg_odds']:.2f}")
c2.metric("Biggest win", fmt_kc(met["biggest_win"]))
c3.metric("Biggest loss", fmt_kc(met["biggest_loss"]))
c4.metric("Best win streak", str(met["best_win_streak"]))
streak = met["current_streak"]
c5.metric("Current streak", "-" if streak[0] == "none" else f"{streak[0]} × {streak[1]}")

st.divider()

# --- add bet form ---
st.subheader("➕ Přidat sázku")
with st.form("add_bet", clear_on_submit=True):
    colA, colB, colC, colD = st.columns(4)

    now = datetime.now()
    date_str = colA.text_input("Datum (DD.MM.YYYY)", value=now.strftime(DATE_FMT))
    time_str = colA.text_input("Čas (HH:MM)", value=now.strftime("%H:%M"))

    sport = colB.text_input("Sport", value="Football")
    competition = colB.text_input("Soutěž / Liga", value="")

    bet_type = colC.selectbox("Typ sázky", BET_TYPES, index=0)
    result = colC.selectbox("Výsledek", RESULTS, index=0)

    odds_s = colD.text_input("Kurz (odds)", value="")
    stake_s = colD.text_input("Stake (Kč)", value="")

    reason = st.selectbox("Důvod", REASONS, index=0)
    mental_before = st.selectbox("Mentální stav před", MENTAL_STATES, index=0)
    emotional_after = st.selectbox("Emoce po (volitelné)", EMOTIONAL_AFTER, index=0)
    notes = st.text_area("Poznámky (co fungovalo/nefungovalo, chyby, lessons learned)", value="", height=90)

    submit = st.form_submit_button("Uložit sázku", type="primary")

    if submit:
        try:
            dt = parse_datetime(date_str, time_str)
            if not sport.strip():
                raise ValueError("Sport: povinné")
            if not competition.strip():
                raise ValueError("Soutěž/Liga: povinné")

            odds = parse_float(odds_s, "Kurz")
            stake = parse_float(stake_s, "Stake")
            if odds <= 0:
                raise ValueError("Kurz musí být > 0")
            if stake < 0:
                raise ValueError("Stake musí být >= 0")

            # bankroll before = initial + total_pl (full history, not filtered)
            df_history = fetch_bets()
            bankroll_before = float(get_initial_bankroll()) + (float(df_history["profit_loss"].sum()) if not df_history.empty else 0.0)

            pl = calc_profit_loss(result, odds, stake)
            bankroll_after = bankroll_before + pl
            pct_bankroll = (stake / bankroll_before) if bankroll_before > 0 else 0.0

            add_bet_row({
                "dt_utc": dt.isoformat(timespec="seconds"),
                "sport": sport.strip(),
                "competition": competition.strip(),
                "bet_type": bet_type,
                "odds": float(odds),
                "stake": float(stake),
                "result": result,
                "profit_loss": float(pl),
                "reason": reason,
                "mental_before": mental_before,
                "emotional_after": (emotional_after.strip() or None),
                "bankroll_before": float(bankroll_before),
                "bankroll_after": float(bankroll_after),
                "pct_bankroll": float(pct_bankroll),
                "notes": (notes.strip() or None),
            })
            st.success("Uloženo. Aktualizuj stránku nebo použij 'Rerun' (většinou se refreshne automaticky).")
            st.rerun()
        except Exception as e:
            st.error(str(e))

st.divider()

# --- table view ---
st.subheader("📋 Záznamy (tabulka)")
if df.empty:
    st.info("Zatím žádné záznamy (nebo filtry nic nenašly).")
else:
    dft = df.copy()
    dft["dt"] = pd.to_datetime(dft["dt_utc"]).dt.strftime(DT_FMT)
    dft["stake_kc"] = dft["stake"].map(fmt_kc)
    dft["pl_kc"] = dft["profit_loss"].map(fmt_kc)
    dft["br_before_kc"] = dft["bankroll_before"].map(fmt_kc)
    dft["br_after_kc"] = dft["bankroll_after"].map(fmt_kc)
    dft["pct_br"] = (dft["pct_bankroll"] * 100).round(2).astype(str) + "%"

    show_cols = [
        "id", "dt", "sport", "competition", "bet_type", "odds",
        "stake_kc", "result", "pl_kc", "br_before_kc", "br_after_kc", "pct_br",
        "reason", "mental_before", "emotional_after", "notes"
    ]
    st.dataframe(dft[show_cols], use_container_width=True, hide_index=True)

# --- update / delete ---
st.subheader("✅ Update výsledku / 🗑️ Smazat")
col1, col2, col3 = st.columns(3)
bet_id = col1.number_input("ID záznamu", min_value=0, value=0, step=1)
new_result = col2.selectbox("Nový výsledek", RESULTS, index=0)
new_emo = col3.selectbox("Emoce po (volitelné)", EMOTIONAL_AFTER, index=0)

cA, cB = st.columns(2)
if cA.button("Uložit výsledek", type="primary"):
    try:
        if bet_id <= 0:
            raise ValueError("Zadej platné ID.")
        update_result(int(bet_id), new_result, new_emo.strip())
        st.success("Výsledek uložen.")
        st.rerun()
    except Exception as e:
        st.error(str(e))

if cB.button("Smazat záznam", type="secondary"):
    try:
        if bet_id <= 0:
            raise ValueError("Zadej platné ID.")
        delete_bet(int(bet_id))
        st.success("Smazáno.")
        st.rerun()
    except Exception as e:
        st.error(str(e))

st.divider()

# --- exports ---
st.subheader("⬇️ Export")
if not df.empty:
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button("Stáhnout CSV", data=csv_bytes, file_name="betting_journal.csv", mime="text/csv")

    # Excel export (in-memory)
    import io
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Bets")
    st.download_button(
        "Stáhnout Excel (XLSX)",
        data=output.getvalue(),
        file_name="betting_journal.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
else:
    st.caption("Export se objeví až po přidání záznamů.")

st.divider()

# --- charts ---
st.subheader("📈 Grafy")
df_hist = fetch_bets()  # full history for charts
if df_hist.empty:
    st.info("Grafy se zobrazí po prvních záznamech.")
else:
    dh = df_hist.copy()
    dh["dt"] = pd.to_datetime(dh["dt_utc"])
    dh = dh.sort_values("dt")

    initial = float(get_initial_bankroll())
    dh["bankroll"] = initial + dh["profit_loss"].cumsum()

    # ROI over time (cumulative)
    dh["stake_cum"] = dh["stake"].cumsum()
    dh["pl_cum"] = dh["profit_loss"].cumsum()
    dh["roi"] = dh.apply(lambda r: (r["pl_cum"] / r["stake_cum"]) if r["stake_cum"] > 0 else 0.0, axis=1)

    # Win rate (cumulative) based on won/lost only
    dh["is_w"] = (dh["result"] == "won").astype(int)
    dh["is_l"] = (dh["result"] == "lost").astype(int)
    dh["w_cum"] = dh["is_w"].cumsum()
    dh["l_cum"] = dh["is_l"].cumsum()
    dh["winrate"] = dh.apply(lambda r: (r["w_cum"] / (r["w_cum"] + r["l_cum"])) if (r["w_cum"] + r["l_cum"]) > 0 else 0.0, axis=1)

    c1, c2 = st.columns(2)
    c1.line_chart(dh.set_index("dt")["bankroll"], height=260)
    c2.line_chart((dh.set_index("dt")["roi"] * 100), height=260)

    st.line_chart((dh.set_index("dt")["winrate"] * 100), height=260)

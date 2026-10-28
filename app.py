import numpy as np, joblib, librosa, tempfile, os, random, json, hashlib, secrets, sqlite3, torch
from datetime import datetime
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

st.set_page_config(page_title="Sonar", layout="centered")

SR, CHUNK = 16000, 3.0
NAME = "facebook/wav2vec2-base"
PEACH, FLAMINGO, ROUGE = "#FEDDD4", "#EEA8C0", "#B03644"
MERLOT, FILM, CREAM, CHART = "#5A070B", "#200B14", "#FCF5F1", "#A5982D"
USERS_FILE, DB_FILE = "users.json", "sonar.db"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@1,400&display=swap');
.stApp, .stMarkdown, label, button {{ font-family: Arial, Helvetica, sans-serif; }}
footer {{ visibility: hidden; }}
.block-container {{ padding-top: 3.5rem !important; }}
h1, h2, h3 {{ font-family: Georgia, 'Times New Roman', serif !important;
              color: {MERLOT} !important; font-weight: 400 !important; }}
.bw {{ display: flex; justify-content: center; margin: 28px 0 18px 0; }}
.b1 {{ border: 6px dotted {FLAMINGO}; border-radius: 50%; padding: 6px; }}
.b2 {{ border: 1.5px solid {ROUGE}; border-radius: 50%; padding: 5px; }}
.b3 {{ border: 1px dotted {MERLOT}; border-radius: 50%; text-align: center; background: {CREAM}; }}
.bn {{ font-family: 'Playfair Display', Georgia, serif; font-style: italic;
       color: {MERLOT}; line-height: 1.05; }}
.bs {{ font-family: Georgia, serif; font-size: .62rem; letter-spacing: .26em;
       color: {FILM}; margin-top: 4px; }}
.bsince {{ font-family: Georgia, serif; font-size: .55rem; letter-spacing: .3em;
           color: {CHART}; margin-top: 2px; }}
.note {{ color: {FILM}; opacity: .75; font-size: .9rem; margin: 4px 0 10px 0; }}
.q {{ background: {PEACH}; padding: 18px 20px; border-radius: 2px;
      font-family: Georgia, serif; font-size: 1.3rem; color: {MERLOT}; margin: 8px 0 14px 0; }}
.panel {{ background: {PEACH}; padding: 18px 20px; border-radius: 2px; }}
.ptitle {{ font-family: Georgia, serif; font-size: 1.15rem; color: {MERLOT}; margin-bottom: 8px; }}
.step {{ margin: 6px 0; color: {FILM}; }}
.srow {{ display: flex; justify-content: space-between; padding: 7px 0;
         border-bottom: 1px solid {FLAMINGO}; color: {FILM}; }}
.srow b {{ color: {MERLOT}; font-weight: 600; text-transform: capitalize; }}
.row {{ display: flex; align-items: center; margin: 7px 0; }}
.lab {{ width: 90px; text-transform: capitalize; color: {FILM}; }}
.track {{ flex: 1; background: {PEACH}; height: 14px; }}
.fill {{ height: 14px; }}
.pct {{ width: 48px; text-align: right; color: {FILM}; }}
.stButton > button, .stFormSubmitButton > button {{
    background: {ROUGE}; color: {PEACH}; border: none; border-radius: 2px; box-shadow: none; }}
.stButton > button:hover, .stFormSubmitButton > button:hover {{
    background: {ROUGE}; color: {PEACH}; border: none; }}
.stButton > button:disabled {{ background: {PEACH}; color: {MERLOT}; opacity: 1; }}
</style>
""", unsafe_allow_html=True)

# ---------- model ----------
@st.cache_resource
def load_all():
    sc, clf = joblib.load("sonar_w2v.joblib")
    fe = Wav2Vec2FeatureExtractor.from_pretrained(NAME)
    enc = Wav2Vec2Model.from_pretrained(NAME).eval()
    return sc, clf, fe, enc

sc, clf, fe, enc = load_all()
classes = clf.classes_

def embed(y):
    inp = fe(y, sampling_rate=SR, return_tensors="pt")
    with torch.no_grad():
        h = enc(**inp).last_hidden_state[0]
    return torch.cat([h.mean(0), h.std(0)]).numpy()

def analyze(audio_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        f.write(audio_bytes); path = f.name
    y, _ = librosa.load(path, sr=SR)
    os.remove(path)
    n = int(CHUNK * SR)
    chunks = [y[i:i+n] for i in range(0, len(y), n) if len(y[i:i+n]) > n // 2]
    if not chunks:
        return None
    X = sc.transform(np.array([embed(c) for c in chunks]))
    return clf.predict_proba(X)

# ---------- accounts ----------
def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE) as f:
            return json.load(f)
    return {}

def hash_pw(pw, salt):
    return hashlib.pbkdf2_hmac("sha256", pw.encode(), bytes.fromhex(salt), 200_000).hex()

def signup(name, pw):
    name = name.strip().lower()
    if len(name) < 3:
        return "Username needs at least 3 characters."
    if len(pw) < 6:
        return "Password needs at least 6 characters."
    users = load_users()
    if name in users:
        return "That username is taken."
    salt = secrets.token_hex(16)
    users[name] = {"salt": salt, "hash": hash_pw(pw, salt)}
    with open(USERS_FILE, "w") as f:
        json.dump(users, f)
    return None

def check_login(name, pw):
    u = load_users().get(name.strip().lower())
    return bool(u) and secrets.compare_digest(u["hash"], hash_pw(pw, u["salt"]))

# ---------- saved sessions (scores only, never audio) ----------
def db():
    conn = sqlite3.connect(DB_FILE)
    conn.execute("""CREATE TABLE IF NOT EXISTS sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT, user TEXT, ts TEXT,
        track TEXT, sub TEXT, n INTEGER, overall TEXT, perq TEXT)""")
    return conn

def save_session(user, track, sub, overall, perq):
    conn = db()
    conn.execute("INSERT INTO sessions (user, ts, track, sub, n, overall, perq) VALUES (?,?,?,?,?,?,?)",
                 (user, datetime.now().strftime("%Y-%m-%d %H:%M"), track, sub,
                  len(perq), json.dumps(overall), json.dumps(perq)))
    conn.commit(); conn.close()

def get_sessions(user):
    conn = db()
    rows = conn.execute("SELECT ts, track, sub, n, overall, perq FROM sessions "
                        "WHERE user=? ORDER BY id DESC", (user,)).fetchall()
    conn.close()
    return [{"ts": r[0], "track": r[1], "sub": r[2], "n": r[3],
             "overall": json.loads(r[4]), "perq": json.loads(r[5])} for r in rows]

def nice_date(ts):
    return datetime.strptime(ts, "%Y-%m-%d %H:%M").strftime("%d %b %Y")

# ---------- question banks: (question, key points) ----------
ROLES = {
    "Software Engineer": [
        ("Walk me through a project you built end to end.",
         ["The problem and your role", "Tech stack and one hard decision", "Result and what you learned"]),
        ("How do you debug a problem you have never seen before?",
         ["Reproduce it and read the logs", "Narrow down the cause step by step", "Fix it and add a test"]),
        ("Tell me about a time you disagreed with a teammate.",
         ["The situation, briefly", "How you listened and explained", "How it ended and what you learned"]),
        ("How do you keep your code readable and maintainable?",
         ["Clear names and small functions", "Code review and tests", "Documentation and consistency"]),
        ("Describe something new you learned quickly.",
         ["Why you needed it", "How you learned it", "How you applied it"]),
    ],
    "Data Analyst": [
        ("Walk me through an analysis you did from start to finish.",
         ["The question you answered", "Data cleaning and method", "The insight and the decision it led to"]),
        ("How do you handle missing or messy data?",
         ["Check how much is missing and why", "Drop, fill or flag it", "Document your choice"]),
        ("How would you explain a finding to a non technical manager?",
         ["Lead with the takeaway", "Use one simple chart", "Say what action to take"]),
        ("What is the difference between a mean and a median, and when does it matter?",
         ["Define both", "Outliers and skew", "A quick example"]),
        ("Which tools do you use and why?",
         ["SQL and Excel basics", "Python or a BI tool", "Why they fit the task"]),
    ],
    "Data Scientist": [
        ("Describe a machine learning project you are proud of.",
         ["Problem and data", "Model choice and evaluation", "Result and limits"]),
        ("What is overfitting and how do you prevent it?",
         ["Define it", "Validation and regularization", "More data or a simpler model"]),
        ("How do you choose an evaluation metric?",
         ["Match it to the goal", "Accuracy versus precision and recall", "Class imbalance"]),
        ("Explain the bias and variance trade off.",
         ["What each means", "How model complexity changes them", "How you balance them"]),
        ("How do you check that your results are trustworthy?",
         ["Split by group, not at random", "Cross validation", "Compare to a simple baseline"]),
    ],
    "Campus Placement (HR)": [
        ("Tell me about yourself.",
         ["Education and strongest skills", "One project or achievement", "Why this role"]),
        ("What are your strengths and your weakness?",
         ["Two strengths with proof", "One real weakness", "What you are doing about it"]),
        ("Why should we hire you?",
         ["Skills that match the role", "Proof from a project", "Eagerness to learn"]),
        ("Where do you see yourself in five years?",
         ["Skills you want to build", "How you will contribute", "Realistic growth"]),
        ("Tell me about a time you worked under pressure.",
         ["The situation", "What you did", "The outcome"]),
    ],
}

DSA = {
    "Complexity and arrays": [
        ("Explain time complexity and what Big O means.",
         ["How cost grows with input size", "Examples: O(1), O(n), O(n log n)", "Focus on the worst case"]),
        ("How would you find two numbers in an array that sum to a target?",
         ["Brute force and its cost", "Hash map in one pass", "Time and space complexity"]),
        ("Explain the two pointer technique with an example.",
         ["Where it applies, such as sorted data", "How the pointers move", "Complexity"]),
        ("How do you reverse a string in place?",
         ["Two pointers from both ends", "Swap and move inward", "O(n) time, O(1) space"]),
    ],
    "Hashing": [
        ("How does a hash map work?",
         ["Hash function maps a key to an index", "Collisions", "Average O(1) lookup"]),
        ("How do you handle collisions?",
         ["Chaining", "Open addressing", "Load factor and resizing"]),
        ("When would you use a hash set instead of a list?",
         ["Fast membership check", "Removing duplicates", "Trade off: memory and no order"]),
        ("Find the first non repeating character in a string.",
         ["Count with a hash map", "Second pass to find it", "O(n) time"]),
    ],
    "Linked lists, stacks, queues": [
        ("Array versus linked list: when do you pick which?",
         ["Access time", "Insert and delete cost", "Memory layout"]),
        ("How do you detect a cycle in a linked list?",
         ["Fast and slow pointers", "Why they eventually meet", "O(n) time, O(1) space"]),
        ("Explain a stack and give a real use.",
         ["Last in, first out", "Push and pop", "Undo, bracket matching, recursion"]),
        ("How would you build a queue using two stacks?",
         ["An input stack and an output stack", "Move items when the output is empty", "Amortized O(1)"]),
    ],
    "Trees and graphs": [
        ("Explain BFS and DFS and when to use each.",
         ["Queue versus stack or recursion", "BFS gives shortest path when unweighted", "Complexity O(V + E)"]),
        ("What is a binary search tree and what is its search cost?",
         ["Left smaller, right larger", "O(log n) if balanced", "Worst case O(n)"]),
        ("How do you find the height of a binary tree?",
         ["Recursion", "One plus the larger child height", "O(n) time"]),
        ("Explain Dijkstra's algorithm at a high level.",
         ["Shortest paths with non negative weights", "A priority queue picks the closest node", "Relax edges, then complexity"]),
    ],
    "Sorting, searching, DP": [
        ("Explain binary search and its requirements.",
         ["Data must be sorted", "Halve the range each step", "O(log n)"]),
        ("Compare quicksort and merge sort.",
         ["Both divide and conquer", "Quicksort worst case, merge sort extra space", "Both average O(n log n)"]),
        ("What is dynamic programming? Give an example.",
         ["Overlapping subproblems", "Memoization or tabulation", "Fibonacci or knapsack"]),
        ("How would you solve the climbing stairs problem?",
         ["ways(n) = ways(n-1) + ways(n-2)", "Base cases", "O(n) time, O(1) space version"]),
    ],
}

TIPS = {
    "nervous": "Your voice reads as nervous. Slow down, pause between sentences, and breathe before you start.",
    "flat": "Your delivery reads flat. Stress the key words, vary your pace, and raise your energy a little.",
    "confident": "Your delivery reads confident and warm. Keep this pace and energy.",
    "intense": "Your delivery reads very forceful. Soften your tone and slow down a little.",
}

# ---------- ui helpers ----------
def logo(big=True):
    fs = "3.2rem" if big else "1.6rem"
    pad = "14px 56px" if big else "5px 26px"
    since = '<div class="bsince">SINCE 2026</div>' if big else ""
    st.markdown(
        f'<div class="bw"><div class="b1"><div class="b2"><div class="b3" style="padding:{pad}">'
        f'<div class="bn" style="font-size:{fs}">Sonar</div>'
        f'<div class="bs">VOICE &nbsp;·&nbsp; INTERVIEWS</div>{since}'
        f'</div></div></div></div>', unsafe_allow_html=True)

def bars(avg):
    html = ""
    for rank, j in enumerate(np.argsort(-avg)):
        color = ROUGE if rank == 0 else FLAMINGO
        html += (f'<div class="row"><div class="lab">{classes[j]}</div>'
                 f'<div class="track"><div class="fill" style="width:{avg[j]*100:.0f}%;'
                 f'background:{color}"></div></div>'
                 f'<div class="pct">{avg[j]*100:.0f}%</div></div>')
    st.markdown(html, unsafe_allow_html=True)

def dict_to_arr(d):
    return np.array([d.get(c, 0.0) for c in classes])

def heatmap(P):
    cmap = LinearSegmentedColormap.from_list("sonar", [PEACH, FLAMINGO, ROUGE, MERLOT])
    fig, ax = plt.subplots(figsize=(8, 3), facecolor=CREAM)
    ax.set_facecolor(CREAM)
    ax.imshow(P.T, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes, color=FILM)
    ax.set_xticks(range(len(P)))
    ax.set_xticklabels([f"{int(i*CHUNK)}-{int((i+1)*CHUNK)}s" for i in range(len(P))],
                       color=FILM, fontsize=8)
    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            ax.text(i, j, f"{P[i, j]*100:.0f}", ha="center", va="center", fontsize=8,
                    color=CREAM if P[i, j] > 0.5 else FILM)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    st.pyplot(fig)

def trend_chart(rows):
    chrono = rows[::-1]
    x = list(range(1, len(chrono) + 1))
    fig, ax = plt.subplots(figsize=(8, 3), facecolor=CREAM)
    ax.set_facecolor(CREAM)
    ax.plot(x, [r["overall"].get("confident", 0) * 100 for r in chrono],
            color=ROUGE, marker="o", label="confident")
    ax.plot(x, [r["overall"].get("nervous", 0) * 100 for r in chrono],
            color=CHART, marker="o", label="nervous")
    ax.set_ylim(0, 100); ax.set_xticks(x)
    ax.set_xlabel("session", color=FILM); ax.set_ylabel("percent", color=FILM)
    ax.tick_params(colors=FILM)
    ax.legend(frameon=False, labelcolor=FILM)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(FLAMINGO)
    st.pyplot(fig)

def privacy():
    with st.expander("Privacy and use"):
        st.write("Audio is analyzed on this computer and deleted right after. Only the scores are saved, "
                 "never the recording. The model was trained on acted studio speech (RAVDESS and CREMA-D) "
                 "and is less accurate on a laptop microphone. Labels are interview oriented groupings of "
                 "acted emotions, not a measure of your real feelings. Use it for practice only.")

S = st.session_state
for k, v in {"page": "auth", "user": None, "qs": [], "idx": 0, "results": {},
             "track": "", "sub": "", "sid": 0, "saved": False}.items():
    S.setdefault(k, v)

def go(page):
    S.page = page
    st.rerun()

def start_session(track, sub, n):
    bank = ROLES[sub] if track == "Role interview" else DSA[sub]
    S.qs = random.sample(bank, min(n, len(bank)))
    S.track, S.sub = track, sub
    S.idx, S.results, S.saved = 0, {}, False
    S.sid += 1
    go("practice")

def nav(active):
    items = [("Practice", "home"), ("Dashboard", "dashboard"), ("History", "history")]
    cols = st.columns(4)
    for col, (label, page) in zip(cols, items):
        with col:
            if st.button(label, key=f"nav_{page}", disabled=(active == page),
                         use_container_width=True):
                go(page)
    with cols[3]:
        if st.button("Log out", key="nav_out", use_container_width=True):
            S.user = None; go("auth")
    st.write("")

# ---------- pages ----------
def page_auth():
    logo()
    st.header("We help you master interviews.")
    st.write("Practice real questions out loud and hear how your voice comes across.")
    t1, t2, t3 = st.tabs(["Log in", "Create account", "Guest"])
    with t1:
        with st.form("login"):
            u = st.text_input("Username")
            p = st.text_input("Password", type="password")
            if st.form_submit_button("Log in"):
                if check_login(u, p):
                    S.user = u.strip().lower(); go("home")
                else:
                    st.error("Wrong username or password.")
    with t2:
        with st.form("signup"):
            u = st.text_input("Choose a username")
            p = st.text_input("Choose a password", type="password")
            if st.form_submit_button("Create account"):
                err = signup(u, p)
                if err:
                    st.error(err)
                else:
                    S.user = u.strip().lower(); go("home")
    with t3:
        st.write("Try it without an account. Guest sessions are not saved.")
        if st.button("Continue as guest"):
            S.user = "guest"; go("home")

def page_home():
    logo()
    nav("home")
    st.header(f"Welcome, {S.user.title()}.")
    st.write("Pick an interview, answer out loud, and see how your voice comes across.")

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Choose your interview")
        track = st.radio("Type", ["Role interview", "DSA interview"], horizontal=True)
        if track == "Role interview":
            sub = st.selectbox("Role", list(ROLES))
            st.markdown('<div class="note">Role and behavioural questions. Speak as you would in the room.</div>',
                        unsafe_allow_html=True)
        else:
            sub = st.selectbox("Topic", list(DSA))
            st.markdown('<div class="note">Explain each concept out loud. Sonar scores your delivery, '
                        'not whether the answer is correct.</div>', unsafe_allow_html=True)
        n = st.select_slider("Number of questions", options=[3, 4, 5], value=3)
        if st.button("Start interview"):
            start_session(track, sub, n)
    with right:
        st.markdown(
            '<div class="panel"><div class="ptitle">How it works</div>'
            '<div class="step">1. Choose a role or a DSA topic.</div>'
            '<div class="step">2. Record your answer to each question.</div>'
            '<div class="step">3. See how you sounded and what to cover.</div>'
            '<div class="ptitle" style="margin-top:16px">Recording tips</div>'
            '<div class="step">Use a quiet room.</div>'
            '<div class="step">Answer for 15 to 30 seconds.</div>'
            '<div class="step">Sit about an arm from the mic.</div>'
            '</div>', unsafe_allow_html=True)
    st.write("")
    privacy()

def page_dashboard():
    logo(False)
    nav("dashboard")
    st.header("Your progress")
    if S.user == "guest":
        st.write("Guest sessions are not saved. Create an account to track your progress.")
        return
    rows = get_sessions(S.user)
    if not rows:
        st.write("No sessions yet. Finish one interview and your progress appears here.")
        if st.button("Start your first interview"):
            go("home")
        return

    mean_all = np.mean([dict_to_arr(r["overall"]) for r in rows], axis=0)
    ci, ni = list(classes).index("confident"), list(classes).index("nervous")
    st.markdown(
        '<div class="panel">'
        f'<div class="srow"><span>Sessions completed</span><b>{len(rows)}</b></div>'
        f'<div class="srow"><span>Questions answered</span><b>{sum(r["n"] for r in rows)}</b></div>'
        f'<div class="srow"><span>Most common delivery</span><b>{classes[int(mean_all.argmax())]}</b></div>'
        f'<div class="srow"><span>Average confident</span><b>{mean_all[ci]*100:.0f}%</b></div>'
        f'<div class="srow"><span>Average nervous</span><b>{mean_all[ni]*100:.0f}%</b></div>'
        '</div>', unsafe_allow_html=True)

    st.subheader("Trend")
    if len(rows) < 2:
        st.markdown('<div class="note">Finish one more session to see your trend.</div>',
                    unsafe_allow_html=True)
    else:
        trend_chart(rows)
        last = rows[0]["overall"].get("nervous", 0) * 100
        before = np.mean([r["overall"].get("nervous", 0) for r in rows[1:]]) * 100
        diff = last - before
        word = "down" if diff < 0 else "up"
        st.write(f"Nervous in your last session: {last:.0f}%, {word} {abs(diff):.0f} points "
                 f"from your earlier average of {before:.0f}%.")
        st.markdown('<div class="note">Model estimates on a few sessions. Treat small changes as noise.</div>',
                    unsafe_allow_html=True)

    st.subheader("Recent sessions")
    for r in rows[:5]:
        top = max(r["overall"], key=r["overall"].get)
        st.markdown(f'<div class="step"><b>{nice_date(r["ts"])}</b> &nbsp;·&nbsp; {r["sub"]} '
                    f'<span class="note">({r["track"]}, mostly {top})</span></div>',
                    unsafe_allow_html=True)

def page_history():
    logo(False)
    nav("history")
    st.header("History")
    if S.user == "guest":
        st.write("Guest sessions are not saved. Create an account to keep a history.")
        return
    rows = get_sessions(S.user)
    if not rows:
        st.write("No sessions yet.")
        return
    for r in rows:
        with st.expander(f'{nice_date(r["ts"])}  ·  {r["sub"]}  ·  {r["track"]}'):
            bars(dict_to_arr(r["overall"]))
            for i, pq in enumerate(r["perq"], 1):
                top = max(pq["scores"], key=pq["scores"].get)
                st.markdown(f'<div class="step"><b>{i}.</b> {pq["q"]}<br>'
                            f'<span class="note">Mostly {top} ({pq["scores"][top]*100:.0f}%)</span></div>',
                            unsafe_allow_html=True)

def page_practice():
    i, total = S.idx, len(S.qs)
    q, points = S.qs[i]
    logo(False)
    st.markdown(f'<div class="note">{S.sub} &nbsp;·&nbsp; Question {i+1} of {total}</div>',
                unsafe_allow_html=True)
    st.progress(i / total)
    st.markdown(f'<div class="q">{q}</div>', unsafe_allow_html=True)
    if S.track == "DSA interview":
        st.markdown('<div class="note">Explain it as you would to an interviewer. '
                    'Sonar scores delivery, not correctness.</div>', unsafe_allow_html=True)

    if i not in S.results:
        up = st.audio_input("Record your answer, 15 to 30 seconds", key=f"rec_{S.sid}_{i}")
        if st.button("Analyze my answer", disabled=up is None):
            with st.spinner("Analyzing your answer"):
                P = analyze(up.getvalue())
            if P is None:
                st.warning("Clip too short. Record at least 2 seconds.")
            else:
                S.results[i] = P.tolist(); st.rerun()
    else:
        P = np.array(S.results[i])
        avg = P.mean(axis=0)
        st.subheader("How you sounded")
        bars(avg)
        st.subheader("Over time")
        st.markdown('<div class="note">Each column is 3 seconds. Darker means stronger. Numbers are percent.</div>',
                    unsafe_allow_html=True)
        heatmap(P)
        st.subheader("Coach note")
        st.write(TIPS[classes[avg.argmax()]])
        st.subheader("Key points to cover")
        st.markdown('<div class="note">Check which of these you mentioned.</div>', unsafe_allow_html=True)
        st.markdown("\n".join(f"- {p}" for p in points))
        st.write("")
        if st.button("Next question" if i + 1 < total else "See my results"):
            if i + 1 < total:
                S.idx += 1; st.rerun()
            else:
                go("summary")

def page_summary():
    logo(False)
    st.header("Session complete.")
    st.markdown(f'<div class="note">{S.track} &nbsp;·&nbsp; {S.sub}</div>', unsafe_allow_html=True)
    done = [i for i in range(len(S.qs)) if i in S.results]
    if not done:
        st.write("No answers were analyzed.")
    else:
        avgs = [np.array(S.results[i]).mean(axis=0) for i in done]
        overall = np.mean(avgs, axis=0)

        if S.user != "guest" and not S.saved:
            save_session(S.user, S.track, S.sub,
                         {c: float(v) for c, v in zip(classes, overall)},
                         [{"q": S.qs[i][0], "scores": {c: float(v) for c, v in zip(classes, a)}}
                          for i, a in zip(done, avgs)])
            S.saved = True

        st.subheader("Overall")
        bars(overall)
        st.subheader("By question")
        for i, a in zip(done, avgs):
            j = int(a.argmax())
            st.markdown(f'<div class="step"><b>{i+1}.</b> {S.qs[i][0]}<br>'
                        f'<span class="note">Mostly {classes[j]} ({a[j]*100:.0f}%)</span></div>',
                        unsafe_allow_html=True)
        st.subheader("Coach notes")
        st.write(TIPS[classes[overall.argmax()]])
        nj = list(classes).index("nervous")
        tense = int(np.argmax([a[nj] for a in avgs]))
        if avgs[tense][nj] > 0.15:
            st.write(f"You sounded most nervous on question {done[tense]+1}. Practice that one again.")
        st.markdown('<div class="note">These are model estimates for practice, not a judgment of you.</div>',
                    unsafe_allow_html=True)
        if S.user == "guest":
            st.markdown('<div class="note">Guest sessions are not saved.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="note">Saved to your dashboard.</div>', unsafe_allow_html=True)
    st.write("")
    if st.button("Practice again"):
        start_session(S.track, S.sub, len(S.qs))
    if st.button("Back to home"):
        go("home")
    privacy()

if S.user is None and S.page != "auth":
    S.page = "auth"
{"auth": page_auth, "home": page_home, "practice": page_practice,
 "summary": page_summary, "dashboard": page_dashboard, "history": page_history}[S.page]()
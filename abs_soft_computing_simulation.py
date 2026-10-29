"""
ABS Soft Computing Case Study - Interactive Streamlit App
Educational simulation only. Not for real vehicle control.
The model is simplified and is not an exact reproduction of the cited paper.
"""
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="ABS Soft Computing Simulator", page_icon="🚗", layout="wide")

# -------------------- Model functions --------------------
def triangular(x, a, b, c):
    if x <= a or x >= c:
        return 0.0
    if x == b:
        return 1.0
    return (x-a)/(b-a) if x < b else (c-x)/(c-b)

def fuzzy_mu_estimate(slip, speed, road_cue):
    """Illustrative T-S-style fuzzy estimate; road_cue is simulated input."""
    low = triangular(slip, -0.1, 0.0, 0.2)
    med = triangular(slip, 0.05, 0.2, 0.4)
    high = triangular(slip, 0.2, 0.6, 1.05)
    slow = triangular(speed, -5.0, 0.0, 12.0)
    fast = triangular(speed, 5.0, 25.0, 45.0)
    weights = np.array([low*slow, low*fast, med*slow, med*fast, high*slow, high*fast])
    factors = np.array([0.95, 0.98, 1.0, 1.02, 0.88, 0.90])
    if weights.sum() < 1e-9:
        return float(np.clip(road_cue, 0.15, 1.0))
    return float(np.clip(np.dot(weights, road_cue*factors)/weights.sum(), 0.15, 1.0))

@st.cache_resource
def train_ann():
    rng = np.random.default_rng(42)
    s = rng.uniform(-1.0, 1.0, 5000).reshape(-1, 1)
    y = (-1.2*np.sign(s) - 2.0*np.abs(s)**0.6*np.sign(s)
         - np.abs(s)**1.4*np.sign(s) - 3.0*s).ravel()
    model = make_pipeline(
        StandardScaler(),
        MLPRegressor(hidden_layer_sizes=(24, 12), activation="tanh",
                     max_iter=1000, early_stopping=True, random_state=42)
    )
    model.fit(s, y)
    return model

def pso_gain():
    """Small deterministic PSO demo; surrogate objective, not paper's exact objective."""
    rng = np.random.default_rng(7)
    n, lo, hi = 20, 0.2, 6.0
    x = rng.uniform(lo, hi, n)
    vel = np.zeros(n)
    def cost(g):
        t = np.linspace(0, 1.5, 200)
        e = 0.12*np.exp(-(1+g)*t)
        return np.trapezoid(np.abs(e), t) + 0.003*g*g
    pb, pc = x.copy(), np.array([cost(z) for z in x])
    idx = int(np.argmin(pc)); gb, gc = pb[idx], pc[idx]
    for _ in range(50):
        vel = 0.68*vel + 1.4*rng.random(n)*(pb-x) + 1.4*rng.random(n)*(gb-x)
        x = np.clip(x+vel, lo, hi)
        c = np.array([cost(z) for z in x])
        better = c < pc
        pb[better], pc[better] = x[better], c[better]
        idx = int(np.argmin(pc))
        if pc[idx] < gc:
            gb, gc = pb[idx], pc[idx]
    return float(gb)

def run_sim(initial_speed, desired_slip, dry_mu, wet_mu, transition_time,
            duration, max_torque, controller, dt=0.002):
    # Quarter-car simplified model
    mass, wheel_inertia, radius, gravity = 1500.0, 1.2, 0.31, 9.81
    quarter_mass = mass / 4.0
    normal_load = quarter_mass * gravity
    n = int(duration/dt) + 1
    t = np.arange(n)*dt
    v = np.zeros(n); omega = np.zeros(n); slip = np.full(n, np.nan)
    mu_true = np.zeros(n); mu_hat = np.zeros(n); torque = np.zeros(n)
    err = np.full(n, np.nan)
    v[0] = initial_speed; omega[0] = initial_speed/radius
    ann = train_ann()
    gain = pso_gain()
    integral = 0.0
    prev_e = 0.0
    moving = np.ones(n, dtype=bool)
    stop_index = n-1
    # Controller gains chosen for this educational model.
    kp, ki, kd = 2400.0, 450.0, 0.0
    for i in range(n-1):
        if v[i] <= 0.5:
            v[i:] = 0.0
            omega[i:] = 0.0
            torque[i:] = 0.0
            moving[i:] = False
            stop_index = i
            break
        road = dry_mu if t[i] < transition_time else wet_mu
        mu_true[i] = road
        # Slip is evaluated only while vehicle is moving.
        s = np.clip((v[i]-radius*omega[i])/max(v[i], 0.5), 0.0, 1.0)
        slip[i] = s
        e = desired_slip-s
        err[i] = e
        road_cue = np.clip(road + 0.008*np.sin(13*t[i]) + 0.004*np.sin(31*t[i]), 0.15, 1.0)
        mu_hat[i] = fuzzy_mu_estimate(s, v[i], road_cue)
        # Smooth illustrative tire-force curve peaking at desired slip.
        shape = (s/desired_slip)*np.exp(1.0-s/desired_slip) if s > 1e-8 else 0.0
        tire_mu = road*np.clip(shape, 0.0, 1.0)
        fx = tire_mu*normal_load
        e_dot = (e-prev_e)/dt
        prev_e = e
        integral = np.clip(integral + e*dt, -0.4, 0.4)
        # Friction feed-forward + feedback. The "soft" controller adds
        # ANN reaching-law and PSO-tuned sliding corrections.
        base = radius*mu_hat[i]*normal_load*0.72
        feedback = kp*e + ki*integral + kd*e_dot
        if controller == "Soft-computing (Fuzzy + ANN + PSO)":
            sliding = s-desired_slip
            ann_correction = float(ann.predict(np.array([[sliding]]))[0])
            pso_correction = -gain*np.tanh(sliding/0.04)
            command = base + feedback + 18.0*(ann_correction+pso_correction)
        else:
            command = base + feedback
        torque[i] = np.clip(command, 0.0, max_torque)
        # Forward dynamics
        v_next = max(0.0, v[i] - fx/quarter_mass*dt)
        omega_next = max(0.0, omega[i] + (radius*fx-torque[i])/wheel_inertia*dt)
        # Do not allow reverse wheel rotation.
        omega[i+1] = max(0.0, omega_next)
        v[i+1] = v_next
    # Final sample handling: no fake slip/error once stopped.
    for i in range(n):
        if v[i] > 0.5:
            road = dry_mu if t[i] < transition_time else wet_mu
            mu_true[i] = road
            s = np.clip((v[i]-radius*omega[i])/max(v[i], 0.5), 0.0, 1.0)
            slip[i] = s
            err[i] = desired_slip-s
            cue = road + 0.008*np.sin(13*t[i]) + 0.004*np.sin(31*t[i])
            mu_hat[i] = fuzzy_mu_estimate(s, v[i], np.clip(cue, 0.15, 1.0))
        else:
            moving[i] = False
            slip[i] = np.nan
            err[i] = np.nan
            mu_true[i] = dry_mu if t[i] < transition_time else wet_mu
            mu_hat[i] = np.nan
    # Metrics only use moving samples, avoiding misleading stop-state slip.
    valid = moving & np.isfinite(err)
    if valid.sum() == 0:
        valid = np.isfinite(err)
    dist = float(np.trapezoid(v, t))
    stop_time = float(t[np.where(v <= 0.5)[0][0]]) if np.any(v <= 0.5) else float("nan")
    metrics = {
        "Initial speed (m/s)": float(v[0]),
        "Final speed (m/s)": float(v[-1]),
        "Braking distance (m)": dist,
        "Stopping time (s)": stop_time,
        "Mean absolute slip error while moving": float(np.mean(np.abs(err[valid]))),
        "Maximum absolute slip error while moving": float(np.max(np.abs(err[valid]))),
        "Peak braking torque (Nm)": float(np.max(torque)),
        "RMS braking torque (Nm)": float(np.sqrt(np.mean(torque**2))),
        "Moving samples used for slip metrics": int(valid.sum())
    }
    return {"time":t,"vehicle_speed":v,"wheel_speed":omega,"slip":slip,
            "mu_true":mu_true,"mu_hat":mu_hat,"torque":torque,"error":err,
            "metrics":metrics,"gain":gain}

def plot_result(result, desired_slip, transition_time):
    t=result["time"]
    fig, ax=plt.subplots(3,2,figsize=(13,9))
    ax[0,0].plot(t,result["slip"],label="Actual slip")
    ax[0,0].axhline(desired_slip,ls="--",label="Desired slip")
    ax[0,0].axvline(transition_time,ls=":",label="Road transition")
    ax[0,0].set_title("Wheel-slip tracking"); ax[0,0].set_ylabel("Slip ratio"); ax[0,0].legend()
    ax[0,1].plot(t,result["torque"]); ax[0,1].axvline(transition_time,ls=":")
    ax[0,1].set_title("Braking torque"); ax[0,1].set_ylabel("Torque (Nm)")
    ax[1,0].plot(t,result["vehicle_speed"]); ax[1,0].set_title("Vehicle speed"); ax[1,0].set_ylabel("m/s")
    ax[1,1].plot(t,result["wheel_speed"]); ax[1,1].set_title("Wheel angular speed"); ax[1,1].set_ylabel("rad/s")
    ax[2,0].plot(t,result["error"]); ax[2,0].axhline(0,ls="--"); ax[2,0].axvline(transition_time,ls=":")
    ax[2,0].set_title("Slip tracking error (moving only)"); ax[2,0].set_ylabel("Desired - actual")
    ax[2,1].plot(t,result["mu_true"],label="Simulated road friction")
    ax[2,1].plot(t,result["mu_hat"],label="Fuzzy estimate"); ax[2,1].axvline(transition_time,ls=":")
    ax[2,1].set_title("Road-friction estimation"); ax[2,1].set_ylabel("Coefficient"); ax[2,1].legend()
    for a in ax.flat:
        a.set_xlabel("Time (s)"); a.grid(True,alpha=0.3)
    fig.tight_layout()
    return fig

# -------------------- UI --------------------
st.title("🚗 Anti-Lock Braking System (ABS)")
st.subheader("Soft Computing Case Study Simulator")
st.write("Explore how a simplified ABS model responds to braking, target slip, and a road-friction change.")

with st.sidebar:
    st.header("Simulation inputs")
    initial_speed = st.slider("Initial vehicle speed (km/h)", 30, 140, 97, 1)
    desired_slip = st.slider("Desired wheel slip", 0.10, 0.30, 0.20, 0.01)
    dry_mu = st.slider("Initial road friction coefficient", 0.30, 1.00, 0.85, 0.05)
    wet_mu = st.slider("Friction after road change", 0.15, 0.80, 0.40, 0.05)
    duration = st.slider("Simulation duration (seconds)", 2.0, 8.0, 5.0, 0.5)
    transition_time = st.slider("Road transition time (seconds)", 0.5, float(duration), min(2.5, float(duration)), 0.5)
    max_torque = st.slider("Maximum braking torque (Nm)", 500, 3000, 2500, 100)
    controller = st.selectbox("Controller", [
        "Soft-computing (Fuzzy + ANN + PSO)",
        "Baseline (friction feed-forward + PI)"
    ])
    run = st.button("Run simulation", type="primary")

st.caption("Model note: this is an educational simulation. The friction cue and fuzzy rules are illustrative; the model is not a validated vehicle controller or an exact reproduction of the reference paper.")

if run:
    with st.spinner("Training ANN and running the simulation..."):
        result = run_sim(
            initial_speed=initial_speed/3.6,
            desired_slip=desired_slip,
            dry_mu=dry_mu,
            wet_mu=wet_mu,
            transition_time=transition_time,
            duration=duration,
            max_torque=max_torque,
            controller=controller
        )
    st.success("Simulation completed.")
    st.metric("PSO-selected gain", f'{result["gain"]:.3f}')
    st.subheader("Performance metrics")
    cols = st.columns(4)
    items = list(result["metrics"].items())
    for i, (name, value) in enumerate(items):
        with cols[i % 4]:
            if isinstance(value, (int, np.integer)):
                st.metric(name, f"{value}")
            else:
                st.metric(name, f"{value:.4f}")
    st.subheader("Simulation graphs")
    fig = plot_result(result, desired_slip, transition_time)
    st.pyplot(fig)
    fig.savefig("abs_soft_computing_results.png", dpi=160, bbox_inches="tight")
    st.download_button(
        "Download results graph (PNG)",
        data=open("abs_soft_computing_results.png", "rb").read(),
        file_name="abs_soft_computing_results.png",
        mime="image/png"
    )
else:
    st.info("Choose your input values in the sidebar and click **Run simulation**.")
    st.markdown("""
    **Included techniques**
    - T-S-style fuzzy friction estimation
    - ANN approximation of a nonlinear reaching law
    - PSO gain search
    - Slip feedback controller and baseline comparison

    **Important:** The included PSO objective is a simplified surrogate, and the ANN learns a generated mathematical function. Use this as a transparent educational demonstration, not as a claim of exact paper reproduction.
    """)

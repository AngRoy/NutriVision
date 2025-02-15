import os
import time
import json
import datetime
import sqlite3
import streamlit as st
import torch
import torch.nn as nn
import plotly.express as px
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, BlipProcessor, BlipForConditionalGeneration

# =============================
# PAGE CONFIG for WIDE / MOBILE
# =============================
st.set_page_config(page_title="NutriVision", layout="wide")

# =============================
# STARFIELD BACKGROUND & CSS THEME
# =============================
STARFIELD_CSS = """
<style>
#starfield {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: url('https://i.imgur.com/9z4xUO7.png') repeat;
    animation: moveStars 120s linear infinite;
    z-index: -2;
    opacity: 0.3;
}
@keyframes moveStars {
    from { background-position: 0 0; }
    to { background-position: -10000px 5000px; }
}
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #000000, #121212) !important;
}
header {
    background-color: #1F1F1F !important;
}
[data-testid="stSidebar"] {
    background-color: #1F1F1F !important;
}
body, .stMarkdown, .stMetric, .css-1aumxhk {
    color: #E8EAF6 !important;
}
.stTabs > div[role="tablist"] {
    background-color: #263238 !important;
    border-bottom: 2px solid #000000 !important;
}
.stTabs div[role="tab"] {
    font-size: 18px;
    font-weight: bold;
    color: #E8EAF6 !important;
    padding: 10px !important;
    transition: background-color 0.3s ease !important;
}
.stTabs div[role="tab"]:hover {
    background-color: #37474F !important;
}
.stButton>button {
    background-color: #E53935 !important;
    color: #FFFFFF !important;
    font-weight: bold !important;
    border: none !important;
    padding: 10px 20px !important;
    border-radius: 5px !important;
    transition: background-color 0.3s ease !important;
}
.stButton>button:hover {
    background-color: #EF5350 !important;
}
.stNumberInput input, .stTextInput input, .stSelectbox select {
    background-color: #263238 !important;
    color: #E8EAF6 !important;
    border: 1px solid #37474F !important;
}
.stFileUploader {
    background-color: #263238 !important;
    color: #E8EAF6 !important;
    border: 1px solid #37474F !important;
}
</style>
<div id="starfield"></div>
"""

st.markdown(STARFIELD_CSS, unsafe_allow_html=True)

# =============================
# SHOW LOADING SCREEN ONCE
# =============================
if "has_loaded" not in st.session_state:
    spinner_container = st.empty()
    spinner_container.markdown("""
    <div class="loader"></div>
    <style>
    .loader {
      border: 16px solid #444;
      border-top: 16px solid #E53935;
      border-radius: 50%;
      width: 120px;
      height: 120px;
      animation: spin 2s linear infinite;
      position: fixed;
      top: 50%;
      left: 50%;
      transform: translate(-50%, -50%);
      z-index: 9999;
    }
    @keyframes spin {
      0% { transform: translate(-50%, -50%) rotate(0deg); }
      100% { transform: translate(-50%, -50%) rotate(360deg); }
    }
    </style>
    """, unsafe_allow_html=True)
    time.sleep(2)
    spinner_container.empty()
    st.session_state.has_loaded = True

# =============================
# DB INITIALIZATION
# =============================
@st.cache_resource
def init_db():
    conn = sqlite3.connect("nutrivision.db", check_same_thread=False)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT,
            height REAL,
            weight REAL,
            age INTEGER,
            gender TEXT,
            profile_pic TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS meals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            meal_time TIMESTAMP,
            source TEXT,
            caption TEXT,
            predicted TEXT,
            calories REAL,
            meal_image TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)
    conn.commit()
    return conn

conn = init_db()

# =============================
# DB HELPER FUNCTIONS
# =============================
def register_user(username, password, height, weight, age, gender, pic):
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (username, password, height, weight, age, gender, profile_pic) VALUES (?,?,?,?,?,?,?)",
                  (username, password, height, weight, age, gender, pic))
        conn.commit()
        c.execute("SELECT id FROM users WHERE username=?", (username,))
        uid = c.fetchone()[0]
        return True, "Registration successful!", uid
    except sqlite3.IntegrityError:
        return False, "Username already exists.", None

def login_user(username, password):
    c = conn.cursor()
    c.execute("SELECT id FROM users WHERE username=? AND password=?", (username, password))
    row = c.fetchone()
    return row[0] if row else None

def store_meal(uid, source, caption, pred_list, cal, img_path):
    c = conn.cursor()
    c.execute("INSERT INTO meals (user_id, meal_time, source, caption, predicted, calories, meal_image) VALUES (?,?,?,?,?,?,?)",
              (uid, datetime.datetime.now(), source, caption, json.dumps(pred_list), cal, img_path))
    conn.commit()

def get_meals(uid):
    c = conn.cursor()
    c.execute("SELECT meal_time, source, caption, predicted, calories, meal_image FROM meals WHERE user_id=? ORDER BY meal_time DESC", (uid,))
    return c.fetchall()

def get_daily_cals(uid, date):
    c = conn.cursor()
    start = datetime.datetime.combine(date, datetime.time.min)
    end = datetime.datetime.combine(date, datetime.time.max)
    c.execute("SELECT SUM(calories) FROM meals WHERE user_id=? AND meal_time BETWEEN ? AND ?", (uid, start, end))
    val = c.fetchone()[0]
    return val if val else 0

def get_all_daily_cals(uid):
    c = conn.cursor()
    c.execute("SELECT DATE(meal_time), SUM(calories) FROM meals WHERE user_id=? GROUP BY DATE(meal_time) ORDER BY DATE(meal_time)", (uid,))
    return c.fetchall()

def save_file(up, folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    path = os.path.join(folder, up.name)
    with open(path, "wb") as f:
        f.write(up.getbuffer())
    return path

# =============================
# MODEL
# =============================
class NutriVisionNetMultiHead(nn.Module):
    def __init__(self, food_dim=3, fv_dim=9, fast_dim=8, device="cuda", fine_tune_clip=False):
        super().__init__()
        self.device = device
        from transformers import CLIPProcessor, CLIPModel, BlipProcessor, BlipForConditionalGeneration

        self.clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_model= CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        for p in self.clip_model.text_model.parameters():
            p.requires_grad=False
        if not fine_tune_clip:
            for p in self.clip_model.vision_model.parameters():
                p.requires_grad=False

        self.blip_proc = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.blip_model=BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        for p in self.blip_model.parameters():
            p.requires_grad=False

        fusion_dim=1024
        hidden=512
        self.food_head=nn.Sequential(
            nn.Linear(fusion_dim, hidden), nn.ReLU(), nn.Dropout(0.2), nn.Linear(hidden, food_dim)
        )
        self.fv_head=nn.Sequential(
            nn.Linear(fusion_dim, hidden), nn.ReLU(), nn.Dropout(0.2), nn.Linear(hidden, fv_dim)
        )
        self.fast_head=nn.Sequential(
            nn.Linear(fusion_dim, hidden), nn.ReLU(), nn.Dropout(0.2), nn.Linear(hidden, fast_dim)
        )

    def forward(self, img, source):
        # BLIP caption
        p = self.blip_proc(images=img, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out_ids = self.blip_model.generate(**p, max_length=50, num_beams=5)
        caption = self.blip_proc.decode(out_ids[0], skip_special_tokens=True)

        # CLIP embeddings
        clip_input = self.clip_proc(images=img, return_tensors="pt").to(self.device)
        img_emb = self.clip_model.get_image_features(**clip_input)
        clip_text = self.clip_proc(text=[caption], return_tensors="pt").to(self.device)
        txt_emb = self.clip_model.get_text_features(**clip_text)

        # Normalize
        img_emb = img_emb / img_emb.norm(p=2, dim=-1, keepdim=True)
        txt_emb = txt_emb / txt_emb.norm(p=2, dim=-1, keepdim=True)
        fused = torch.cat([img_emb, txt_emb], dim=-1)

        # Switch on source
        if source=="food_nutrition":
            out = self.food_head(fused)
        elif source=="fv":
            out = self.fv_head(fused)
        elif source=="fastfood":
            out = self.fast_head(fused)
        else:
            raise ValueError("Unknown source")
        return out.squeeze(0).cpu().numpy().tolist(), caption

@st.cache_resource
def load_model():
    dev="cuda" if torch.cuda.is_available() else "cpu"
    net= NutriVisionNetMultiHead(device=dev, fine_tune_clip=False).to(dev)
    ckpt="nutrivision_multihand.pt"
    if os.path.exists(ckpt):
        sd=torch.load(ckpt, map_location=dev)
        net.load_state_dict(sd)
    net.eval()
    return net, dev

def do_infer(image, source, net, dev):
    with torch.no_grad():
        arr, caption = net(image, source)
    cals= arr[0] if len(arr)>0 else 0
    return arr, caption, cals

# =============================
# SESSION STATE
# =============================
if "logged_in" not in st.session_state:  st.session_state.logged_in=False
if "user_id" not in st.session_state:    st.session_state.user_id=None
if "username" not in st.session_state:   st.session_state.username=""
if "user_info" not in st.session_state:  st.session_state.user_info={}
if "preferred_diet" not in st.session_state: st.session_state.preferred_diet="Not specified"

# =============================
# FORMS
# =============================
def show_login_form():
    st.subheader("Login")
    user= st.text_input("Username", key="login_user")
    pw=   st.text_input("Password", type="password", key="login_pw")
    if st.button("Log In"):
        uid= login_user(user, pw)
        if uid:
            st.session_state.logged_in=True
            st.session_state.user_id=uid
            st.session_state.username=user
            c=conn.cursor()
            c.execute("SELECT height, weight, age, gender, profile_pic FROM users WHERE id=?", (uid,))
            row=c.fetchone()
            st.session_state.user_info={
                "height": row[0],
                "weight": row[1],
                "age": row[2],
                "gender": row[3],
                "profile_pic": row[4]
            }
            st.success("You are now logged in!")
        else:
            st.error("Invalid credentials.")

def show_register_form():
    st.subheader("Register")
    reg_user= st.text_input("Username", key="reg_user")
    reg_pass= st.text_input("Password", type="password", key="reg_pass")
    h= st.number_input("Height (cm)", 0.0, 300.0, step=0.1)
    w= st.number_input("Weight (kg)", 0.0, 300.0, step=0.1)
    a= st.number_input("Age", 0, 120, step=1)
    g= st.selectbox("Gender", ["","Male","Female","Other"])
    pdiet= st.text_input("Preferred Diet (optional)")
    pic= st.file_uploader("Profile Pic (optional)", type=["jpg","jpeg","png"])

    if st.button("Register"):
        picpath=""
        if pic:
            picpath= save_file(pic,"profile_pics")
        if reg_user=="" or reg_pass=="":
            st.error("Username & password required!")
        else:
            success, msg, uid= register_user(
                reg_user, reg_pass,
                h if h>0 else None,
                w if w>0 else None,
                a if a>0 else None,
                g if g else None,
                picpath
            )
            if success:
                st.success(msg)
                st.session_state.logged_in=True
                st.session_state.user_id=uid
                st.session_state.username=reg_user
                st.session_state.user_info={
                    "height": h if h>0 else None,
                    "weight": w if w>0 else None,
                    "age": a if a>0 else None,
                    "gender": g if g else None,
                    "profile_pic": picpath
                }
                st.session_state.preferred_diet= pdiet if pdiet else "Not specified"
            else:
                st.error(msg)

# If not logged in, show a minimal login/register approach
if not st.session_state.logged_in:
    st.write("### Welcome to NutriVision\nPlease Login or Register below:")
    colA, colB= st.columns(2)
    with colA:
        st.write("**Login**")
        show_login_form()
    with colB:
        st.write("**Register**")
        show_register_form()
    st.stop()

# =============================
# MAIN TABS
# =============================
tabs= st.tabs(["Home","Upload Meal","Meal History","Account","Logout"])

# HOME TAB
with tabs[0]:
    st.header("Dashboard")
    st.write(f"Hello, {st.session_state.username}!")
    # Show daily cals
    today= datetime.date.today()
    cals_today= get_daily_cals(st.session_state.user_id, today)
    st.metric("Today's Calorie Intake", f"{cals_today:.1f} kcal")

    # Manual Refresh
    if st.button("Refresh Dashboard"):
        st.experimental_rerun()

    # Show daily chart
    allcals= get_all_daily_cals(st.session_state.user_id)
    if allcals:
        import pandas as pd
        df= pd.DataFrame(allcals, columns=["Date","Calories"])
        fig= px.line(df, x="Date", y="Calories", title="Daily Calorie Intake", markers=True)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("No meal records yet.")

# UPLOAD TAB
with tabs[1]:
    st.header("Upload a Meal")
    st.write("Use the AI model to analyze your meal.")
    cat_map= {"Food Nutrition":"food_nutrition","Fruits & Vegetables":"fv","Fast Food":"fastfood"}
    sel_cat= st.selectbox("Meal Category", list(cat_map.keys()))
    source= cat_map[sel_cat]

    upmeal= st.file_uploader("Choose a meal image", type=["jpg","jpeg","png"])
    if upmeal:
        try:
            meal_img= Image.open(upmeal).convert("RGB")
            st.image(meal_img, caption="Meal Image", use_container_width=True)
        except:
            st.error("Could not open image.")
        if st.button("Analyze Meal"):
            with st.spinner("Analyzing with AI..."):
                model, dev= load_model()
                preds, caption, cals= do_infer(meal_img, source, model, dev)
            st.success("Analysis complete!")
            st.write("**Caption**:", caption)
            if source=="food_nutrition":
                columns=["Caloric Value","Fat","Carbohydrates"]
            elif source=="fv":
                columns=["energy (kcal/kJ)","water (g)","protein (g)","total fat (g)","carbohydrates (g)","fiber (g)","sugars (g)","calcium (mg)","iron (mg)"]
            else:
                columns=["calories","cal_fat","total_fat","sat_fat","trans_fat","cholesterol","sodium","total_carb"]
            st.table({
                "Nutrient": columns,
                "Value": [round(x,2) for x in preds]
            })
            st.write("**Predicted Calories**:", f"{cals:.1f} kcal")

            path= save_file(upmeal, "meal_images")
            store_meal(st.session_state.user_id, source, caption, preds, cals, path)
            st.success("Meal saved to history!")
            st.info("Go to 'Meal History' tab or refresh the dashboard to see updates.")

# HISTORY TAB
with tabs[2]:
    st.header("Meal History")
    hist= get_meals(st.session_state.user_id)
    if hist:
        for me in hist:
            tme, src, cap, pred_str, cals, imgpth= me
            st.write(f"**Time**: {tme}, **Category**: {src}")
            if imgpth and os.path.exists(imgpth):
                st.image(imgpth, use_container_width=True)
            st.write(f"**Caption**: {cap}")
            st.write(f"**Calories**: {cals:.1f}")
            try:
                arr= json.loads(pred_str)
                if src=="food_nutrition":
                    cCols=["Caloric Value","Fat","Carbohydrates"]
                elif src=="fv":
                    cCols=["energy (kcal/kJ)","water (g)","protein (g)","total fat (g)","carbohydrates (g)","fiber (g)","sugars (g)","calcium (mg)","iron (mg)"]
                else:
                    cCols=["calories","cal_fat","total_fat","sat_fat","trans_fat","cholesterol","sodium","total_carb"]
                st.table({
                    "Nutrient": cCols,
                    "Value": [round(v,2) for v in arr]
                })
            except:
                st.write("Raw predictions:", pred_str)
            st.markdown("---")
    else:
        st.write("No meals yet.")

    daily= get_all_daily_cals(st.session_state.user_id)
    if daily:
        import pandas as pd
        df= pd.DataFrame(daily, columns=["Date","Calories"])
        fig= px.bar(df, x="Date", y="Calories", title="Daily Calorie Intake")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("No daily data to show.")

# ACCOUNT TAB
with tabs[3]:
    st.header("Account Information")
    st.write(f"**Username**: {st.session_state.username}")
    info= st.session_state.user_info
    if info:
        ht= info.get("height", None)
        wt= info.get("weight", None)
        ag= info.get("age", None)
        gd= info.get("gender", None)
        st.write(f"**Height**: {ht if ht else 'N/A'} cm")
        st.write(f"**Weight**: {wt if wt else 'N/A'} kg")
        st.write(f"**Age**: {ag if ag else 'N/A'}")
        st.write(f"**Gender**: {gd if gd else 'N/A'}")
        if ht and wt and ht>0:
            bmi = wt/((ht/100)**2)
            st.write(f"**BMI**: {bmi:.2f}")
        st.write(f"**Preferred Diet**: {st.session_state.preferred_diet}")
        pic= info.get("profile_pic","")
        if pic and os.path.exists(pic):
            st.image(pic, use_container_width=True)
        else:
            st.write("No profile picture.")
    else:
        st.write("No user info found.")
    
    st.markdown("---")
    st.subheader("Update Profile")
    new_ht= st.number_input("Height (cm)",0.0,300.0, step=0.1, value=float(info.get('height') or 0))
    new_wt= st.number_input("Weight (kg)",0.0,300.0, step=0.1, value=float(info.get('weight') or 0))
    new_ag= st.number_input("Age", 0,120, step=1, value=int(info.get('age') or 0))
    new_gd= st.selectbox("Gender", ["","Male","Female","Other"], index=0 if not info.get('gender') else ["","Male","Female","Other"].index(info['gender']))
    new_pd= st.text_input("Preferred Diet", st.session_state.preferred_diet)
    upf= st.file_uploader("Update Profile Pic", type=["jpg","jpeg","png"])
    if st.button("Save Profile"):
        c=conn.cursor()
        c.execute("UPDATE users SET height=?, weight=?, age=?, gender=? WHERE id=?",
                  (new_ht if new_ht>0 else None, new_wt if new_wt>0 else None, new_ag if new_ag>0 else None, new_gd if new_gd else None, st.session_state.user_id))
        conn.commit()
        picpat= info.get('profile_pic','')
        if upf:
            picpat= save_file(upf,"profile_pics")
            c.execute("UPDATE users SET profile_pic=? WHERE id=?", (picpat, st.session_state.user_id))
            conn.commit()
        st.session_state.user_info['height']= new_ht if new_ht>0 else None
        st.session_state.user_info['weight']= new_wt if new_wt>0 else None
        st.session_state.user_info['age']= new_ag if new_ag>0 else None
        st.session_state.user_info['gender']= new_gd if new_gd else None
        st.session_state.preferred_diet= new_pd if new_pd else "Not specified"
        if picpat:
            st.session_state.user_info['profile_pic']= picpat
        st.success("Profile updated.")

# LOGOUT TAB
with tabs[4]:
    st.header("Logout")
    if st.button("Confirm Logout"):
        st.session_state.logged_in=False
        st.session_state.user_id=None
        st.session_state.username=""
        st.session_state.user_info={}
        st.session_state.preferred_diet="Not specified"
        st.success("You have been logged out.")
        if st.button("Go to Login"):
            st.experimental_rerun()
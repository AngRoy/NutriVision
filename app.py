import os
import sqlite3
import datetime
import json
import time
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, BlipProcessor, BlipForConditionalGeneration
import plotly.express as px
import numpy as np

st.markdown("""
<style>
/* Starfield background */
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
/* Main app container */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #000000, #121212) !important;
}
/* Header styling */
header {
    background-color: #1F1F1F !important;
}
/* Sidebar styling */
[data-testid="stSidebar"] {
    background-color: #1F1F1F !important;
}
/* General text styling */
body, .stMarkdown, .stMetric, .css-1aumxhk {
    color: #E8EAF6 !important;
}
/* Tabs styling */
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
/* Button styling - bright red */
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
/* Input styling */
.stNumberInput input, .stTextInput input, .stSelectbox select {
    background-color: #263238 !important;
    color: #E8EAF6 !important;
    border: 1px solid #37474F !important;
}
/* File uploader styling */
.stFileUploader {
    background-color: #263238 !important;
    color: #E8EAF6 !important;
    border: 1px solid #37474F !important;
}
</style>
<div id="starfield"></div>
""", unsafe_allow_html=True)

if "loaded_once" not in st.session_state:
    spinner = st.empty()
    spinner.markdown("""
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
    spinner.empty()
    st.session_state.loaded_once = True

# DB & UTILITIES
@st.cache_resource(show_spinner=False)
def init_db():
    conn = sqlite3.connect("myapp.db", check_same_thread=False)
    cur = conn.cursor()
    # Minimal check
    cur.execute("CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE, password TEXT, height REAL, weight REAL, age INTEGER, gender TEXT, profile_pic TEXT)")
    cur.execute("CREATE TABLE IF NOT EXISTS meals (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, meal_time TIMESTAMP, source TEXT, caption TEXT, predicted TEXT, calories REAL, meal_image TEXT, FOREIGN KEY (user_id) REFERENCES users(id))")
    conn.commit()
    return conn

conn = init_db()

def register_user(username, password, height, weight, age, gender, profile_pic):
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (username, password, height, weight, age, gender, profile_pic) VALUES (?,?,?,?,?,?,?)",
                  (username, password, height, weight, age, gender, profile_pic))
        conn.commit()
        c.execute("SELECT id FROM users WHERE username=?", (username,))
        user_id = c.fetchone()[0]
        return True, "Registration successful!", user_id
    except sqlite3.IntegrityError:
        return False, "That username already exists.", None

def login_user(username, password):
    c = conn.cursor()
    c.execute("SELECT id FROM users WHERE username=? AND password=?", (username, password))
    row = c.fetchone()
    return row[0] if row else None

def store_meal(user_id, source, caption, predicted, calories, image_path):
    c = conn.cursor()
    c.execute("""INSERT INTO meals (user_id, meal_time, source, caption, predicted, calories, meal_image)
                 VALUES (?,?,?,?,?,?,?)""",
              (user_id, datetime.datetime.now(), source, caption, json.dumps(predicted), calories, image_path))
    conn.commit()

def get_meals(user_id):
    c = conn.cursor()
    c.execute("SELECT meal_time, source, caption, predicted, calories, meal_image FROM meals WHERE user_id=? ORDER BY meal_time DESC", (user_id,))
    return c.fetchall()

def save_file(uploaded_file, folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    path = os.path.join(folder, uploaded_file.name)
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path

# MODEL
class MultiHeadModel(nn.Module):
    def __init__(self, dim_food=3, dim_fv=9, dim_fast=8, device="cuda", fine_tune_clip=False):
        super().__init__()
        self.device = device
        from transformers import CLIPProcessor, CLIPModel, BlipProcessor, BlipForConditionalGeneration
        self.clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        for p in self.clip_model.text_model.parameters():
            p.requires_grad = False
        if not fine_tune_clip:
            for p in self.clip_model.vision_model.parameters():
                p.requires_grad = False

        self.blip_proc = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        for p in self.blip_model.parameters():
            p.requires_grad = False

        fuse_dim = 1024
        h = 512
        self.food_head = nn.Sequential(
            nn.Linear(fuse_dim, h),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(h, dim_food)
        )
        self.fv_head = nn.Sequential(
            nn.Linear(fuse_dim, h),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(h, dim_fv)
        )
        self.fast_head = nn.Sequential(
            nn.Linear(fuse_dim, h),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(h, dim_fast)
        )

    def forward(self, img, source):
        # BLIP caption
        proc = self.blip_proc(images=img, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out_ids = self.blip_model.generate(**proc, max_length=50, num_beams=5)
        caption = self.blip_proc.decode(out_ids[0], skip_special_tokens=True)

        # CLIP embeddings
        clip_in = self.clip_proc(images=img, return_tensors="pt").to(self.device)
        img_emb = self.clip_model.get_image_features(**clip_in)
        txt_in = self.clip_proc(text=[caption], return_tensors="pt").to(self.device)
        txt_emb = self.clip_model.get_text_features(**txt_in)

        img_emb = img_emb / img_emb.norm(p=2, dim=-1, keepdim=True)
        txt_emb = txt_emb / txt_emb.norm(p=2, dim=-1, keepdim=True)
        fused = torch.cat([img_emb, txt_emb], dim=-1)

        if source=="food_nutrition":
            out = self.food_head(fused)
        elif source=="fv":
            out = self.fv_head(fused)
        elif source=="fastfood":
            out = self.fast_head(fused)
        else:
            raise ValueError("Unknown source!")
        return out.squeeze(0).cpu().numpy(), caption

@st.cache_resource(show_spinner=False)
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MultiHeadModel().to(device)
    ckpt_path = "nutrivision_multihand.pt"
    if os.path.exists(ckpt_path):
        sd = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(sd)
    model.eval()
    return model, device

def infer(img, source, model, device):
    preds, caption = model(img, source)
    cals = preds[0] if len(preds)>0 else 0
    return preds.tolist(), caption, cals

# INIT SESSION
if "logged_in" not in st.session_state: st.session_state.logged_in=False
if "user_id" not in st.session_state: st.session_state.user_id=None
if "username" not in st.session_state: st.session_state.username=""
if "user_info" not in st.session_state: st.session_state.user_info={}
if "preferred_diet" not in st.session_state: st.session_state.preferred_diet="Not specified"

# AUTH FORMS
def do_login():
    st.subheader("Login")
    user = st.text_input("Username", key="l_user")
    pw = st.text_input("Password", type="password", key="l_pw")
    if st.button("Login"):
        uid = login_user(user, pw)
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
            st.success("Logged in!")
            if st.button("Continue"):
                st.experimental_rerun()
        else:
            st.error("Invalid login")

def do_register():
    st.subheader("Register")
    r_user = st.text_input("Username", key="r_user")
    r_pw   = st.text_input("Password", type="password", key="r_pw")
    r_h    = st.number_input("Height (cm)", 0.0, 300.0, value=0.0, step=0.1)
    r_w    = st.number_input("Weight (kg)", 0.0, 300.0, value=0.0, step=0.1)
    r_a    = st.number_input("Age",0,120, value=0)
    r_g    = st.selectbox("Gender", ["","Male","Female","Other"])
    r_pd   = st.text_input("Preferred Diet","")
    r_pic  = st.file_uploader("Upload Profile Pic (optional)", type=["jpg","jpeg","png"])
    if st.button("Register"):
        pic_path=""
        if r_pic:
            pic_path=save_file(r_pic,"profile_pics")
        if r_user=="" or r_pw=="":
            st.error("Username & Password required")
        else:
            succ, msg, rid= register_user(r_user, r_pw, r_h if r_h>0 else None, r_w if r_w>0 else None, r_a if r_a>0 else None, r_g if r_g else None, pic_path)
            if succ:
                st.success(msg)
                st.session_state.logged_in=True
                st.session_state.user_id=rid
                st.session_state.username=r_user
                st.session_state.user_info={
                    "height": r_h if r_h>0 else None,
                    "weight": r_w if r_w>0 else None,
                    "age": r_a if r_a>0 else None,
                    "gender": r_g if r_g else None,
                    "profile_pic": pic_path
                }
                st.session_state.preferred_diet=r_pd if r_pd else "Not specified"
                if st.button("Continue"):
                    st.experimental_rerun()
            else:
                st.error(msg)

# MAIN
if not st.session_state.logged_in:
    st.subheader("Welcome! Please choose an action:")
    if st.button("Show Login"):
        do_login()
    if st.button("Show Register"):
        do_register()
    st.stop()

# Show tabs
tabs=st.tabs(["Home","Upload Meal","Meal History","Account","Logout"])

with tabs[0]:
    st.title("Dashboard")
    st.write(f"Hello, {st.session_state.username}!")
    today=datetime.date.today()
    cal_today=get_daily_calories(st.session_state.user_id, today)
    st.metric("Today's Calorie Intake", f"{cal_today:.1f} kcal")

    all_cals=get_all_daily_calories(st.session_state.user_id)
    if all_cals:
        import pandas as pd
        df=pd.DataFrame(all_cals, columns=["Date","Calories"])
        fig=px.line(df, x="Date", y="Calories", title="Daily Calorie Intake", markers=True)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("No meal records yet.")

with tabs[1]:
    st.title("Upload Meal")
    st.write("Analyze your meal's nutritional content using AI.")
    cat_map={"Food Nutrition":"food_nutrition","Fruits & Vegetables":"fv","Fast Food":"fastfood"}
    choice=st.selectbox("Meal Category", list(cat_map.keys()))
    cat=cat_map[choice]
    up=st.file_uploader("Upload image", type=["jpg","jpeg","png"], key="up_meal")
    if up:
        try:
            meal_img=Image.open(up).convert("RGB")
            st.image(meal_img, caption="Meal Image", use_container_width=True)
        except:
            st.error("Could not open image.")
        if st.button("Analyze"):
            with st.spinner("Analyzing..."):
                model,device=load_model()
                preds, cap, cals=infer(meal_img, cat, model, device)
            st.success("Done!")
            st.write("Caption:", cap)
            if cat=="food_nutrition":
                columns=["Caloric Value","Fat","Carbohydrates"]
            elif cat=="fv":
                columns=["energy (kcal/kJ)","water (g)","protein (g)","total fat (g)","carbohydrates (g)","fiber (g)","sugars (g)","calcium (mg)","iron (mg)"]
            else:
                columns=["calories","cal_fat","total_fat","sat_fat","trans_fat","cholesterol","sodium","total_carb"]
            st.table({
                "Nutrient": columns,
                "Value": [round(x,2) for x in preds]
            })
            st.write("Calories:", f"{cals:.1f}")
            path=save_file(up,"meal_images")
            store_meal(st.session_state.user_id, cat, cap, preds, cals, path)
            st.success("Meal stored successfully!")
            # Ask user if they want to refresh the page
            if st.button("Refresh Meal History Now?"):
                st.experimental_rerun()

with tabs[2]:
    st.title("Meal History")
    meals=get_meals(st.session_state.user_id)
    if meals:
        for m in meals:
            t, so, cap, pr, cals, imgp=m
            st.write(f"**Time**: {t}, **Category**: {so}")
            st.write(f"**Caption**: {cap}, **Calories**: {cals:.1f}")
            if imgp and os.path.exists(imgp):
                st.image(imgp, use_container_width=True)
            try:
                arr=json.loads(pr)
                if so=="food_nutrition":
                    cCols=["Caloric Value","Fat","Carbohydrates"]
                elif so=="fv":
                    cCols=["energy (kcal/kJ)","water (g)","protein (g)","total fat (g)","carbohydrates (g)","fiber (g)","sugars (g)","calcium (mg)","iron (mg)"]
                else:
                    cCols=["calories","cal_fat","total_fat","sat_fat","trans_fat","cholesterol","sodium","total_carb"]
                st.table({
                    "Nutrient": cCols,
                    "Value":[round(x,2) for x in arr]
                })
            except:
                st.write("Raw preds:", pr)
            st.markdown("---")
    else:
        st.write("No meals recorded yet.")

with tabs[3]:
    st.title("Account Info")
    st.write(f"**Username**: {st.session_state.username}")
    inf=st.session_state.user_info
    if inf:
        h=inf.get("height",None)
        w=inf.get("weight",None)
        a=inf.get("age",None)
        g=inf.get("gender",None)
        st.write(f"**Height**: {h if h else 'N/A'} cm")
        st.write(f"**Weight**: {w if w else 'N/A'} kg")
        st.write(f"**Age**: {a if a else 'N/A'}")
        st.write(f"**Gender**: {g if g else 'N/A'}")
        if h and w and h>0:
            bmi=w/((h/100)**2)
            st.write(f"**BMI**: {bmi:.2f}")
        st.write(f"**Preferred Diet**: {st.session_state.preferred_diet}")
        pic=inf.get("profile_pic","")
        if pic and os.path.exists(pic):
            st.image(pic, use_container_width=True)
        else:
            st.write("No profile picture.")
    else:
        st.write("No user info found.")

with tabs[4]:
    st.title("Logout")
    if st.button("Confirm Logout"):
        st.session_state.logged_in=False
        st.session_state.user_id=None
        st.session_state.username=""
        st.session_state.user_info={}
        st.session_state.preferred_diet="Not specified"
        st.success("Logged out.")
        if st.button("Go to Login"):
            st.experimental_rerun()
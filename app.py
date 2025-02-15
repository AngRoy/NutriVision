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

# =============================
# PAGE CONFIG & MOBILE FRIENDLY
# =============================
st.set_page_config(page_title="NutriVision", layout="wide")

# =============================
# STARFIELD THEME & LOADING SCREEN
# =============================

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

# Show loading spinner once for 2 seconds
if "loaded_once" not in st.session_state:
    loader_spot = st.empty()
    loader_spot.markdown("""
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
    loader_spot.empty()
    st.session_state.loaded_once = True

# =============================
# DB & UTILITIES
# =============================
@st.cache_resource(show_spinner=False)
def init_db():
    conn = sqlite3.connect("nutrivision_app.db", check_same_thread=False)
    c = conn.cursor()
    c.execute("PRAGMA table_info(users)")
    user_cols = [col[1] for col in c.fetchall()]
    if not user_cols or "height" not in user_cols or "profile_pic" not in user_cols:
        c.execute("DROP TABLE IF EXISTS users")
    c.execute("PRAGMA table_info(meals)")
    meal_cols = [col[1] for col in c.fetchall()]
    if not meal_cols or "meal_image" not in meal_cols:
        c.execute("DROP TABLE IF EXISTS meals")
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

def register_user(username, password, height, weight, age, gender, profile_pic):
    try:
        c = conn.cursor()
        c.execute("INSERT INTO users (username, password, height, weight, age, gender, profile_pic) VALUES (?,?,?,?,?,?,?)",
                  (username, password, height, weight, age, gender, profile_pic))
        conn.commit()
        c.execute("SELECT id FROM users WHERE username=?", (username,))
        uid = c.fetchone()[0]
        return True, "User registered successfully.", uid
    except sqlite3.IntegrityError:
        return False, "Username already exists.", None

def login_user(username, password):
    c = conn.cursor()
    c.execute("SELECT id FROM users WHERE username=? AND password=?", (username, password))
    row = c.fetchone()
    if row: return row[0]
    return None

def store_meal(user_id, source, caption, predicted, calories, meal_path):
    c = conn.cursor()
    c.execute("INSERT INTO meals (user_id, meal_time, source, caption, predicted, calories, meal_image) VALUES (?,?,?,?,?,?,?)",
              (user_id, datetime.datetime.now(), source, caption, json.dumps(predicted), calories, meal_path))
    conn.commit()

def get_meal_history(user_id):
    c = conn.cursor()
    c.execute("SELECT meal_time, source, caption, predicted, calories, meal_image FROM meals WHERE user_id=? ORDER BY meal_time DESC", (user_id,))
    return c.fetchall()

def get_daily_calories(user_id, date):
    c = conn.cursor()
    start = datetime.datetime.combine(date, datetime.time.min)
    end = datetime.datetime.combine(date, datetime.time.max)
    c.execute("SELECT SUM(calories) FROM meals WHERE user_id=? AND meal_time BETWEEN ? AND ?", (user_id, start, end))
    val = c.fetchone()[0]
    return val if val else 0

def get_all_daily_calories(user_id):
    c = conn.cursor()
    c.execute("SELECT DATE(meal_time), SUM(calories) FROM meals WHERE user_id=? GROUP BY DATE(meal_time) ORDER BY DATE(meal_time)", (user_id,))
    return c.fetchall()

def save_uploaded_file(upfile, folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    path = os.path.join(folder, upfile.name)
    with open(path, "wb") as f:
        f.write(upfile.getbuffer())
    return path

# =============================
# MODEL
# =============================
class NutriVisionNetMultiHead(nn.Module):
    def __init__(self, food_dim=3, fv_dim=9, fast_dim=8, device="cuda", fine_tune_clip=False):
        super().__init__()
        from transformers import CLIPProcessor, CLIPModel, BlipProcessor, BlipForConditionalGeneration
        self.device = device
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

        fuse_dim=1024
        hid=512
        self.food_head=nn.Sequential(
            nn.Linear(fuse_dim, hid), nn.ReLU(), nn.Dropout(0.2), nn.Linear(hid, food_dim)
        )
        self.fv_head=nn.Sequential(
            nn.Linear(fuse_dim, hid), nn.ReLU(), nn.Dropout(0.2), nn.Linear(hid, fv_dim)
        )
        self.fast_head=nn.Sequential(
            nn.Linear(fuse_dim, hid), nn.ReLU(), nn.Dropout(0.2), nn.Linear(hid, fast_dim)
        )

    def forward(self, img, source):
        # BLIP caption
        proc= self.blip_proc(images=img, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out_ids=self.blip_model.generate(**proc, max_length=50, num_beams=5)
        caption=self.blip_proc.decode(out_ids[0], skip_special_tokens=True)

        # CLIP
        clip_in=self.clip_proc(images=img, return_tensors="pt").to(self.device)
        img_emb=self.clip_model.get_image_features(**clip_in)
        clip_txt=self.clip_proc(text=[caption], return_tensors="pt").to(self.device)
        txt_emb=self.clip_model.get_text_features(**clip_txt)

        img_emb=img_emb/img_emb.norm(p=2, dim=-1, keepdim=True)
        txt_emb=txt_emb/txt_emb.norm(p=2, dim=-1, keepdim=True)
        fused=torch.cat([img_emb, txt_emb], dim=-1)

        if source=="food_nutrition":
            out=self.food_head(fused)
        elif source=="fv":
            out=self.fv_head(fused)
        elif source=="fastfood":
            out=self.fast_head(fused)
        else:
            raise ValueError("Unknown source")
        return out, caption

@st.cache_resource(show_spinner=False)
def load_model():
    dev="cuda" if torch.cuda.is_available() else "cpu"
    net=NutriVisionNetMultiHead(device=dev, fine_tune_clip=False).to(dev)
    ckpt="nutrivision_multihand.pt"
    if os.path.exists(ckpt):
        sd=torch.load(ckpt, map_location=dev)
        net.load_state_dict(sd)
    net.eval()
    return net, dev

def infer_meal(img, source, net, dev):
    with torch.no_grad():
        out, cap= net(img, source)
    arr= out.squeeze(0).cpu().numpy().tolist()
    cals= arr[0] if arr else 0
    return arr, cap, cals

# =============================
# SESSION
# =============================
if "logged_in" not in st.session_state: st.session_state.logged_in=False
if "user_id" not in st.session_state:   st.session_state.user_id=None
if "username" not in st.session_state:  st.session_state.username=""
if "user_info" not in st.session_state: st.session_state.user_info={}
if "preferred_diet" not in st.session_state: st.session_state.preferred_diet="Not specified"


# =============================
# AUTH
# =============================
def login_form():
    st.subheader("Login")
    user= st.text_input("Username", key="login_username")
    pw=   st.text_input("Password", type="password", key="login_password")
    if st.button("Login", key="login_button"):
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
            st.session_state.preferred_diet="Not specified"
            st.success("Logged in successfully!")
        else:
            st.error("Invalid username or password.")

def registration_form():
    st.subheader("Register")
    reg_user= st.text_input("Username", key="reg_username")
    reg_pw=   st.text_input("Password", type="password", key="reg_password")
    reg_h=    st.number_input("Height (cm) [Optional]",0.0,300.0,step=0.1, key="reg_height", value=0.0)
    reg_w=    st.number_input("Weight (kg) [Optional]",0.0,300.0,step=0.1, key="reg_weight", value=0.0)
    reg_a=    st.number_input("Age [Optional]",0,120,step=1, key="reg_age", value=0)
    reg_g=    st.selectbox("Gender [Optional]", ["","Male","Female","Other"], key="reg_gender")
    reg_pd=   st.text_input("Preferred Diet [Optional]", key="reg_preferred_diet")
    reg_pic=  st.file_uploader("Upload Profile Picture [Optional]", type=["jpg","jpeg","png"], key="reg_profile")

    if st.button("Register", key="reg_button"):
        pic_path=""
        if reg_pic:
            pic_path=save_uploaded_file(reg_pic,"profile_pics")
        if reg_user=="" or reg_pw=="":
            st.error("Username & password required!")
        else:
            succ, msg, new_uid= register_user(
                reg_user, reg_pw,
                float(reg_h) if reg_h>0 else None,
                float(reg_w) if reg_w>0 else None,
                int(reg_a) if reg_a>0 else None,
                reg_g if reg_g else None,
                pic_path
            )
            if succ:
                st.success(msg)
                st.session_state.logged_in=True
                st.session_state.user_id=new_uid
                st.session_state.username=reg_user
                st.session_state.user_info={
                    "height": float(reg_h) if reg_h>0 else None,
                    "weight": float(reg_w) if reg_w>0 else None,
                    "age": int(reg_a) if reg_a>0 else None,
                    "gender": reg_g if reg_g else None,
                    "profile_pic": pic_path
                }
                st.session_state.preferred_diet= reg_pd if reg_pd else "Not specified"
                st.query_params={"user_id":[str(new_uid)],"username":[reg_user]}
                st.success("Registered & Logged in!")
            else:
                st.error(msg)

if not st.session_state.logged_in:
    st.write("**Welcome!** Please either login or register below.")
    colA, colB= st.columns(2)
    with colA:
        if st.button("Show Login Form"):
            login_form()
    with colB:
        if st.button("Show Registration Form"):
            registration_form()
    st.stop()

# =============================
# TABS
# =============================
tabs= st.tabs(["Home","Upload Meal","Meal History","Account","Logout"])

with tabs[0]:
    st.header("Dashboard")
    st.write(f"Hello, {st.session_state.username}!")
    today= datetime.date.today()
    daily_cal= get_daily_calories(st.session_state.user_id, today)
    st.metric("Today's Calorie Intake", f"{daily_cal:.2f} kcal")

    # Optional: Manual refresh
    if st.button("Refresh Dashboard"):
        st.experimental_rerun()

    # Show daily chart
    daily_data= get_all_daily_calories(st.session_state.user_id)
    if daily_data:
        import pandas as pd
        df= pd.DataFrame(daily_data, columns=["Date","Calories"])
        fig= px.line(df, x="Date", y="Calories", title="Daily Calorie Intake", markers=True)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("No meal records yet.")

with tabs[1]:
    st.header("Upload a Meal")
    st.write("Upload a meal image for AI-based nutritional analysis.")
    source_opts= {"Food Nutrition":"food_nutrition","Fruits & Vegetables":"fv","Fast Food":"fastfood"}
    cat_choice= st.selectbox("Meal Category", list(source_opts.keys()))
    cat= source_opts[cat_choice]

    up= st.file_uploader("Choose meal image", type=["jpg","jpeg","png"], key="meal_upload")
    if up:
        try:
            meal_image= Image.open(up).convert("RGB")
            st.image(meal_image, caption="Meal Image", use_container_width=True)
        except:
            st.error("Couldn't load image.")
        if st.button("Analyze Meal"):
            with st.spinner("Analyzing..."):
                net, dev= load_model()
                preds, cap, cals= infer_meal(meal_image, cat, net, dev)
            st.success("Inference complete!")
            st.write("**Caption**:", cap)
            # Nutrient columns
            if cat=="food_nutrition":
                col_list= ["Caloric Value","Fat","Carbohydrates"]
            elif cat=="fv":
                col_list= ["energy (kcal/kJ)","water (g)","protein (g)","total fat (g)","carbohydrates (g)","fiber (g)","sugars (g)","calcium (mg)","iron (mg)"]
            else:
                col_list= ["calories","cal_fat","total_fat","sat_fat","trans_fat","cholesterol","sodium","total_carb"]
            st.table({
                "Nutrient": col_list,
                "Value": [round(x,2) for x in preds]
            })
            st.write("Predicted Calories:", f"{cals:.2f}")
            path= save_uploaded_file(up,"meal_images")
            store_meal(st.session_state.user_id, cat, cap, preds, cals, path)
            st.success("Meal stored successfully!")
            # Prompt manual refresh for Meal History
            st.info("Switch to 'Meal History' or refresh the Dashboard to see updates.")

with tabs[2]:
    st.header("Meal History")
    meals= get_meal_history(st.session_state.user_id)
    if meals:
        for m in meals:
            meal_time, source, cap, predicted, cals, meal_img = m
            st.write(f"**Time**: {meal_time}, **Category**: {source}")
            if meal_img and os.path.exists(meal_img):
                st.image(meal_img, use_container_width=True)
            st.write(f"**Caption**: {cap}")
            st.write(f"**Calories**: {cals:.2f}")
            try:
                arr= json.loads(predicted)
                if source=="food_nutrition":
                    cCols= ["Caloric Value","Fat","Carbohydrates"]
                elif source=="fv":
                    cCols= ["energy (kcal/kJ)","water (g)","protein (g)","total fat (g)","carbohydrates (g)","fiber (g)","sugars (g)","calcium (mg)","iron (mg)"]
                else:
                    cCols= ["calories","cal_fat","total_fat","sat_fat","trans_fat","cholesterol","sodium","total_carb"]
                st.table({
                    "Nutrient": cCols,
                    "Value": [round(val,2) for val in arr]
                })
            except:
                st.write("Predicted raw:", predicted)
            st.markdown("---")
    else:
        st.write("No meals recorded yet.")
    
    # Chart of daily cals
    daily_data= get_all_daily_calories(st.session_state.user_id)
    if daily_data:
        import pandas as pd
        df= pd.DataFrame(daily_data, columns=["Date","Calories"])
        fig= px.bar(df, x="Date", y="Calories", title="Daily Calorie Intake")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("No daily data to chart.")

with tabs[3]:
    st.header("Account Information")
    st.write(f"**Username**: {st.session_state.username}")
    ui= st.session_state.user_info
    if ui:
        h= ui.get('height', None)
        w= ui.get('weight', None)
        a= ui.get('age', None)
        g= ui.get('gender', None)
        st.write(f"**Height**: {h if h else 'N/A'} cm")
        st.write(f"**Weight**: {w if w else 'N/A'} kg")
        st.write(f"**Age**: {a if a else 'N/A'}")
        st.write(f"**Gender**: {g if g else 'N/A'}")
        if h and w and h>0:
            bmi= w/((h/100)**2)
            st.write(f"**BMI**: {bmi:.2f}")
        st.write(f"**Preferred Diet**: {st.session_state.preferred_diet}")
        pic= ui.get('profile_pic','')
        if pic and os.path.exists(pic):
            st.image(pic, use_container_width=True)
        else:
            st.write("No profile picture.")
    else:
        st.write("No user info available.")
    
    st.markdown("---")
    st.subheader("Update Profile")
    new_h= st.number_input("Height (cm)", 0.0, 300.0, step=0.1, value=float(ui.get('height') or 0))
    new_w= st.number_input("Weight (kg)", 0.0, 300.0, step=0.1, value=float(ui.get('weight') or 0))
    new_a= st.number_input("Age", 0,120, step=1, value=int(ui.get('age') or 0))
    new_g= st.selectbox("Gender", ["","Male","Female","Other"], index=0 if not ui.get('gender') else ["","Male","Female","Other"].index(ui['gender']))
    new_pd= st.text_input("Preferred Diet", st.session_state.preferred_diet)
    new_pic= st.file_uploader("Update Profile Picture", type=["jpg","jpeg","png"])
    if st.button("Update Profile"):
        c=conn.cursor()
        c.execute("UPDATE users SET height=?, weight=?, age=?, gender=? WHERE id=?",
                  (new_h if new_h>0 else None, new_w if new_w>0 else None, new_a if new_a>0 else None,
                   new_g if new_g else None, st.session_state.user_id))
        conn.commit()
        picpath= ui.get('profile_pic','')
        if new_pic:
            picpath= save_uploaded_file(new_pic, "profile_pics")
            c.execute("UPDATE users SET profile_pic=? WHERE id=?", (picpath, st.session_state.user_id))
            conn.commit()
        st.session_state.user_info['height']= new_h if new_h>0 else None
        st.session_state.user_info['weight']= new_w if new_w>0 else None
        st.session_state.user_info['age']= new_a if new_a>0 else None
        st.session_state.user_info['gender']= new_g if new_g else None
        st.session_state.preferred_diet= new_pd if new_pd else "Not specified"
        if picpath:
            st.session_state.user_info['profile_pic']= picpath
        st.success("Profile updated successfully!")

with tabs[4]:
    st.header("Logout")
    if st.button("Confirm Logout"):
        st.session_state.logged_in=False
        st.session_state.user_id=None
        st.session_state.username=""
        st.session_state.user_info={}
        st.session_state.preferred_diet="Not specified"
        st.query_params={}
        st.success("You have been logged out.")
        if st.button("Confirm"):
            st.experimental_rerun()

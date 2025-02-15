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
# PAGE CONFIG & MOBILE FRIENDLY
# =============================
st.set_page_config(page_title="NutriVision", layout="wide")

# =============================
# STARFIELD BACKGROUND & STYLING
# =============================
STARFIELD_CSS = """
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
/* Gradient background */
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
/* Make titles bigger & more eye-catching */
h1, .app-title {
    font-size: 2.5rem !important;
    color: #fff !important;
    text-align: center;
    margin-top: 10px;
}
h2, .app-subtitle {
    font-size: 1.75rem !important;
    color: #E8EAF6 !important;
    margin-top: 10px;
}
h3 {
    font-size: 1.4rem !important;
    color: #E8EAF6 !important;
    margin-top: 10px;
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
/* Buttons - bright red */
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
/* Inputs */
.stNumberInput input, .stTextInput input, .stSelectbox select {
    background-color: #263238 !important;
    color: #E8EAF6 !important;
    border: 1px solid #37474F !important;
}
/* File uploader */
.stFileUploader {
    background-color: #263238 !important;
    color: #E8EAF6 !important;
    border: 1px solid #37474F !important;
}
/* Make images 50% width of container, auto height */
.css-1lyj5qt img, .stImage > img {
    max-width: 50% !important;
    height: auto !important;
    display: block;
    margin: 0 auto;
}
</style>
<div id="starfield"></div>
"""

st.markdown(STARFIELD_CSS, unsafe_allow_html=True)

# =============================
# SHOW LOADING SCREEN ONCE
# =============================
if "loaded_once" not in st.session_state:
    spin_area = st.empty()
    spin_area.markdown("""
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
    spin_area.empty()
    st.session_state.loaded_once = True

# =============================
# INITIALIZE DB
# =============================
@st.cache_resource
def init_db():
    conn = sqlite3.connect("nutrivision_app.db", check_same_thread=False)
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
# DB UTILS
# =============================
def register_user(username, password, height, weight, age, gender, pic_path):
    c = conn.cursor()
    try:
        c.execute("""
            INSERT INTO users (username, password, height, weight, age, gender, profile_pic)
            VALUES (?,?,?,?,?,?,?)
        """, (username, password, height, weight, age, gender, pic_path))
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

def store_meal(uid, src, cap, preds, cals, img_path):
    c = conn.cursor()
    c.execute("""
        INSERT INTO meals (user_id, meal_time, source, caption, predicted, calories, meal_image)
        VALUES (?,?,?,?,?,?,?)
    """, (uid, datetime.datetime.now(), src, cap, json.dumps(preds), cals, img_path))
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

def save_file(uploaded, folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    path = os.path.join(folder, uploaded.name)
    with open(path,"wb") as f:
        f.write(uploaded.getbuffer())
    return path

# =============================
# MODEL
# =============================
class NutriVisionNetMultiHead(nn.Module):
    def __init__(self, food_dim=3, fv_dim=9, fast_dim=8, device="cuda", fine_tune_clip=False):
        super().__init__()
        from transformers import CLIPProcessor, CLIPModel, BlipProcessor, BlipForConditionalGeneration
        
        self.device = device
        self.clip_proc= CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_model=CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        for p in self.clip_model.text_model.parameters():
            p.requires_grad=False
        if not fine_tune_clip:
            for p in self.clip_model.vision_model.parameters():
                p.requires_grad=False

        self.blip_proc= BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.blip_model=BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        for p in self.blip_model.parameters():
            p.requires_grad=False

        fuse_dim=1024
        hidden=512
        self.food_head= nn.Sequential(
            nn.Linear(fuse_dim, hidden), nn.ReLU(), nn.Dropout(0.2), nn.Linear(hidden, food_dim)
        )
        self.fv_head= nn.Sequential(
            nn.Linear(fuse_dim, hidden), nn.ReLU(), nn.Dropout(0.2), nn.Linear(hidden, fv_dim)
        )
        self.fast_head=nn.Sequential(
            nn.Linear(fuse_dim, hidden), nn.ReLU(), nn.Dropout(0.2), nn.Linear(hidden, fast_dim)
        )

    def forward(self, img, source):
        # BLIP for caption
        p=self.blip_proc(images=img, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out_ids= self.blip_model.generate(**p, max_length=50, num_beams=5)
        caption= self.blip_proc.decode(out_ids[0], skip_special_tokens=True)

        # CLIP embeddings
        clip_in= self.clip_proc(images=img, return_tensors="pt").to(self.device)
        img_emb= self.clip_model.get_image_features(**clip_in)
        txt_in= self.clip_proc(text=[caption], return_tensors="pt").to(self.device)
        txt_emb= self.clip_model.get_text_features(**txt_in)

        # Normalize & fuse
        img_emb= img_emb / img_emb.norm(p=2, dim=-1, keepdim=True)
        txt_emb= txt_emb / txt_emb.norm(p=2, dim=-1, keepdim=True)
        fused= torch.cat([img_emb, txt_emb], dim=-1)

        if source=="food_nutrition":
            out= self.food_head(fused)
        elif source=="fv":
            out= self.fv_head(fused)
        elif source=="fastfood":
            out= self.fast_head(fused)
        else:
            raise ValueError("Unknown source")
        return out.squeeze(0).cpu().numpy().tolist(), caption

@st.cache_resource
def load_model():
    dev="cuda" if torch.cuda.is_available() else "cpu"
    net= NutriVisionNetMultiHead(device=dev, fine_tune_clip=False).to(dev)
    ckpt= "nutrivision_multihand.pt"
    if os.path.exists(ckpt):
        sd=torch.load(ckpt, map_location=dev)
        net.load_state_dict(sd)
    net.eval()
    return net, dev

def run_inference(img, src, net, dev):
    with torch.no_grad():
        arr, caption= net(img, src)
    cals= arr[0] if len(arr)>0 else 0
    return arr, caption, cals

# =============================
# SESSION
# =============================
if "logged_in" not in st.session_state: st.session_state.logged_in=False
if "user_id" not in st.session_state:   st.session_state.user_id=None
if "username" not in st.session_state:  st.session_state.username=""
if "user_info" not in st.session_state: st.session_state.user_info={}
if "preferred_diet" not in st.session_state: st.session_state.preferred_diet="Not specified"

# =============================
# FORMS
# =============================
def show_login_form():
    st.subheader("Login")
    user=st.text_input("Username", key="login_user")
    pw=  st.text_input("Password", type="password", key="login_pw")
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
            st.session_state.preferred_diet="Not specified"
            st.button("Continue")
            st.success("You are now logged in!")
        else:
            st.error("Invalid username or password.")

def show_register_form():
    st.subheader("Register")
    r_user=st.text_input("Username", key="reg_user")
    r_pw=  st.text_input("Password", type="password", key="reg_pw")
    r_h=   st.number_input("Height (cm)",0.0,300.0,step=0.1)
    r_w=   st.number_input("Weight (kg)",0.0,300.0,step=0.1)
    r_a=   st.number_input("Age",0,120,step=1)
    r_g=   st.selectbox("Gender",["","Male","Female","Other"])
    r_pd=  st.text_input("Preferred Diet (optional)")
    pic=   st.file_uploader("Profile Pic (optional)", type=["jpg","jpeg","png"])

    if st.button("Register"):
        ppath=""
        if pic:
            ppath= save_file(pic,"profile_pics")
        if r_user=="" or r_pw=="":
            st.error("Username & Password are required!")
        else:
            succ,msg, uid= register_user(
                r_user, r_pw,
                r_h if r_h>0 else None,
                r_w if r_w>0 else None,
                r_a if r_a>0 else None,
                r_g if r_g else None,
                ppath
            )
            if succ:
                st.success(msg)
                st.session_state.logged_in=True
                st.session_state.user_id=uid
                st.session_state.username=r_user
                st.session_state.user_info={
                    "height": r_h if r_h>0 else None,
                    "weight": r_w if r_w>0 else None,
                    "age": r_a if r_a>0 else None,
                    "gender": r_g if r_g else None,
                    "profile_pic": ppath
                }
                st.session_state.preferred_diet= r_pd if r_pd else "Not specified"
                st.button("Continue")
            else:
                st.error(msg)

# If not logged in, show minimal forms
if not st.session_state.logged_in:
    st.markdown("<h1 class='app-title'>NutriVision</h1>", unsafe_allow_html=True)
    st.write("**Please login or register:**")
    colA, colB= st.columns(2)
    with colA:
        show_login_form()
    with colB:
        show_register_form()
    st.stop()

# =============================
# MAIN TABS
# =============================
tabs= st.tabs(["Home","Upload Meal","Meal History","Account","Logout"])

# HOME TAB
with tabs[0]:
    st.markdown("<h1 class='app-title'>Dashboard</h1>", unsafe_allow_html=True)
    st.write(f"Hello, **{st.session_state.username}**!")
    
    # Show daily cals
    today = datetime.date.today()
    cals_today= get_daily_cals(st.session_state.user_id, today)
    st.metric("Today's Calorie Intake", f"{cals_today:.2f} kcal")

    # Manual Refresh button if needed
    if st.button("Refresh"):
        st.experimental_set_query_params()  # Clear if any
        # No st.experimental_rerun to avoid the attribute error.
        # The user can manually refresh page if needed.

    # Plot daily chart
    daily_data= get_all_daily_cals(st.session_state.user_id)
    if daily_data:
        import pandas as pd
        df= pd.DataFrame(daily_data, columns=["Date","Calories"])
        fig= px.line(df, x="Date", y="Calories", title="Daily Calorie Intake", markers=True)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("No meal records available.")

# UPLOAD TAB
with tabs[1]:
    st.markdown("<h1 class='app-title'>Upload a Meal</h1>", unsafe_allow_html=True)
    st.write("Use AI model to analyze your meal.")
    cat_map= {"Food Nutrition":"food_nutrition","Fruits & Vegetables":"fv","Fast Food":"fastfood"}
    choice= st.selectbox("Meal Category", list(cat_map.keys()))
    source= cat_map[choice]

    meal_file= st.file_uploader("Choose an image", type=["jpg","jpeg","png"])
    if meal_file:
        try:
            meal_img= Image.open(meal_file).convert("RGB")
            st.image(meal_img, caption="Meal Image", use_container_width=True)
        except:
            st.error("Error reading image.")
        if st.button("Analyze Meal"):
            with st.spinner("Analyzing..."):
                model, dev= load_model()
                preds, cap, cals= run_inference(meal_img, source, model, dev)
            st.success("Analysis complete!")
            st.write("**Caption**:", cap)
            if source=="food_nutrition":
                cols=["Caloric Value","Fat","Carbohydrates"]
            elif source=="fv":
                cols=["energy (kcal/kJ)","water (g)","protein (g)","total fat (g)","carbohydrates (g)","fiber (g)","sugars (g)","calcium (mg)","iron (mg)"]
            else:
                cols=["calories","cal_fat","total_fat","sat_fat","trans_fat","cholesterol","sodium","total_carb"]
            st.table({
                "Nutrient": cols,
                "Value": [round(x,2) for x in preds]
            })
            st.write("**Predicted Calories**:", f"{cals:.2f}")
            path= save_file(meal_file,"meal_images")
            store_meal(st.session_state.user_id, source, cap, preds, cals, path)
            st.success("Meal saved to history!")
            st.info("Check 'Meal History' or refresh if you like.")

# HISTORY TAB
with tabs[2]:
    st.markdown("<h1 class='app-title'>Meal History</h1>", unsafe_allow_html=True)
    all_meals= get_meals(st.session_state.user_id)
    if all_meals:
        for m in all_meals:
            m_time, src, cap, pred_str, cals, m_img = m
            st.write(f"**Time**: {m_time}, **Category**: {src}")
            if m_img and os.path.exists(m_img):
                st.image(m_img, use_container_width=True)
            st.write(f"**Caption**: {cap}")
            st.write(f"**Calories**: {cals:.2f}")
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
                    "Value": [round(a,2) for a in arr]
                })
            except:
                st.write("Raw predictions:", pred_str)
            st.markdown("---")
    else:
        st.write("No meals yet.")

    # Chart of daily cals
    daily= get_all_daily_cals(st.session_state.user_id)
    if daily:
        import pandas as pd
        df= pd.DataFrame(daily, columns=["Date","Calories"])
        fig= px.bar(df, x="Date", y="Calories", title="Daily Calorie Intake")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("No daily data available.")

# ACCOUNT TAB
with tabs[3]:
    st.markdown("<h1 class='app-title'>Account Info</h1>", unsafe_allow_html=True)
    info= st.session_state.user_info
    st.write(f"**Username**: {st.session_state.username}")
    if info:
        h= info.get('height', None)
        w= info.get('weight', None)
        a= info.get('age', None)
        g= info.get('gender', None)
        st.write(f"**Height**: {h if h else 'N/A'} cm")
        st.write(f"**Weight**: {w if w else 'N/A'} kg")
        st.write(f"**Age**: {a if a else 'N/A'}")
        st.write(f"**Gender**: {g if g else 'N/A'}")
        if h and w and h>0:
            bmi= w/((h/100)**2)
            st.write(f"**BMI**: {bmi:.2f}")
        st.write(f"**Preferred Diet**: {st.session_state.preferred_diet}")
        pic= info.get('profile_pic','')
        if pic and os.path.exists(pic):
            st.image(pic, use_container_width=True)
        else:
            st.write("No profile picture.")
    else:
        st.write("No user info found.")

    st.markdown("---")
    st.subheader("Update Profile")
    new_h= st.number_input("Height (cm)",0.0,300.0,step=0.1, value=float(info.get('height') or 0))
    new_w= st.number_input("Weight (kg)",0.0,300.0,step=0.1, value=float(info.get('weight') or 0))
    new_a= st.number_input("Age", 0,120, step=1, value=int(info.get('age') or 0))
    new_g= st.selectbox("Gender", ["","Male","Female","Other"], index=0 if not info.get('gender') else ["","Male","Female","Other"].index(info['gender']))
    new_pd= st.text_input("Preferred Diet", st.session_state.preferred_diet)
    up_f= st.file_uploader("Update Profile Pic", type=["jpg","jpeg","png"])

    if st.button("Save Profile"):
        c= conn.cursor()
        c.execute("UPDATE users SET height=?, weight=?, age=?, gender=? WHERE id=?",
                  (new_h if new_h>0 else None, new_w if new_w>0 else None,
                   new_a if new_a>0 else None, new_g if new_g else None,
                   st.session_state.user_id))
        conn.commit()
        old_pic= info.get('profile_pic','')
        if up_f:
            new_pic= save_file(up_f,"profile_pics")
            c.execute("UPDATE users SET profile_pic=? WHERE id=?", (new_pic, st.session_state.user_id))
            conn.commit()
            st.session_state.user_info['profile_pic']= new_pic
        st.session_state.user_info['height']= new_h if new_h>0 else None
        st.session_state.user_info['weight']= new_w if new_w>0 else None
        st.session_state.user_info['age']= new_a if new_a>0 else None
        st.session_state.user_info['gender']= new_g if new_g else None
        st.session_state.preferred_diet= new_pd if new_pd else "Not specified"
        st.success("Profile updated.")

# LOGOUT TAB
with tabs[4]:
    st.markdown("<h1 class='app-title'>Logout</h1>", unsafe_allow_html=True)
    if st.button("Confirm Logout"):
        st.session_state.logged_in=False
        st.session_state.user_id=None
        st.session_state.username=""
        st.session_state.user_info={}
        st.session_state.preferred_diet="Not specified"
        st.success("You have been logged out.")
        st.button("Continue")

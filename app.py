import os
import time
import json
import datetime
import sqlite3
import base64
import io
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
# STARFIELD & CSS THEME
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
/* Header and Sidebar */
header, [data-testid="stSidebar"] {
    background-color: #1F1F1F !important;
}
/* Text styling */
body, .stMarkdown, .stMetric, .css-1aumxhk {
    color: #E8EAF6 !important;
}
/* Larger titles & text */
h1, .app-title {
    font-size: 2.5rem !important;
    text-align: center;
    color: #FFF !important;
    margin-top: 10px;
}
h2, .app-subtitle {
    font-size: 1.75rem !important;
    color: #E8EAF6 !important;
}
h3 {
    font-size: 1.4rem !important;
    color: #E8EAF6 !important;
}
/* Tabs styling - more advanced look */
.stTabs > div[role="tablist"] {
    background-color: #263238 !important;
    border-bottom: 2px solid #000000 !important;
    display: flex;
    justify-content: center;
}
.stTabs div[role="tab"] {
    font-size: 18px;
    font-weight: bold;
    margin: 0 10px;
    color: #E8EAF6 !important;
    padding: 10px;
    transition: background-color 0.3s ease;
    border-radius: 8px 8px 0 0;
}
.stTabs div[role="tab"]:hover {
    background-color: #37474F !important;
    cursor: pointer;
}
/* Buttons */
.stButton>button {
    background-color: #E53935 !important;
    color: #FFF !important;
    font-weight: bold;
    border: none;
    padding: 10px 20px;
    border-radius: 5px;
    transition: background-color 0.3s ease;
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
/* Smooth transitions for images */
.upload-image img {
    max-height: 50vh;
    width: auto;
    display: block;
    margin: 0 auto;
    transition: opacity 0.5s ease-in-out;
}
.history-image img {
    max-height: 20vh;
    width: auto;
    display: block;
    margin: 0 auto;
    transition: opacity 0.5s ease-in-out;
}
</style>
<div id="starfield"></div>
"""

st.markdown(STARFIELD_CSS, unsafe_allow_html=True)

# =============================
# LOADING SCREEN ONCE
# =============================
if "loaded_once" not in st.session_state:
    spinner_holder = st.empty()
    spinner_holder.markdown("""
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
    spinner_holder.empty()
    st.session_state.loaded_once = True

# =============================
# DATABASE SETUP (DROP MISMATCH TABLES)
# =============================
@st.cache_resource
def init_db():
    conn = sqlite3.connect("nutrivision_app.db", check_same_thread=False)
    c = conn.cursor()
    
    # 1. Check 'users' table
    c.execute("PRAGMA table_info(users)")
    user_cols = [col[1] for col in c.fetchall()]
    desired_user_cols = ["id","username","password","height","weight","age","gender","profile_pic"]
    # if mismatch, drop
    if sorted(user_cols) != sorted(desired_user_cols):
        c.execute("DROP TABLE IF EXISTS users")

    # 2. Check 'meals' table
    c.execute("PRAGMA table_info(meals)")
    meal_cols = [col[1] for col in c.fetchall()]
    desired_meal_cols = ["id","user_id","meal_time","source","caption","predicted","calories","meal_image"]
    if sorted(meal_cols) != sorted(desired_meal_cols):
        c.execute("DROP TABLE IF EXISTS meals")

    # Recreate them if missing
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
# HELPER FUNCTIONS
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

def get_meal_history(uid):
    c = conn.cursor()
    c.execute("""
        SELECT meal_time, source, caption, predicted, calories, meal_image
        FROM meals
        WHERE user_id=?
        ORDER BY meal_time DESC
    """, (uid,))
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
    c.execute("""
        SELECT DATE(meal_time), SUM(calories)
        FROM meals
        WHERE user_id=?
        GROUP BY DATE(meal_time)
        ORDER BY DATE(meal_time)
    """, (uid,))
    return c.fetchall()

def save_uploaded_file(upfile, folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    file_path = os.path.join(folder, upfile.name)
    with open(file_path, "wb") as f:
        f.write(upfile.getbuffer())
    return file_path

# =============================
# MODEL
# =============================
class NutriVisionNetMultiHead(nn.Module):
    def __init__(self, food_dim=3, fv_dim=9, fast_dim=8, device="cuda", fine_tune_clip=False):
        super().__init__()
        self.device = device
        from transformers import CLIPProcessor, CLIPModel, BlipProcessor, BlipForConditionalGeneration
        self.clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_model=CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
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
        hidden=512
        self.food_head= nn.Sequential(
            nn.Linear(fuse_dim, hidden), nn.ReLU(), nn.Dropout(0.2), nn.Linear(hidden, food_dim)
        )
        self.fv_head= nn.Sequential(
            nn.Linear(fuse_dim, hidden), nn.ReLU(), nn.Dropout(0.2), nn.Linear(hidden, fv_dim)
        )
        self.fast_head= nn.Sequential(
            nn.Linear(fuse_dim, hidden), nn.ReLU(), nn.Dropout(0.2), nn.Linear(hidden, fast_dim)
        )

    def forward(self, img, source):
        # BLIP caption
        ip = self.blip_proc(images=img, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out_ids= self.blip_model.generate(**ip, max_length=50, num_beams=5)
        caption= self.blip_proc.decode(out_ids[0], skip_special_tokens=True)

        # CLIP embeddings
        clip_in= self.clip_proc(images=img, return_tensors="pt").to(self.device)
        img_emb= self.clip_model.get_image_features(**clip_in)
        txt_in= self.clip_proc(text=[caption], return_tensors="pt").to(self.device)
        txt_emb= self.clip_model.get_text_features(**txt_in)
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
    dev= "cuda" if torch.cuda.is_available() else "cpu"
    net= NutriVisionNetMultiHead(device=dev, fine_tune_clip=False).to(dev)
    ckpt="nutrivision_multihand.pt"
    if os.path.exists(ckpt):
        sd= torch.load(ckpt, map_location=dev)
        net.load_state_dict(sd)
    net.eval()
    return net, dev

def run_inference(img, src, net, dev):
    with torch.no_grad():
        arr, capp= net(img, src)
    cals= arr[0] if arr else 0
    return arr, capp, cals

# =============================
# SESSION STATE
# =============================
if "logged_in" not in st.session_state:  st.session_state.logged_in=False
if "user_id" not in st.session_state:    st.session_state.user_id=None
if "username" not in st.session_state:   st.session_state.username=""
if "user_info" not in st.session_state:  st.session_state.user_info={}
if "preferred_diet" not in st.session_state: st.session_state.preferred_diet="Not specified"

# =============================
# LOGIN / REGISTER Toggle
# =============================
def show_login_form():
    st.markdown("<h2 class='app-subtitle'>Login</h2>", unsafe_allow_html=True)
    user= st.text_input("Username", key="login_user")
    pw=   st.text_input("Password", type="password", key="login_pw")
    if st.button("Log In"):
        uid= login_user(user, pw)
        if uid:
            st.session_state.logged_in= True
            st.session_state.user_id= uid
            st.session_state.username= user
            c= conn.cursor()
            c.execute("SELECT height, weight, age, gender, profile_pic FROM users WHERE id=?", (uid,))
            row= c.fetchone()
            st.session_state.user_info={
                "height": row[0],
                "weight": row[1],
                "age": row[2],
                "gender": row[3],
                "profile_pic": row[4]
            }
            st.session_state.preferred_diet= "Not specified"
            st.success("Logged in!")
        else:
            st.error("Invalid credentials.")

def show_register_form():
    st.markdown("<h2 class='app-subtitle'>Register</h2>", unsafe_allow_html=True)
    reg_user= st.text_input("Username", key="reg_user")
    reg_pw=   st.text_input("Password", type="password", key="reg_pw")
    reg_h=    st.number_input("Height (cm) [Optional]",0.0,300.0,step=0.1)
    reg_w=    st.number_input("Weight (kg) [Optional]",0.0,300.0,step=0.1)
    reg_a=    st.number_input("Age [Optional]",0,120,step=1)
    reg_g=    st.selectbox("Gender [Optional]",["","Male","Female","Other"])
    reg_pd=   st.text_input("Preferred Diet [Optional]")
    reg_pic=  st.file_uploader("Profile Picture [Optional]", type=["jpg","jpeg","png"])

    if st.button("Register"):
        pic_path= ""
        if reg_pic:
            pic_path= save_uploaded_file(reg_pic,"profile_pics")
        if reg_user=="" or reg_pw=="":
            st.error("Username & Password required!")
        else:
            succ,msg, uid= register_user(
                reg_user, reg_pw,
                reg_h if reg_h>0 else None,
                reg_w if reg_w>0 else None,
                reg_a if reg_a>0 else None,
                reg_g if reg_g else None,
                pic_path
            )
            if succ:
                st.success(msg)
                st.session_state.logged_in= True
                st.session_state.user_id= uid
                st.session_state.username= reg_user
                st.session_state.user_info={
                    "height": reg_h if reg_h>0 else None,
                    "weight": reg_w if reg_w>0 else None,
                    "age": reg_a if reg_a>0 else None,
                    "gender": reg_g if reg_g else None,
                    "profile_pic": pic_path
                }
                st.session_state.preferred_diet= reg_pd if reg_pd else "Not specified"
                st.query_params= {"user_id":[str(uid)], "username":[reg_user]}
                st.success("Registered & logged in!")
            else:
                st.error(msg)

if not st.session_state.logged_in:
    st.markdown("<h1 class='app-title'>NutriVision</h1>", unsafe_allow_html=True)
    st.write("Toggle between login and register below:")
    auth_choice= st.radio("Select Mode", ["Login","Register"], horizontal=True)
    if auth_choice=="Login":
        show_login_form()
    else:
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
    today= datetime.date.today()
    cals_today= get_daily_cals(st.session_state.user_id, today)
    st.metric("Today's Calorie Intake", f"{cals_today:.2f} kcal")

    if st.button("Refresh Dashboard"):
        # user can manually refresh the page or do nothing
        pass

    all_cals= get_all_daily_cals(st.session_state.user_id)
    if all_cals:
        import pandas as pd
        df= pd.DataFrame(all_cals, columns=["Date","Calories"])
        fig= px.line(df, x="Date", y="Calories", title="Daily Calorie Intake", markers=True)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("No meal records available.")

# UPLOAD TAB
with tabs[1]:
    st.markdown("<h1 class='app-title'>Upload a Meal</h1>", unsafe_allow_html=True)
    st.write("Use the AI model to analyze your meal.")
    cat_map= {"Food Nutrition":"food_nutrition","Fruits & Vegetables":"fv","Fast Food":"fastfood"}
    chosen_cat= st.selectbox("Meal Category", list(cat_map.keys()))
    source= cat_map[chosen_cat]

    upload_file= st.file_uploader("Choose a meal image (JPG, PNG)", type=["jpg","jpeg","png"])
    if upload_file:
        try:
            meal_img= Image.open(upload_file).convert("RGB")
            # Show at 50% max height
            buf= io.BytesIO()
            meal_img.save(buf, format="PNG")
            b64= base64.b64encode(buf.getvalue()).decode()
            st.markdown(f"<div class='upload-image'><img src='data:image/png;base64,{b64}'/></div>", unsafe_allow_html=True)
        except:
            st.error("Couldn't read image.")
        if st.button("Analyze Meal"):
            with st.spinner("Analyzing..."):
                net, dev= load_model()
                preds, capp, cals= run_inference(meal_img, source, net, dev)
            st.success("Analysis complete!")
            st.write("**Caption**:", capp)
            if source=="food_nutrition":
                columns= ["Caloric Value","Fat","Carbohydrates"]
            elif source=="fv":
                columns= ["energy (kcal/kJ)","water (g)","protein (g)","total fat (g)","carbohydrates (g)","fiber (g)","sugars (g)","calcium (mg)","iron (mg)"]
            else:
                columns= ["calories","cal_fat","total_fat","sat_fat","trans_fat","cholesterol","sodium","total_carb"]
            st.table({"Nutrient": columns, "Value": [round(x,2) for x in preds]})
            st.write("**Predicted Calories**:", f"{cals:.2f}")
            saved_path= save_uploaded_file(upload_file,"meal_images")
            store_meal(st.session_state.user_id, source, capp, preds, cals, saved_path)
            st.success("Meal stored in history!")

# HISTORY TAB
with tabs[2]:
    st.markdown("<h1 class='app-title'>Meal History</h1>", unsafe_allow_html=True)
    if st.session_state.user_id is None:
        st.write("No user. Please log in again.")
    else:
        meal_hist= get_meal_history(st.session_state.user_id)
        if meal_hist:
            for record in meal_hist:
                meal_time, meal_src, meal_cap, meal_pred, meal_cals, meal_path= record
                st.write(f"**Time**: {meal_time} | **Category**: {meal_src}")
                if meal_path and os.path.exists(meal_path):
                    pic= Image.open(meal_path)
                    buff= io.BytesIO()
                    pic.save(buff, format="PNG")
                    b64= base64.b64encode(buff.getvalue()).decode()
                    st.markdown(f"<div class='history-image'><img src='data:image/png;base64,{b64}'/></div>", unsafe_allow_html=True)
                st.write(f"**Caption**: {meal_cap}")
                st.write(f"**Calories**: {meal_cals:.2f}")
                try:
                    arr= json.loads(meal_pred)
                    if meal_src=="food_nutrition":
                        cC= ["Caloric Value","Fat","Carbohydrates"]
                    elif meal_src=="fv":
                        cC= ["energy (kcal/kJ)","water (g)","protein (g)","total fat (g)","carbohydrates (g)","fiber (g)","sugars (g)","calcium (mg)","iron (mg)"]
                    else:
                        cC= ["calories","cal_fat","total_fat","sat_fat","trans_fat","cholesterol","sodium","total_carb"]
                    st.table({"Nutrient": cC, "Value": [round(val,2) for val in arr]})
                except:
                    st.write("Raw predictions:", meal_pred)
                st.markdown("---")
        else:
            st.write("No meals recorded yet.")

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
    st.write(f"**Username**: {st.session_state.username}")
    userinfo= st.session_state.user_info
    if userinfo:
        st.write(f"**Height**: {userinfo.get('height','N/A')} cm")
        st.write(f"**Weight**: {userinfo.get('weight','N/A')} kg")
        st.write(f"**Age**: {userinfo.get('age','N/A')}")
        st.write(f"**Gender**: {userinfo.get('gender','N/A')}")
        if userinfo.get('height') and userinfo.get('weight') and userinfo['height']>0:
            bmi= userinfo['weight']/((userinfo['height']/100)**2)
            st.write(f"**BMI**: {bmi:.2f}")
        st.write(f"**Preferred Diet**: {st.session_state.preferred_diet}")
        pic= userinfo.get("profile_pic","")
        if pic and os.path.exists(pic):
            with open(pic, "rb") as f:
                encoded= base64.b64encode(f.read()).decode()
            ext= os.path.splitext(pic)[1].lower()
            mime= "image/png" if ext==".png" else "image/jpeg"
            st.markdown(
                f"<div style='text-align:left;'><img src='data:{mime};base64,{encoded}' style='max-height:40vh;'/></div>",
                unsafe_allow_html=True
            )
        else:
            st.write("No profile picture.")
    else:
        st.write("No user info available.")
    st.markdown("---")
    st.subheader("Update Profile")
    new_ht= st.number_input("Height (cm)", 0.0,300.0, step=0.1, value=float(userinfo.get('height') or 0))
    new_wt= st.number_input("Weight (kg)", 0.0,300.0, step=0.1, value=float(userinfo.get('weight') or 0))
    new_ag= st.number_input("Age", 0,120, step=1, value=int(userinfo.get('age') or 0))
    new_gd= st.selectbox("Gender", ["","Male","Female","Other"], index=0 if not userinfo.get('gender') else ["","Male","Female","Other"].index(userinfo['gender']))
    new_pd= st.text_input("Preferred Diet", st.session_state.preferred_diet)
    upf= st.file_uploader("Update Profile Pic", type=["jpg","jpeg","png"], key="upd_pic")
    if st.button("Save Profile"):
        c= conn.cursor()
        c.execute("UPDATE users SET height=?, weight=?, age=?, gender=? WHERE id=?",
                  (new_ht if new_ht>0 else None,
                   new_wt if new_wt>0 else None,
                   new_ag if new_ag>0 else None,
                   new_gd if new_gd else None,
                   st.session_state.user_id))
        conn.commit()
        new_path= userinfo.get('profile_pic','')
        if upf:
            new_path= save_uploaded_file(upf,"profile_pics")
            c.execute("UPDATE users SET profile_pic=? WHERE id=?", (new_path, st.session_state.user_id))
            conn.commit()
        st.session_state.user_info['height']= new_ht if new_ht>0 else None
        st.session_state.user_info['weight']= new_wt if new_wt>0 else None
        st.session_state.user_info['age']= new_ag if new_ag>0 else None
        st.session_state.user_info['gender']= new_gd if new_gd else None
        st.session_state.preferred_diet= new_pd if new_pd else "Not specified"
        if new_path:
            st.session_state.user_info['profile_pic']= new_path
        st.success("Profile updated successfully!")

# LOGOUT TAB
with tabs[4]:
    st.markdown("<h1 class='app-title'>Logout</h1>", unsafe_allow_html=True)
    if st.button("Confirm Logout"):
        st.session_state.logged_in=False
        st.session_state.user_id=None
        st.session_state.username=""
        st.session_state.user_info={}
        st.session_state.preferred_diet="Not specified"
        # Clear query params
        st.query_params={}
        st.success("You have been logged out.")
        st.info("Go to the Login/Register toggle above or simply refresh to log in again.")
        st.button("Ok")
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
from huggingface_hub import hf_hub_download
from transformers import CLIPProcessor, CLIPModel, BlipProcessor, BlipForConditionalGeneration

####################################
# PAGE CONFIG & THEME
####################################
st.set_page_config(page_title="NutriVision", layout="wide")

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
/* Tabs styling - advanced look */
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

####################################
# LOADING SCREEN ONCE
####################################
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

####################################
# DATABASE SETUP (DROP MISMATCH TABLES)
####################################
@st.cache_resource
def init_db():
    conn = sqlite3.connect("nutrivision_app.db", check_same_thread=False)
    c = conn.cursor()
    
    # Check 'users' table
    c.execute("PRAGMA table_info(users)")
    user_cols = [col[1] for col in c.fetchall()]
    desired_users = ["id","username","password","height","weight","age","gender","profile_pic"]
    if sorted(user_cols) != sorted(desired_users):
        c.execute("DROP TABLE IF EXISTS users")

    # Check 'meals' table
    c.execute("PRAGMA table_info(meals)")
    meal_cols = [col[1] for col in c.fetchall()]
    desired_meals = ["id","user_id","meal_time","source","caption","predicted","calories","meal_image"]
    if sorted(meal_cols) != sorted(desired_meals):
        c.execute("DROP TABLE IF EXISTS meals")

    # Recreate if missing
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

####################################
# DB HELPERS
####################################
def register_user(username, password, height, weight, age, gender, pic_path):
    c = conn.cursor()
    try:
        c.execute("""
            INSERT INTO users (username, password, height, weight, age, gender, profile_pic)
            VALUES (?,?,?,?,?,?,?)
        """,(username, password, height, weight, age, gender, pic_path))
        conn.commit()
        c.execute("SELECT id FROM users WHERE username=?",(username,))
        row= c.fetchone()
        return True, "Registration successful!", row[0]
    except sqlite3.IntegrityError:
        return False, "Username already exists.", None

def login_user(username, password):
    c= conn.cursor()
    c.execute("SELECT id FROM users WHERE username=? AND password=?",(username, password))
    row= c.fetchone()
    return row[0] if row else None

def store_meal(uid, source, caption, preds, cals, path):
    c= conn.cursor()
    c.execute("""
        INSERT INTO meals (user_id, meal_time, source, caption, predicted, calories, meal_image)
        VALUES (?,?,?,?,?,?,?)
    """,(uid, datetime.datetime.now(), source, caption, json.dumps(preds), cals, path))
    conn.commit()

def get_meal_history(uid):
    c= conn.cursor()
    c.execute("""
    SELECT meal_time, source, caption, predicted, calories, meal_image
    FROM meals
    WHERE user_id=?
    ORDER BY meal_time DESC
    """,(uid,))
    return c.fetchall()

def get_daily_cals(uid, date):
    c= conn.cursor()
    start= datetime.datetime.combine(date, datetime.time.min)
    end= datetime.datetime.combine(date, datetime.time.max)
    c.execute("""SELECT SUM(calories) FROM meals
                 WHERE user_id=? AND meal_time BETWEEN ? AND ?""",(uid, start, end))
    val= c.fetchone()[0]
    return val if val else 0

def get_all_daily_cals(uid):
    c= conn.cursor()
    c.execute("""SELECT DATE(meal_time), SUM(calories)
                 FROM meals
                 WHERE user_id=?
                 GROUP BY DATE(meal_time)
                 ORDER BY DATE(meal_time)
              """,(uid,))
    return c.fetchall()

def save_uploaded_file(file, folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    path= os.path.join(folder, file.name)
    with open(path,"wb") as f:
        f.write(file.getbuffer())
    return path

####################################
# QUANTITY PARSER
####################################
NUM_WORDS= {
    "zero":0,"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,"eight":8,
    "nine":9,"ten":10,"eleven":11,"twelve":12,"thirteen":13,"fourteen":14,"fifteen":15,
    "twenty":20,"thirty":30,"forty":40,"fifty":50
}
def parse_caption_items_naive(caption, item_map):
    tokens= caption.lower().split()
    results={}
    i=0
    while i<len(tokens):
        word= tokens[i].strip(".,!?")
        qty=1
        # Try digit or spelled number
        if word.isdigit():
            qty= int(word)
            i+=1
            if i<len(tokens):
                item= tokens[i].strip(".,!?")
                i+=1
            else:
                break
        elif word in NUM_WORDS:
            qty= NUM_WORDS[word]
            i+=1
            if i<len(tokens):
                item= tokens[i].strip(".,!?")
                i+=1
            else:
                break
        else:
            item= word
            i+=1
        if item in item_map:
            calsEach= item_map[item][0]
            if item not in results:
                results[item]=(0,calsEach)
            oldQ, cVal= results[item]
            results[item]=(oldQ+qty, cVal)
        else:
            pass
    return results

####################################
# MODEL (Heads match old checkpoint)
####################################
class NutriVisionNetMultiHead(nn.Module):
    def __init__(self, food_dim=3, fv_dim=9, fast_dim=8, device="cuda", fine_tune_clip=False):
        super().__init__()
        self.device= device
        self.clip_proc= CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_model= CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        for p in self.clip_model.text_model.parameters():
            p.requires_grad=False
        if not fine_tune_clip:
            for p in self.clip_model.vision_model.parameters():
                p.requires_grad=False

        self.blip_proc= BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.blip_model= BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        for p in self.blip_model.parameters():
            p.requires_grad=False

        fuse_dim=1024
        hidden=512
        self.food_nutrition_head= nn.Sequential(
            nn.Linear(fuse_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, food_dim)
        )
        self.fv_head= nn.Sequential(
            nn.Linear(fuse_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, fv_dim)
        )
        self.fastfood_head= nn.Sequential(
            nn.Linear(fuse_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, fast_dim)
        )

    def forward(self, image, source):
        bip= self.blip_proc(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out_ids= self.blip_model.generate(**bip, max_length=50, num_beams=5)
        caption= self.blip_proc.decode(out_ids[0], skip_special_tokens=True)

        c_in= self.clip_proc(images=image, return_tensors="pt").to(self.device)
        img_emb= self.clip_model.get_image_features(**c_in)
        txt_in= self.clip_proc(text=[caption], return_tensors="pt").to(self.device)
        txt_emb= self.clip_model.get_text_features(**txt_in)
        img_emb= img_emb / img_emb.norm(p=2, dim=-1, keepdim=True)
        txt_emb= txt_emb / txt_emb.norm(p=2, dim=-1, keepdim=True)
        fused= torch.cat([img_emb, txt_emb], dim=-1)

        if source=="food_nutrition":
            out= self.food_nutrition_head(fused)
        elif source=="fv":
            out= self.fv_head(fused)
        elif source=="fastfood":
            out= self.fastfood_head(fused)
        else:
            raise ValueError("Unknown source.")
        return out, caption

@st.cache_resource
def load_model():
    device= "cuda" if torch.cuda.is_available() else "cpu"
    net= NutriVisionNetMultiHead(3,9,8, device=device, fine_tune_clip=False).to(device)
    # Pull the checkpoint from HuggingFace Hub
    ckpt_path= hf_hub_download(
        repo_id="Angshuman28/NutriVision",
        filename="nutrivision_multihand.pt"
    )
    if os.path.exists(ckpt_path):
        sd= torch.load(ckpt_path, map_location=device)
        net.load_state_dict(sd)
    net.eval()
    return net, device

####################################
# APP STATE
####################################
if "logged_in" not in st.session_state:  st.session_state.logged_in=False
if "user_id" not in st.session_state:    st.session_state.user_id=None
if "username" not in st.session_state:   st.session_state.username=""
if "user_info" not in st.session_state:  st.session_state.user_info={}
if "preferred_diet" not in st.session_state: st.session_state.preferred_diet="Not specified"

####################################
# FORMS
####################################
def login_form():
    st.markdown("<h2 class='app-subtitle'>Login</h2>", unsafe_allow_html=True)
    user= st.text_input("Username",key="login_user")
    pw=   st.text_input("Password",type="password",key="login_pw")
    if st.button("Log In"):
        uid= login_user(user,pw)
        if uid:
            st.session_state.logged_in= True
            st.session_state.user_id= uid
            st.session_state.username= user
            c= conn.cursor()
            c.execute("SELECT height,weight,age,gender,profile_pic FROM users WHERE id=?",(uid,))
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
            st.button("Continue")
        else:
            st.error("Invalid credentials.")

def register_form():
    st.markdown("<h2 class='app-subtitle'>Register</h2>", unsafe_allow_html=True)
    r_user= st.text_input("Username",key="reg_user")
    r_pw=   st.text_input("Password",type="password",key="reg_pw")
    r_h=    st.number_input("Height (cm) [Optional]",0.0,300.0,step=0.1)
    r_w=    st.number_input("Weight (kg) [Optional]",0.0,300.0,step=0.1)
    r_a=    st.number_input("Age [Optional]",0,120,step=1)
    r_g=    st.selectbox("Gender [Optional]",["","Male","Female","Other"])
    r_pd=   st.text_input("Preferred Diet [Optional]",key="reg_pd")
    r_pic=  st.file_uploader("Profile Picture [Optional]",type=["jpg","jpeg","png"],key="reg_pic")

    if st.button("Register"):
        pic= ""
        if r_pic:
            pic= save_uploaded_file(r_pic,"profile_pics")
        if r_user=="" or r_pw=="":
            st.error("Username & Password required!")
        else:
            succ,msg,uid= register_user(
                r_user,r_pw,
                r_h if r_h>0 else None,
                r_w if r_w>0 else None,
                r_a if r_a>0 else None,
                r_g if r_g else None,
                pic
            )
            if succ:
                st.success(msg)
                st.session_state.logged_in= True
                st.session_state.user_id= uid
                st.session_state.username= r_user
                st.session_state.user_info={
                    "height": r_h if r_h>0 else None,
                    "weight": r_w if r_w>0 else None,
                    "age": r_a if r_a>0 else None,
                    "gender": r_g if r_g else None,
                    "profile_pic": pic
                }
                st.session_state.preferred_diet= r_pd if r_pd else "Not specified"
                st.query_params= {"user_id":[str(uid)], "username":[r_user]}
                st.success("Registered & logged in!")
                st.button("Continue")
            else:
                st.error(msg)

if not st.session_state.logged_in:
    st.markdown("<h1 class='app-title'>NutriVision</h1>", unsafe_allow_html=True)
    st.write("Toggle between login and register:")
    mode= st.radio("Select Mode",["Login","Register"], horizontal=True)
    if mode=="Login":
        login_form()
    else:
        register_form()
    st.stop()

####################################
# TABS
####################################
tabs= st.tabs(["Home","Upload Meal","Meal History","Account","Logout"])

####################################
# TAB 0: HOME
####################################
with tabs[0]:
    st.markdown("<h1 class='app-title'>Dashboard</h1>",unsafe_allow_html=True)
    st.write(f"Hello, **{st.session_state.username}**!")
    today= datetime.date.today()
    cals= get_daily_cals(st.session_state.user_id, today)
    st.metric("Today's Calorie Intake", f"{cals:.2f} kcal")

    if st.button("Refresh Dashboard"):
        pass

    alld= get_all_daily_cals(st.session_state.user_id)
    if alld:
        import pandas as pd
        df= pd.DataFrame(alld, columns=["Date","Calories"])
        fig= px.line(df, x="Date", y="Calories", title="Daily Calorie Intake", markers=True)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("No meal records to show.")

####################################
# TAB 1: UPLOAD
####################################
# Provide option for file upload or camera
ITEM_MAP= {
   "fish":[100.0],"lemon":[15.0],"lemons":[15.0],"herb":[5.0],"herbs":[5.0],
   "bread":[80.0],"tomato":[20.0],"cheese":[70.0],"rice":[90.0],"egg":[70.0]
}

with tabs[1]:
    st.markdown("<h1 class='app-title'>Upload a Meal</h1>", unsafe_allow_html=True)
    st.write("Analyze your meal with AI.")

    cat_map= {
      "Food Nutrition":"food_nutrition",
      "Fruits & Vegetables":"fv",
      "Fast Food":"fastfood"
    }
    s_cat= st.selectbox("Meal Category", list(cat_map.keys()))
    source= cat_map[s_cat]

    image_choice= st.radio("Choose Image Source",["Upload File","Camera"])
    captured_image= None
    if image_choice=="Upload File":
        up_file= st.file_uploader("Select a meal image (JPG/PNG)", type=["jpg","jpeg","png"])
        if up_file:
            captured_image= up_file
    else:
        # st.camera_input returns an UploadedFile or None
        cam= st.camera_input("Take a meal photo")
        if cam:
            captured_image= cam

    if captured_image:
        try:
            meal_img= Image.open(captured_image).convert("RGB")
            bf= io.BytesIO()
            meal_img.save(bf, format="PNG")
            b64= base64.b64encode(bf.getvalue()).decode()
            st.markdown(f"<div class='upload-image'><img src='data:image/png;base64,{b64}'/></div>", unsafe_allow_html=True)
        except:
            st.error("Error reading image.")
        
        if st.button("Analyze Meal"):
            with st.spinner("Performing inference..."):
                model, dev= load_model()
                raw_preds, caption= model(meal_img, source)
            st.success("Inference completed!")
            
            st.write("**Generated Caption**:", caption)
            # parse items from the caption
            parsed= parse_caption_items_naive(caption, ITEM_MAP)
            tot=0.0
            lines=[]
            for nm, (q, cE) in parsed.items():
                line_cal= q*cE
                tot+= line_cal
                lines.append((nm,q,cE,line_cal))
            if lines:
                import pandas as pd
                df= pd.DataFrame(lines, columns=["Item","Qty","Cals/Item","Subtotal"])
                st.table(df)
                st.write("**Parsed Items Total**:", f"{tot:.2f}")
            else:
                st.write("No recognized item-quantity combos from the caption.")

            # The regression head numeric
            if len(raw_preds)>0:
                ai_cal= float(raw_preds[0][0].item())
            else:
                ai_cal=0.0
            st.write("**AI Calories**:", f"{ai_cal:.2f}")

            # Choose final cals
            final_cals= ai_cal
            # Save in DB
            path= save_uploaded_file(captured_image,"meal_images")
            store_meal(st.session_state.user_id, source, caption, raw_preds.tolist(), final_cals, path)
            st.success("Meal recorded successfully!")

####################################
# TAB 2: MEAL HISTORY
####################################
with tabs[2]:
    st.markdown("<h1 class='app-title'>Meal History</h1>", unsafe_allow_html=True)
    if st.session_state.user_id is None:
        st.write("No user. Please login again.")
    else:
        meals= get_meal_history(st.session_state.user_id)
        if meals:
            for rec in meals:
                meal_time, meal_src, meal_cap, meal_pred, meal_cals, meal_im= rec
                st.write(f"**Time**: {meal_time}, **Category**: {meal_src}")
                if meal_im and os.path.exists(meal_im):
                    ig= Image.open(meal_im)
                    bb= io.BytesIO()
                    ig.save(bb,format="PNG")
                    b64= base64.b64encode(bb.getvalue()).decode()
                    st.markdown(f"<div class='history-image'><img src='data:image/png;base64,{b64}'/></div>",unsafe_allow_html=True)
                st.write(f"**Caption**: {meal_cap}")
                st.write(f"**Calories**: {meal_cals:.2f}")
                try:
                    arr= json.loads(meal_pred)
                    st.write("**Raw Regression**:", arr)
                except:
                    st.write("pred:", meal_pred)
                st.markdown("---")
        else:
            st.write("No meals recorded yet.")

        dd= get_all_daily_cals(st.session_state.user_id)
        if dd:
            import pandas as pd
            ddf= pd.DataFrame(dd, columns=["Date","Calories"])
            fig= px.bar(ddf, x="Date", y="Calories", title="Daily Calorie Intake")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("No daily data available.")

####################################
# TAB 3: ACCOUNT
####################################
with tabs[3]:
    st.markdown("<h1 class='app-title'>Account Info</h1>",unsafe_allow_html=True)
    st.write(f"**Username**: {st.session_state.username}")
    usr= st.session_state.user_info
    if usr:
        st.write(f"**Height**: {usr.get('height','N/A')} cm")
        st.write(f"**Weight**: {usr.get('weight','N/A')} kg")
        st.write(f"**Age**: {usr.get('age','N/A')}")
        st.write(f"**Gender**: {usr.get('gender','N/A')}")
        if usr.get('height') and usr.get('weight') and usr['height']>0:
            bmi= usr['weight']/((usr['height']/100)**2)
            st.write(f"**BMI**: {bmi:.2f}")
        st.write(f"**Preferred Diet**: {st.session_state.preferred_diet}")
        p= usr.get("profile_pic","")
        if p and os.path.exists(p):
            with open(p,"rb") as f:
                enc= base64.b64encode(f.read()).decode()
            ext= os.path.splitext(p)[1].lower()
            mime= "image/png" if ext==".png" else "image/jpeg"
            st.markdown(
                f"<div style='text-align:left;'><img src='data:{mime};base64,{enc}' style='max-height:40vh;'/></div>",
                unsafe_allow_html=True
            )
        else:
            st.write("No profile picture.")
    else:
        st.write("No user info available.")
    st.markdown("---")
    st.subheader("Update Profile")
    new_ht= st.number_input("Height (cm)",0.0,300.0, step=0.1, value=float(usr.get('height') or 0))
    new_wt= st.number_input("Weight (kg)",0.0,300.0, step=0.1, value=float(usr.get('weight') or 0))
    new_ag= st.number_input("Age",0,120, step=1, value=int(usr.get('age') or 0))
    new_gd= st.selectbox("Gender",["","Male","Female","Other"], index=0 if not usr.get('gender') else ["","Male","Female","Other"].index(usr['gender']))
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
        new_path= usr.get('profile_pic','')
        if upf:
            new_path= save_uploaded_file(upf,"profile_pics")
            c.execute("UPDATE users SET profile_pic=? WHERE id=?",(new_path, st.session_state.user_id))
            conn.commit()
        st.session_state.user_info['height']= new_ht if new_ht>0 else None
        st.session_state.user_info['weight']= new_wt if new_wt>0 else None
        st.session_state.user_info['age']= new_ag if new_ag>0 else None
        st.session_state.user_info['gender']= new_gd if new_gd else None
        st.session_state.preferred_diet= new_pd if new_pd else "Not specified"
        if new_path:
            st.session_state.user_info['profile_pic']= new_path
        st.success("Profile updated successfully!")

####################################
# TAB 4: LOGOUT
####################################
with tabs[4]:
    st.markdown("<h1 class='app-title'>Logout</h1>",unsafe_allow_html=True)
    if st.button("Confirm Logout"):
        st.session_state.logged_in= False
        st.session_state.user_id= None
        st.session_state.username= ""
        st.session_state.user_info= {}
        st.session_state.preferred_diet= "Not specified"
        st.query_params= {}
        st.success("You have been logged out.")
        st.info("Go to Login/Register toggle or refresh.")
        st.button("Ok")

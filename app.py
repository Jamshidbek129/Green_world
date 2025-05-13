import streamlit as st
from fastai.vision.all import *
from pathlib import Path
import plotly.express as px
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

st.title("O'simliklardagi kasallik turlari")
model_path=Path("model.pkl")
if model_path.exists():
  model=load_learner(model_path)
else:
  st.error("Model fayli topilmadi")

fayl=st.file_uploader("Rasmni yuklash: ", type=['png', 'jpg', 'jpeg'])

if fayl:
  rasm=PILImage.create(fayl)
  bashorat, id, ehtimollik = model.predict(rasm)
  st.image(rasm)
  st.success(f"Bu kasallik {bashorat} bo'lishi mumkin!")
  st.info(f"Ehtimolligi: {ehtimollik[id]*100:.1f}%")
  fig=px.bar(x=model.dls.vocab, y=ehtimollik*100)
  st.plotly_chart(fig)

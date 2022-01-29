import streamlit as st
from multiapp import MultiApp
from apps import forestpractice # import your app modules here
from apps import SVR
# from apps import KNN

from apps import RLogistica
# from apps import SVC
# from apps import NewModel

app = MultiApp()

app.add_app("RANDOM FOREST CLASSIFIER", forestpractice.app)
app.add_app("SUPPORT VECTOR REGRESION", SVR.app)
# app.add_app("KNN", KNN.app)

app.add_app("REGRESION LOGISTICA", RLogistica.app)
# app.add_app("SVC", SVC.app)
# app.add_app("PROPHET", NewModel.app)

# The main app
app.run()

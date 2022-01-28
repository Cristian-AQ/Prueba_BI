import streamlit as st
from multiapp import MultiApp
from apps import forestpractice, modeloNew # import your app modules here

app = MultiApp()

app.add_app("RandomForestClassifier", forestpractice.app)
app.add_app("ModeloNew", forestpractice.app)

# The main app
app.run()

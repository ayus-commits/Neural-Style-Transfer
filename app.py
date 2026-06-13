import streamlit as st
import tempfile
import os
from PIL import Image
import threading
import time
# from datetime import datetime
from main import generate_style_transfer

st.set_page_config(
    page_title="Neural Style Transfer",
    layout="wide"
)
if "running" not in st.session_state:
    st.session_state.running = False

st.title("🎨 Neural Style Transfer")
st.write("Upload a content image and a style image, then generate a stylized result.")

# Uploads -------------------------------------------------------------------------
content_file = st.file_uploader("Content Image",type=["jpg", "jpeg", "png"])
style_file = st.file_uploader("Style Image", type=["jpg", "jpeg", "png"])

# Config options-------------------------------------------------------------------------

# config_name = st.text_input("Config",value="default")
# checkpoint_name = st.text_input("Checkpoint",value="imagenette")
# output_dir = st.text_input("Output Directory",value="./output")
# output_name = st.text_input("Output Folder Name",value="streamlit_run")

with st.sidebar:

    config_name = st.selectbox("Config",["default", "default_512", "custom", "exp1", "exp2","exp3","exp4","exp5","exp6","exp7"],index=0)
    if config_name == "custom":
        with st.expander("Custom Hyperparameters",expanded=True):
            image_size = st.slider("Image Size",min_value=64,max_value=1024,value=256,step=64)
            alpha = st.number_input("Content Weight (Alpha)",value=1.0)
            beta_exponent = st.slider("Style Weight (Beta) Exponent",1,10,6)
            beta = 10 ** beta_exponent
            gamma_exponent = st.slider("TV Weight (Gamma) Exponent",-10,-1,-6)
            gamma = 10 ** gamma_exponent
            num_steps = st.slider("Optimization Steps",1,300,50)

    checkpoint_name = st.selectbox("Checkpoint",["imagenette","stl-10"])
    # num_steps = st.slider("Steps",5,300,50)

    output_dir = st.text_input("Output Directory",value="./output")
    output_name = st.text_input("Output Folder Name",value="streamlit_run")

# Preview images-------------------------------------------------------------------------

if content_file and style_file:

    # col1, col2 = st.columns(2)
    _, col1, _ , col2, _ ,col3,_ = st.columns([0.1, 0.2, 0.1, 0.2, 0.1, 0.2, 0.1])
    with col1:
        st.subheader("Content")
        st.image(content_file)

    with col2:
        st.subheader("Style")
        st.image(style_file)
    with col3:
        st.subheader("Generated")
        generated_placeholder = st.empty()

# Generate-------------------------------------------------------------------------

if st.button("Generate Style Transfer",disabled=st.session_state.running):

    if content_file is None or style_file is None:
        st.error("Please upload both images.")
        st.stop()

    run_output_dir = os.path.join(output_dir, output_name)

    os.makedirs(run_output_dir, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".jpg"
    ) as content_temp:

        content_temp.write(content_file.read())
        content_path = content_temp.name

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".jpg"
    ) as style_temp:

        style_temp.write(style_file.read())
        style_path = style_temp.name

    status = {"done": False, "error": None}

    def worker():
        try:
            generate_style_transfer(
                content_path=content_path,
                style_path=style_path,
                output_name=output_name,
                config_name=config_name,
                output_dir=output_dir,
                checkpoint=checkpoint_name,
                image_size=image_size if config_name == "custom" else None,
                alpha=alpha if config_name == "custom" else None,
                beta=beta if config_name == "custom" else None,
                gamma=gamma if config_name == "custom" else None,
                num_steps=num_steps if config_name == "custom" else None,
            )
        except Exception as e:
            status["error"] = str(e)
        finally:
            status["done"] = True

    st.session_state.running = True

    thread = threading.Thread(
        target=worker,
        daemon=True
    )

    thread.start()

    st.subheader("Live Stylization")

    progress_bar = st.progress(0)
    step_text = st.empty()
    loss_text = st.empty()
    # image_placeholder = st.empty()

    while not status["done"]:

        latest_image = os.path.join(run_output_dir, "latest.jpg")
        if os.path.exists(latest_image):
            # image_placeholder.image(latest_image, caption="Live Output")
            # generated_placeholder.image(latest_image, caption="Live Output")
            try:
                img = Image.open(latest_image)
                img.load()          # force full read
                # st.image(img, caption="Live Output")
                generated_placeholder.image(img, caption="Live Output")
            except Exception:
                st.info("⏳ Updating image...")

        progress_file = os.path.join(run_output_dir, "progress.txt")

        if os.path.exists(progress_file):
            try:
                with open(progress_file, "r") as f:
                    step, total_steps, loss = f.read().split(",")
                step = int(step)
                total_steps = int(total_steps)
                loss = float(loss)
                progress_bar.progress(step / total_steps)
                step_text.write(f"### Step {step}/{total_steps}")
                loss_text.write(f"### Loss: {loss:.4f}")
            except:
                pass

        time.sleep(0.5)
    
    st.session_state.running = False

    if status["error"]:
        st.error(f"Style transfer failed: {status['error']}")
        st.stop()

    final_image = os.path.join(
        run_output_dir,
        "final_output.jpg"
    )

    if os.path.exists(final_image):

        generated_placeholder.image(
            final_image,
            caption="Final Result"
        )

        st.success("Style Transfer Complete!")

        progress_bar.progress(1.0)
        step_text.write("### Completed")

        with open(final_image, "rb") as f:
            st.download_button(
                "Download Result",
                f,
                file_name="stylized.jpg"
            )